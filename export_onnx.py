"""Export a CPU-compatible ONNX model from a training checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import network
from utils.checkpoint import checkpoint_model_config, clean_state_dict


DEFAULT_CONFIG = {
    "model": "lite_eraser",
    "num_classes": 3,
    "output_stride": 16,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="handwriting_eraser.onnx")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--fixed-shape", action="store_true",
        help="export fixed H/W instead of dynamic axes; useful for strict mobile pipelines",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="skip ONNX Runtime numerical verification",
    )
    parser.add_argument("--model", default=None,
                        help="required only for metadata-free legacy checkpoints")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--output-stride", type=int, default=None, choices=[8, 16, 32])
    return parser


def load_eager_model(opts):
    checkpoint = torch.load(opts.checkpoint, map_location="cpu")
    config = checkpoint_model_config(checkpoint, DEFAULT_CONFIG)
    if opts.model is not None:
        config["model"] = opts.model
    if opts.num_classes is not None:
        config["num_classes"] = opts.num_classes
    if opts.output_stride is not None:
        config["output_stride"] = opts.output_stride

    model = network.modeling.__dict__[config["model"]](
        num_classes=config["num_classes"],
        output_stride=config["output_stride"],
        pretrained_backbone=False,
    )
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(clean_state_dict(state))
    model.eval()
    return model, config


def export_model(model, output, example, opset, dynamic, joint):
    output_names = ["logits", "candidate", "restored"] if joint else ["logits"]
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
        }
        for name in output_names:
            dynamic_axes[name] = {0: "batch", 2: "height", 3: "width"}
    torch.onnx.export(
        model,
        example,
        output,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def check_onnx_file(output, config):
    try:
        import onnx
    except ImportError:
        print("onnx package is not installed; skipped onnx.checker and metadata")
        return

    model = onnx.load(output)
    onnx.checker.check_model(model)
    metadata = {
        "model_config": json.dumps(config, ensure_ascii=False),
        "input": "float32 NCHW normalized RGB image",
        "output": (
            "logits, clean RGB candidate, and safely blended restored RGB"
            if config.get("task") == "joint_restoration"
            else "float32 logits NCHW; argmax channel gives class id 0/1/2"
        ),
    }
    existing = {item.key: item for item in model.metadata_props}
    for key, value in metadata.items():
        item = existing.get(key)
        if item is None:
            item = model.metadata_props.add()
            item.key = key
        item.value = value
    onnx.save(model, output)


def verify_with_onnxruntime(model, output, examples):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for verification, or pass --no-verify"
        ) from exc

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(output), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    for example in examples:
        with torch.inference_mode():
            expected = model(example)
        if not isinstance(expected, (tuple, list)):
            expected = (expected,)
        actual = session.run(None, {"input": example.cpu().numpy()})
        if len(actual) != len(expected):
            raise RuntimeError("ONNX output count differs from eager model")
        for actual_item, expected_item in zip(actual, expected):
            np.testing.assert_allclose(
                actual_item, expected_item.cpu().numpy(), rtol=1e-3, atol=1e-4
            )


def write_sidecar_config(output, config, opts):
    sidecar = Path(output).with_suffix(".json")
    payload = {
        "model_config": config,
        "input": {
            "name": "input",
            "shape": ["N", 3, "H", "W"] if not opts.fixed_shape
            else [1, 3, opts.height, opts.width],
            "dtype": "float32",
            "color": "RGB",
            "normalization": {"mean": config.get("mean"), "std": config.get("std")},
        },
        "outputs": (
            [
                {"name": "logits", "description": "3-class segmentation logits"},
                {"name": "candidate", "description": "unmasked clean RGB prediction in [0,1]"},
                {"name": "restored", "description": "final RGB page in [0,1]"},
            ]
            if config.get("task") == "joint_restoration"
            else [{"name": "logits", "description": "3-class segmentation logits"}]
        ),
    }
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar


def main():
    opts = get_argparser().parse_args()
    torch.set_num_threads(max(1, opts.threads))
    model, config = load_eager_model(opts)
    example = torch.randn(1, 3, opts.height, opts.width, dtype=torch.float32)
    output = Path(opts.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        export_model(
            model,
            output,
            example,
            opts.opset,
            not opts.fixed_shape,
            config.get("task") == "joint_restoration",
        )
    check_onnx_file(output, config)
    if not opts.no_verify:
        examples = [example]
        if not opts.fixed_shape:
            examples.append(
                torch.randn(
                    1, 3, opts.height + 17, opts.width + 31, dtype=torch.float32
                )
            )
        verify_with_onnxruntime(model, output, examples)
    sidecar = write_sidecar_config(output, config, opts)

    print("exported ONNX model: %s" % output)
    print("wrote config: %s" % sidecar)


if __name__ == "__main__":
    main()
