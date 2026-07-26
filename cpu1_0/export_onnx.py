"""Export a mobile-friendly ONNX model from a training checkpoint."""

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
    parser.add_argument("--output-stride", type=int, default=None, choices=[8, 16])
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


def export_model(model, output, example, opset, dynamic):
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        }
    torch.onnx.export(
        model,
        example,
        output,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
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
        "output": "float32 logits NCHW; argmax channel gives class id 0/1/2",
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
            expected = model(example).cpu().numpy()
        actual = session.run(["logits"], {"input": example.cpu().numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


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
        "output": {
            "name": "logits",
            "shape": ["N", config["num_classes"], "H", "W"] if not opts.fixed_shape
            else [1, config["num_classes"], opts.height, opts.width],
            "postprocess": "argmax over channel dimension; class 1 is handwriting",
        },
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
        export_model(model, output, example, opts.opset, not opts.fixed_shape)
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
