"""Export a portable CPU TorchScript model from a training checkpoint."""

import argparse
import json

import torch

import network
from utils.checkpoint import checkpoint_model_config, clean_state_dict


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="handwriting_eraser_cpu.pt")
    parser.add_argument("--example-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model", default=None,
                        help="required only for metadata-free legacy checkpoints")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--output-stride", type=int, default=None, choices=[8, 16, 32])
    return parser


def main():
    opts = get_argparser().parse_args()
    torch.set_num_threads(max(1, opts.threads))
    checkpoint = torch.load(opts.checkpoint, map_location="cpu")
    config = checkpoint_model_config(
        checkpoint,
        {"model": "lite_eraser", "num_classes": 3, "output_stride": 16,
         "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    )
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

    example = torch.randn(1, 3, opts.example_size, opts.example_size)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=True)
        traced = torch.jit.freeze(traced)
        eager_output = model(example)
        traced_output = traced(example)
        if not torch.allclose(eager_output, traced_output, rtol=1e-4, atol=1e-5):
            raise RuntimeError("TorchScript verification failed")
        # Shape operations must remain dynamic after tracing because pages are
        # neither square nor fixed-resolution in production.
        probe = torch.randn(1, 3, opts.example_size + 17, opts.example_size + 31)
        if not torch.allclose(model(probe), traced(probe), rtol=1e-4, atol=1e-5):
            raise RuntimeError("TorchScript dynamic-shape verification failed")
    extra = {"config.json": json.dumps(config, ensure_ascii=False)}
    torch.jit.save(traced, opts.output, _extra_files=extra)
    print("exported CPU TorchScript model: %s" % opts.output)


if __name__ == "__main__":
    main()
