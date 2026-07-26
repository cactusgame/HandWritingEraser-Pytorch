"""CPU-first, overlap-tiled inference for handwriting removal."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm

import network
from utils.checkpoint import checkpoint_model_config, clean_state_dict


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="image or directory")
    parser.add_argument("--output", required=True, help="output image or directory")
    parser.add_argument("--checkpoint", "--ckpt", required=True,
                        help="training checkpoint (.pth) or TorchScript model (.pt)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--output-stride", type=int, default=None, choices=[8, 16])
    parser.add_argument("--handwriting-class", type=int, default=1)
    parser.add_argument("--dilate", type=int, default=1,
                        help="mask dilation radius in pixels")
    parser.add_argument("--save-mask", action="store_true")
    return parser


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(path, device, cli):
    if str(path).lower().endswith((".pt", ".torchscript")):
        extra = {"config.json": ""}
        model = torch.jit.load(str(path), map_location=device, _extra_files=extra)
        config = json.loads(extra["config.json"] or "{}")
        model.eval()
        return model, config

    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint_model_config(
        checkpoint,
        {"model": "lite_eraser", "num_classes": 3, "output_stride": 16,
         "mean": DEFAULT_MEAN, "std": DEFAULT_STD},
    )
    if cli.model is not None:
        config["model"] = cli.model
    if cli.num_classes is not None:
        config["num_classes"] = cli.num_classes
    if cli.output_stride is not None:
        config["output_stride"] = cli.output_stride
    model = network.modeling.__dict__[config["model"]](
        num_classes=config["num_classes"],
        output_stride=config["output_stride"],
        pretrained_backbone=False,
    )
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(clean_state_dict(state))
    model.to(device).eval()
    return model, config


def tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def blend_window(height, width):
    y = np.hanning(height) if height > 1 else np.ones(1)
    x = np.hanning(width) if width > 1 else np.ones(1)
    # Non-zero borders ensure the outside edges of the page remain covered.
    return np.maximum(np.outer(y, x), 0.05).astype(np.float32)


def image_to_tensor(array, mean, std):
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).float().div_(255.0)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return tensor.sub_(mean_tensor).div_(std_tensor)


@torch.inference_mode()
def predict_tiled(model, image, device, config, tile_size, overlap, batch_size):
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile-size")
    if batch_size <= 0:
        raise ValueError("tile-batch-size must be positive")
    image_array = np.asarray(image, dtype=np.uint8)
    height, width = image_array.shape[:2]
    ys = tile_starts(height, tile_size, overlap)
    xs = tile_starts(width, tile_size, overlap)
    coordinates = [(y, x) for y in ys for x in xs]
    num_classes = int(config.get("num_classes", 3))
    logits_sum = np.zeros((num_classes, height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    mean, std = config.get("mean", DEFAULT_MEAN), config.get("std", DEFAULT_STD)

    for start in range(0, len(coordinates), batch_size):
        batch_coords = coordinates[start:start + batch_size]
        tensors = []
        for y, x in batch_coords:
            crop = image_array[y:min(y + tile_size, height), x:min(x + tile_size, width)]
            tensors.append(image_to_tensor(crop, mean, std))
        # All tiles share a shape except when the complete page is smaller than a tile.
        batch = torch.stack(tensors).to(device)
        batch_logits = model(batch).float().cpu().numpy()
        for logits, (y, x) in zip(batch_logits, batch_coords):
            h, w = logits.shape[-2:]
            weight = blend_window(h, w)
            logits_sum[:, y:y + h, x:x + w] += logits * weight[None]
            weight_sum[y:y + h, x:x + w] += weight
    return np.argmax(logits_sum / np.maximum(weight_sum, 1e-6), axis=0).astype(np.uint8)


def erase_handwriting(image, labels, handwriting_class=1, dilate=1):
    mask = Image.fromarray((labels == handwriting_class).astype(np.uint8) * 255)
    if dilate > 0:
        mask = mask.filter(ImageFilter.MaxFilter(2 * dilate + 1))
    result = np.asarray(image).copy()
    result[np.asarray(mask) > 0] = 255
    return Image.fromarray(result), mask


def collect_inputs(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path], path.parent
    if not path.is_dir():
        raise FileNotFoundError(input_path)
    files = sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError("no supported images under %s" % path)
    return files, path


def output_path_for(source, input_root, output, multiple):
    output = Path(output)
    if not multiple and output.suffix:
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    relative = source.relative_to(input_root) if multiple else Path(source.name)
    destination = output / relative
    destination = destination.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def main():
    opts = get_argparser().parse_args()
    device = resolve_device(opts.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, opts.threads))
    model, config = load_model(opts.checkpoint, device, opts)
    if not 0 <= opts.handwriting_class < int(config.get("num_classes", 3)):
        raise ValueError("handwriting-class is outside the model class range")
    if opts.dilate < 0:
        raise ValueError("dilate must be non-negative")
    files, input_root = collect_inputs(opts.input)
    print("device: %s, model: %s, images: %d" %
          (device, config.get("model", "torchscript"), len(files)))

    for source in tqdm(files):
        image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
        labels = predict_tiled(
            model, image, device, config, opts.tile_size, opts.overlap,
            opts.tile_batch_size,
        )
        result, mask = erase_handwriting(
            image, labels, opts.handwriting_class, opts.dilate
        )
        destination = output_path_for(
            source, input_root, opts.output, len(files) > 1
        )
        result.save(destination)
        if opts.save_mask:
            mask.save(destination.with_name(destination.stem + "_mask.png"))


if __name__ == "__main__":
    main()
