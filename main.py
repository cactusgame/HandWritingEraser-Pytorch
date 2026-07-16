"""Training entry point for handwriting/print/background segmentation."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm

import network
import utils
from datasets import HWSegmentation
from metrics import StreamSegMetrics
from utils import ext_transforms as et


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", "--data_root", default="./datasets/data")
    available_models = sorted(
        name
        for name, value in network.modeling.__dict__.items()
        if name.islower()
        and not name.startswith("_")
        and callable(value)
        and (name.startswith("deeplab") or name == "lite_eraser")
    )
    parser.add_argument("--model", default="lite_eraser", choices=available_models)
    parser.add_argument("--output-stride", "--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--num-classes", type=int, default=3, choices=[3])
    parser.add_argument("--pretrained-backbone", dest="pretrained_backbone", action="store_true")
    parser.add_argument("--no-pretrained-backbone", dest="pretrained_backbone", action="store_false")
    parser.set_defaults(pretrained_backbone=True)

    parser.add_argument("--total-itrs", "--total_itrs", type=int, default=30000)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--crop-size", "--crop_size", type=int, default=768)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-itrs", type=int, default=500)
    parser.add_argument("--loss", "--loss_type", default="hybrid",
                        choices=["hybrid", "focal", "cross_entropy"])
    parser.add_argument("--class-weights", default="1,4,2",
                        help="background,handwriting,print weights")
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-list", default=None,
                        help="optional file containing validation image stems")
    parser.add_argument("--random-seed", "--random_seed", type=int, default=1)
    parser.add_argument("--print-interval", type=int, default=20)
    parser.add_argument("--val-interval", "--val_interval", type=int, default=500)
    parser.add_argument("--val-tile-size", type=int, default=1024)
    parser.add_argument("--val-overlap", type=int, default=128)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gpu-id", "--gpu_id", default="0")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--continue-training", "--continue_training", action="store_true")
    parser.add_argument("--test-only", "--test_only", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    return parser


def parse_class_weights(value, num_classes):
    weights = [float(item.strip()) for item in value.split(",")]
    if len(weights) != num_classes or any(weight <= 0 for weight in weights):
        raise ValueError("--class-weights needs %d positive values" % num_classes)
    return weights


def resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset(opts):
    train_transform = et.ExtCompose([
        et.ExtColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        et.ExtRandomScale((0.75, 1.5)),
        et.ExtForegroundRandomCrop(
            opts.crop_size, target_classes=(1,), min_foreground_ratio=0.001
        ),
        et.ExtToTensor(),
        et.ExtNormalize(MEAN, STD),
    ])
    eval_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(MEAN, STD),
    ])
    common = dict(
        root=opts.data_root,
        val_ratio=opts.val_ratio,
        split_seed=opts.random_seed,
        val_list=opts.val_list,
    )
    return (
        HWSegmentation(transform=train_transform, train=True, **common),
        HWSegmentation(transform=eval_transform, train=False, **common),
    )


def build_model(opts, pretrained_backbone):
    return network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes,
        output_stride=opts.output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def parameter_groups(model, lr, backbone_lr_mult):
    backbone, decoder = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("low_level", "high_level", "backbone")):
            backbone.append(parameter)
        else:
            decoder.append(parameter)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": lr * backbone_lr_mult})
    if decoder:
        groups.append({"params": decoder, "lr": lr})
    return groups


def tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


@torch.no_grad()
def predict_page(model, image, device, tile_size, overlap):
    """Return CPU logits while keeping full-resolution activations off the GPU."""
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("validation needs 0 <= overlap < tile-size")
    height, width = image.shape[-2:]
    if height <= tile_size and width <= tile_size:
        return model(image.to(device, dtype=torch.float32)).cpu()

    ys = tile_starts(height, tile_size, overlap)
    xs = tile_starts(width, tile_size, overlap)
    logits_sum = None
    weight_sum = torch.zeros((height, width), dtype=torch.float32)
    window_cache = {}
    for y in ys:
        for x in xs:
            crop = image[..., y:min(y + tile_size, height),
                         x:min(x + tile_size, width)]
            logits = model(crop.to(device, dtype=torch.float32)).float().cpu()
            crop_h, crop_w = logits.shape[-2:]
            if logits_sum is None:
                logits_sum = torch.zeros(
                    (1, logits.shape[1], height, width), dtype=torch.float32
                )
            key = (crop_h, crop_w)
            if key not in window_cache:
                wy = torch.hann_window(crop_h, periodic=False)
                wx = torch.hann_window(crop_w, periodic=False)
                window_cache[key] = torch.outer(wy, wx).clamp_min_(0.05)
            weight = window_cache[key]
            logits_sum[..., y:y + crop_h, x:x + crop_w] += logits * weight
            weight_sum[y:y + crop_h, x:x + crop_w] += weight
    return logits_sum / weight_sum.clamp_min_(1e-6)


@torch.no_grad()
def validate(model, loader, device, metrics, tile_size, overlap):
    model.eval()
    metrics.reset()
    for images, labels in tqdm(loader, desc="validation", leave=False):
        logits = predict_page(model, images, device, tile_size, overlap)
        predictions = logits.argmax(1).cpu().numpy()
        metrics.update(labels.numpy(), predictions)
    return metrics.get_results()


def save_checkpoint(path, model, optimizer, scheduler, opts, iteration, best_score):
    state = {
        "format_version": 2,
        "iteration": iteration,
        "cur_itrs": iteration,
        "model_state": utils.unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "model_config": {
            "model": opts.model,
            "num_classes": opts.num_classes,
            "output_stride": opts.output_stride,
            "mean": MEAN,
            "std": STD,
        },
    }
    torch.save(state, path)
    print("saved: %s" % path)


def main():
    opts = get_argparser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = resolve_device(opts.device)
    seed_everything(opts.random_seed)
    print("device: %s" % device)

    train_dataset, val_dataset = get_dataset(opts)
    generator = torch.Generator().manual_seed(opts.random_seed)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) >= opts.batch_size,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    # Full-resolution pages can have different shapes, so validation uses batch 1.
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opts.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(opts, opts.pretrained_backbone and opts.ckpt is None)
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        parameter_groups(utils.unwrap_model(model), opts.lr, opts.backbone_lr_mult),
        lr=opts.lr,
        weight_decay=opts.weight_decay,
    )
    scheduler = utils.WarmupPolyLR(
        optimizer, opts.total_itrs, opts.warmup_itrs, power=0.9
    )
    class_weights = parse_class_weights(opts.class_weights, opts.num_classes)
    criterion = utils.build_loss(
        opts.loss, class_weights, opts.dice_weight
    ).to(device)
    metrics = StreamSegMetrics(opts.num_classes)

    iteration, best_score = 0, -1.0
    if opts.ckpt:
        checkpoint = utils.load_checkpoint(opts.ckpt, model, map_location=device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            iteration = checkpoint.get("iteration", checkpoint.get("cur_itrs", 0))
            best_score = checkpoint.get("best_score", -1.0)
        print("loaded: %s" % opts.ckpt)

    if opts.test_only:
        score = validate(
            model, val_loader, device, metrics,
            opts.val_tile_size, opts.val_overlap,
        )
        print(metrics.to_str(score))
        return

    checkpoint_dir = Path(opts.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    running_loss = 0.0
    model.train()

    while iteration < opts.total_itrs:
        for images, labels in train_loader:
            if iteration >= opts.total_itrs:
                break
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            iteration += 1
            running_loss += loss.detach().item()
            if iteration % opts.print_interval == 0:
                print(
                    "iteration %d/%d  loss %.5f  lr %.2e"
                    % (iteration, opts.total_itrs,
                       running_loss / opts.print_interval,
                       optimizer.param_groups[-1]["lr"])
                )
                running_loss = 0.0

            if iteration % opts.val_interval == 0 or iteration == opts.total_itrs:
                score = validate(
                    model, val_loader, device, metrics,
                    opts.val_tile_size, opts.val_overlap,
                )
                print(metrics.to_str(score))
                handwriting_iou = score["Handwriting IoU"]
                improved = (
                    not np.isnan(handwriting_iou) and handwriting_iou > best_score
                )
                if improved:
                    best_score = handwriting_iou
                save_checkpoint(
                    checkpoint_dir / "latest.pth", model, optimizer, scheduler,
                    opts, iteration, best_score,
                )
                if improved:
                    save_checkpoint(
                        checkpoint_dir / "best.pth", model, optimizer, scheduler,
                        opts, iteration, best_score,
                    )
                model.train()


if __name__ == "__main__":
    main()
