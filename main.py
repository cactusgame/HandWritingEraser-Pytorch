"""Training entry point for joint handwriting segmentation and restoration."""

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
from datasets import (
    HWRestorationDataset,
    JointDocumentTransform,
    MultiSourceHWRestoration,
)
from metrics import StreamSegMetrics


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", "--data_root", dest="data_roots", action="append",
        help="Baidu-format dataset root; repeat this option for multiple sources",
    )
    available_models = sorted(
        name
        for name, value in network.modeling.__dict__.items()
        if name.islower()
        and not name.startswith("_")
        and callable(value)
        and (name.startswith("deeplab") or name.endswith("eraser"))
    )
    parser.add_argument("--model", default="layered_eraser", choices=available_models)
    parser.add_argument(
        "--output-stride", "--output_stride", type=int, default=None,
        choices=[8, 16, 32],
        help="defaults to 32 for server/joint eraser and 16 for other models",
    )
    parser.add_argument("--num-classes", type=int, default=3, choices=[3])
    parser.add_argument("--pretrained-backbone", dest="pretrained_backbone", action="store_true")
    parser.add_argument("--no-pretrained-backbone", dest="pretrained_backbone", action="store_false")
    parser.set_defaults(pretrained_backbone=True)

    parser.add_argument("--total-itrs", "--total_itrs", type=int, default=80000)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=2)
    parser.add_argument("--crop-size", "--crop_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-itrs", type=int, default=1000)
    parser.add_argument(
        "--loss", "--loss_type", default="layered",
        choices=[
            "layered", "joint", "structure", "hybrid", "focal",
            "cross_entropy",
        ],
    )
    parser.add_argument("--class-weights", default="1,3,2",
                        help="background,handwriting,print weights")
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--tversky-weight", type=float, default=0.7)
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--tversky-alpha", type=float, default=0.35,
                        help="false-positive weight in handwriting Tversky loss")
    parser.add_argument("--tversky-beta", type=float, default=0.65,
                        help="false-negative weight; larger values reduce missed ink")
    parser.add_argument("--tversky-gamma", type=float, default=0.75)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.4)
    parser.add_argument("--edge-weight", type=float, default=0.3)
    parser.add_argument("--color-weight", type=float, default=0.2)
    parser.add_argument("--identity-weight", type=float, default=0.5)
    parser.add_argument("--print-structure-weight", type=float, default=1.5)
    parser.add_argument("--background-weight", type=float, default=2.0)
    parser.add_argument(
        "--background-low-frequency-weight", type=float, default=0.5
    )
    parser.add_argument("--overlap-weight", type=float, default=8.0)
    parser.add_argument("--overlap-ssim-weight", type=float, default=1.0)
    parser.add_argument("--overlap-edge-weight", type=float, default=2.0)
    parser.add_argument("--composition-weight", type=float, default=1.0)
    parser.add_argument(
        "--freeze-segmentation-itrs", type=int, default=2000,
        help="when initializing from cpu_v2, train only restoration heads first",
    )
    parser.add_argument(
        "--synthetic-restoration-probability", type=float, default=0.85,
        help="for unpaired sources, probability of overlaying real ink onto a clean region",
    )
    parser.add_argument(
        "--color-negative-probability", type=float, default=0.55,
        help="probability of colored paper/print negative augmentation",
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-list", default=None,
                        help="validation stems for a single legacy data root")
    parser.add_argument(
        "--dataset-sampling", choices=["balanced", "proportional"],
        default="balanced",
        help="equalize dataset domains or sample in proportion to their sizes",
    )
    parser.add_argument(
        "--dataset-weights", default=None,
        help="optional comma-separated source weights in --data-root order",
    )
    parser.add_argument("--random-seed", "--random_seed", type=int, default=1)
    parser.add_argument("--print-interval", type=int, default=50)
    parser.add_argument("--val-interval", "--val_interval", type=int, default=2000)
    parser.add_argument("--val-tile-size", type=int, default=1024)
    parser.add_argument("--val-overlap", type=int, default=128)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gpu-id", "--gpu_id", default="0")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument(
        "--init-segmentation-ckpt", default=None,
        help="initialize joint_eraser segmentation trunk from a server_eraser checkpoint",
    )
    parser.add_argument("--continue-training", "--continue_training", action="store_true")
    parser.add_argument("--test-only", "--test_only", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--tensorboard-dir", default=None,
        help="local TensorBoard log directory; defaults to <checkpoint-dir>/runs",
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true",
        help="disable local TensorBoard event logging",
    )
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


def parse_dataset_weights(value, count):
    if value is None:
        return None
    weights = [float(item.strip()) for item in value.split(",")]
    if len(weights) != count or any(weight <= 0 for weight in weights):
        raise ValueError("--dataset-weights needs %d positive values" % count)
    return weights


def get_datasets(opts):
    train_transform = JointDocumentTransform(
        crop_size=opts.crop_size, train=True, mean=MEAN, std=STD,
        color_negative_probability=opts.color_negative_probability,
    )
    eval_transform = JointDocumentTransform(
        crop_size=opts.crop_size, train=False, mean=MEAN, std=STD
    )
    roots = opts.data_roots or ["./datasets/data"]
    if opts.val_list and len(roots) != 1:
        raise ValueError("--val-list is only valid with one --data-root")
    train_sources = []
    validation_sources = []
    used_names = {}
    for root in roots:
        common = dict(
            root=root,
            val_ratio=opts.val_ratio,
            split_seed=opts.random_seed,
            val_list=opts.val_list,
        )
        train_dataset = HWRestorationDataset(
            transform=train_transform,
            split="train",
            synthetic_probability=opts.synthetic_restoration_probability,
            **common,
        )
        validation_dataset = HWRestorationDataset(
            transform=eval_transform,
            split="validation",
            synthetic_probability=0.0,
            **common,
        )
        name = validation_dataset.dataset_name
        used_names[name] = used_names.get(name, 0) + 1
        if used_names[name] > 1:
            name = "%s#%d" % (name, used_names[name])
        train_sources.append(train_dataset)
        validation_sources.append((name, validation_dataset))
    return MultiSourceHWRestoration(train_sources), validation_sources


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
        if name.startswith(("low_level", "high_level", "backbone", "encoder_")):
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
    """Return CPU segmentation, optional print logits, and restored RGB."""
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("validation needs 0 <= overlap < tile-size")
    height, width = image.shape[-2:]
    if height <= tile_size and width <= tile_size:
        outputs = model(image.to(device, dtype=torch.float32))
        if isinstance(outputs, (tuple, list)):
            print_logits = outputs[1] if len(outputs) == 4 else None
            return (
                outputs[0].float().cpu(),
                None if print_logits is None else print_logits.float().cpu(),
                outputs[-1].float().cpu(),
            )
        return outputs.float().cpu(), None, None

    ys = tile_starts(height, tile_size, overlap)
    xs = tile_starts(width, tile_size, overlap)
    logits_sum = None
    print_sum = None
    restored_sum = None
    weight_sum = torch.zeros((height, width), dtype=torch.float32)
    window_cache = {}
    for y in ys:
        for x in xs:
            crop = image[..., y:min(y + tile_size, height),
                         x:min(x + tile_size, width)]
            outputs = model(crop.to(device, dtype=torch.float32))
            if isinstance(outputs, (tuple, list)):
                logits = outputs[0].float().cpu()
                print_logits = (
                    outputs[1].float().cpu() if len(outputs) == 4 else None
                )
                restored = outputs[-1].float().cpu()
            else:
                logits = outputs.float().cpu()
                print_logits = None
                restored = None
            crop_h, crop_w = logits.shape[-2:]
            if logits_sum is None:
                logits_sum = torch.zeros(
                    (1, logits.shape[1], height, width), dtype=torch.float32
                )
                if restored is not None:
                    restored_sum = torch.zeros(
                        (1, 3, height, width), dtype=torch.float32
                    )
                if print_logits is not None:
                    print_sum = torch.zeros(
                        (1, 1, height, width), dtype=torch.float32
                    )
            key = (crop_h, crop_w)
            if key not in window_cache:
                wy = torch.hann_window(crop_h, periodic=False)
                wx = torch.hann_window(crop_w, periodic=False)
                window_cache[key] = torch.outer(wy, wx).clamp_min_(0.05)
            weight = window_cache[key]
            logits_sum[..., y:y + crop_h, x:x + crop_w] += logits * weight
            if restored_sum is not None:
                restored_sum[..., y:y + crop_h, x:x + crop_w] += (
                    restored * weight
                )
            if print_sum is not None:
                print_sum[..., y:y + crop_h, x:x + crop_w] += (
                    print_logits * weight
                )
            weight_sum[y:y + crop_h, x:x + crop_w] += weight
    denominator = weight_sum.clamp_min_(1e-6)
    restored = None if restored_sum is None else restored_sum / denominator
    print_logits = None if print_sum is None else print_sum / denominator
    return logits_sum / denominator, print_logits, restored


def _new_restoration_stats():
    return {
        "absolute": 0.0, "squared": 0.0, "count": 0.0,
        "background_absolute": 0.0, "background_count": 0.0,
        "overlap_absolute": 0.0, "overlap_squared": 0.0,
        "overlap_count": 0.0,
        "print_tp": 0.0, "print_fp": 0.0, "print_fn": 0.0,
    }


def _update_restoration_stats(
    stats, restored, print_logits, clean, labels, clean_print, valid
):
    if restored is None or not bool(valid.item() > 0):
        return
    hand = labels == 1
    print_target = clean_print > 0.5
    mask = hand.unsqueeze(1).expand(-1, 3, -1, -1)
    count = float(mask.sum().item())
    if count == 0:
        return
    difference = (restored - clean).float()
    stats["absolute"] += float(difference.abs()[mask].sum().item())
    stats["squared"] += float(difference.pow(2)[mask].sum().item())
    stats["count"] += count
    background_mask = (
        hand & ~print_target
    ).unsqueeze(1).expand(-1, 3, -1, -1)
    overlap_mask = (
        hand & print_target
    ).unsqueeze(1).expand(-1, 3, -1, -1)
    stats["background_absolute"] += float(
        difference.abs()[background_mask].sum().item()
    )
    stats["background_count"] += float(background_mask.sum().item())
    stats["overlap_absolute"] += float(
        difference.abs()[overlap_mask].sum().item()
    )
    stats["overlap_squared"] += float(
        difference.pow(2)[overlap_mask].sum().item()
    )
    stats["overlap_count"] += float(overlap_mask.sum().item())
    if print_logits is not None:
        predicted_print = torch.sigmoid(print_logits) >= 0.5
        target = print_target.unsqueeze(1)
        valid_hand = hand.unsqueeze(1)
        stats["print_tp"] += float(
            (predicted_print & target & valid_hand).sum().item()
        )
        stats["print_fp"] += float(
            (predicted_print & ~target & valid_hand).sum().item()
        )
        stats["print_fn"] += float(
            (~predicted_print & target & valid_hand).sum().item()
        )


def _add_restoration_results(score, stats):
    if stats["count"] <= 0:
        return
    mae = stats["absolute"] / stats["count"]
    mse = stats["squared"] / stats["count"]
    score["Restoration Masked MAE"] = mae
    score["Restoration Masked PSNR"] = -10.0 * np.log10(max(mse, 1e-10))
    score["Restoration Fidelity"] = max(0.0, 1.0 - mae)
    if stats["background_count"] > 0:
        background_mae = (
            stats["background_absolute"] / stats["background_count"]
        )
        score["Background MAE"] = background_mae
        score["Background Fidelity"] = max(0.0, 1.0 - background_mae)
    if stats["overlap_count"] > 0:
        overlap_mae = stats["overlap_absolute"] / stats["overlap_count"]
        overlap_mse = stats["overlap_squared"] / stats["overlap_count"]
        score["Overlap MAE"] = overlap_mae
        score["Overlap PSNR"] = -10.0 * np.log10(
            max(overlap_mse, 1e-10)
        )
        score["Overlap Fidelity"] = max(0.0, 1.0 - overlap_mae)
    denominator = (
        2.0 * stats["print_tp"] + stats["print_fp"] + stats["print_fn"]
    )
    if denominator > 0:
        score["Occluded Print F1"] = (
            2.0 * stats["print_tp"] / denominator
        )
        score["Occluded Print Recall"] = stats["print_tp"] / max(
            stats["print_tp"] + stats["print_fn"], 1.0
        )


@torch.no_grad()
def validate_sources(model, loaders, device, metrics, tile_size, overlap):
    model.eval()
    metrics.reset()
    source_scores = {}
    overall_restoration = _new_restoration_stats()
    for source_name, loader in loaders:
        source_metrics = StreamSegMetrics(metrics.n_classes)
        source_restoration = _new_restoration_stats()
        for (
            images, labels, clean_targets, clean_print_masks,
            restoration_valid,
        ) in tqdm(
            loader, desc="validation/%s" % source_name, leave=False
        ):
            logits, print_logits, restored = predict_page(
                model, images, device, tile_size, overlap
            )
            predictions = logits.argmax(1).cpu().numpy()
            targets = labels.numpy()
            source_metrics.update(targets, predictions)
            metrics.update(targets, predictions)
            _update_restoration_stats(
                source_restoration, restored, print_logits, clean_targets,
                labels, clean_print_masks, restoration_valid,
            )
            _update_restoration_stats(
                overall_restoration, restored, print_logits, clean_targets,
                labels, clean_print_masks, restoration_valid,
            )
        source_score = source_metrics.get_results()
        _add_restoration_results(source_score, source_restoration)
        source_scores[source_name] = source_score
    overall_score = metrics.get_results()
    _add_restoration_results(overall_score, overall_restoration)
    return overall_score, source_scores


def print_validation_scores(overall_score, source_scores, metrics):
    for source_name, score in source_scores.items():
        print("\n[validation/%s]%s" % (source_name, metrics.to_str(score)))
    print("\n[validation/all]%s" % metrics.to_str(overall_score))


def validation_selection_score(source_scores):
    """Require segmentation, paper color, and occluded print to improve."""
    values = [score["Erase Quality"] for score in source_scores.values()]
    erase_quality = float(np.nanmean(values))
    layered_scores = [
        score for score in source_scores.values()
        if "Overlap Fidelity" in score
    ]
    if layered_scores:
        background = float(np.nanmean([
            score.get("Background Fidelity", score["Restoration Fidelity"])
            for score in layered_scores
        ]))
        overlap = float(np.nanmean([
            score["Overlap Fidelity"] for score in layered_scores
        ]))
        print_f1_values = [
            score["Occluded Print F1"] for score in layered_scores
            if "Occluded Print F1" in score
        ]
        print_f1 = (
            float(np.nanmean(print_f1_values))
            if print_f1_values else overlap
        )
        value = float(
            max(erase_quality * background * overlap * print_f1, 0.0)
            ** 0.25
        )
        print(
            "validation Layered Quality: %.6f "
            "(erase %.6f, background %.6f, overlap %.6f, print-f1 %.6f)"
            % (value, erase_quality, background, overlap, print_f1)
        )
        return value
    restoration = [
        score["Restoration Fidelity"] for score in source_scores.values()
        if "Restoration Fidelity" in score
    ]
    if restoration:
        restoration_fidelity = float(np.nanmean(restoration))
        value = float(np.sqrt(erase_quality * restoration_fidelity))
        print(
            "validation Joint Quality: %.6f (erase %.6f, restoration %.6f)"
            % (value, erase_quality, restoration_fidelity)
        )
        return value
    print("validation macro Erase Quality: %.6f" % erase_quality)
    return erase_quality


def _tb_name(name):
    return str(name).strip().replace(" ", "_").replace("/", "_")


def _write_scalar(writer, tag, value, iteration):
    if value is None:
        return
    value = float(value)
    if np.isfinite(value):
        writer.add_scalar(tag, value, iteration)


def write_score_scalars(writer, prefix, score, iteration):
    for name, value in score.items():
        if name == "Class IoU":
            for class_id, class_iou in value.items():
                _write_scalar(
                    writer,
                    "%s/Class_IoU/class_%s" % (prefix, class_id),
                    class_iou,
                    iteration,
                )
        else:
            _write_scalar(writer, "%s/%s" % (prefix, _tb_name(name)), value, iteration)


def write_validation_scalars(
    writer, iteration, overall_score, source_scores, macro_selection_score
):
    if writer is None:
        return
    write_score_scalars(writer, "validation/all", overall_score, iteration)
    for source_name, score in source_scores.items():
        write_score_scalars(
            writer, "validation/%s" % _tb_name(source_name), score, iteration
        )
    _write_scalar(
        writer,
        "validation/macro/Selection_Score",
        macro_selection_score,
        iteration,
    )
    macro_handwriting_iou = float(
        np.nanmean([score["Handwriting IoU"] for score in source_scores.values()])
    )
    _write_scalar(
        writer,
        "validation/macro/Handwriting_IoU",
        macro_handwriting_iou,
        iteration,
    )
    writer.flush()


def create_summary_writer(opts, checkpoint_dir):
    if opts.no_tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging is enabled, but tensorboard is not installed. "
            "Install it in the training environment or pass --no-tensorboard."
        ) from exc

    tensorboard_dir = Path(opts.tensorboard_dir or checkpoint_dir / "runs")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print("tensorboard logs: %s" % tensorboard_dir)
    return writer


def save_checkpoint(
    path, model, inference_model, optimizer, scheduler, opts, iteration, best_score,
    ema_updates=0,
):
    state = {
        "format_version": 5 if opts.model == "layered_eraser" else 4,
        "iteration": iteration,
        "cur_itrs": iteration,
        # Inference and export load model_state by default. When EMA is enabled,
        # this is the smoother model that was actually evaluated.
        "model_state": utils.unwrap_model(inference_model).state_dict(),
        "training_model_state": utils.unwrap_model(model).state_dict(),
        "ema_updates": ema_updates,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "best_score_name": (
            "validation Layered Quality"
            if opts.model == "layered_eraser"
            else "validation Joint Quality"
        ),
        "model_config": {
            "model": opts.model,
            "num_classes": opts.num_classes,
            "output_stride": opts.output_stride,
            "mean": MEAN,
            "std": STD,
            "task": (
                "layered_restoration"
                if opts.model == "layered_eraser"
                else (
                    "joint_restoration"
                    if opts.model == "joint_eraser"
                    else "segmentation"
                )
            ),
        },
        "data_config": {
            "roots": list(opts.data_roots or ["./datasets/data"]),
            "sampling": opts.dataset_sampling,
            "dataset_weights": opts.dataset_weights,
            "synthetic_restoration_probability": opts.synthetic_restoration_probability,
            "color_negative_probability": opts.color_negative_probability,
        },
        "training_config": {
            "loss": opts.loss,
            "class_weights": opts.class_weights,
            "tversky_weight": opts.tversky_weight,
            "boundary_weight": opts.boundary_weight,
            "tversky_alpha": opts.tversky_alpha,
            "tversky_beta": opts.tversky_beta,
            "tversky_gamma": opts.tversky_gamma,
            "boundary_radius": opts.boundary_radius,
            "label_smoothing": opts.label_smoothing,
            "reconstruction_weight": opts.reconstruction_weight,
            "ssim_weight": opts.ssim_weight,
            "edge_weight": opts.edge_weight,
            "color_weight": opts.color_weight,
            "identity_weight": opts.identity_weight,
            "print_structure_weight": opts.print_structure_weight,
            "background_weight": opts.background_weight,
            "background_low_frequency_weight": (
                opts.background_low_frequency_weight
            ),
            "overlap_weight": opts.overlap_weight,
            "overlap_ssim_weight": opts.overlap_ssim_weight,
            "overlap_edge_weight": opts.overlap_edge_weight,
            "composition_weight": opts.composition_weight,
            "freeze_segmentation_itrs": opts.freeze_segmentation_itrs,
            "ema_decay": None if opts.no_ema else opts.ema_decay,
        },
    }
    torch.save(state, path)
    print("saved: %s" % path)


def initialize_joint_from_segmentation(path, model):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    result = utils.unwrap_model(model).load_state_dict(
        utils.clean_state_dict(state), strict=False
    )
    allowed_prefixes = ("restore_half.", "restore_half_out.", "restore_full.")
    allowed_buffers = {"input_mean", "input_std"}
    invalid_missing = [
        key for key in result.missing_keys
        if key not in allowed_buffers and not key.startswith(allowed_prefixes)
    ]
    if invalid_missing or result.unexpected_keys:
        raise RuntimeError(
            "segmentation initialization is incompatible; missing=%s unexpected=%s"
            % (invalid_missing[:10], result.unexpected_keys[:10])
        )
    print("initialized joint segmentation trunk: %s" % path)


def set_joint_segmentation_trainable(model, trainable):
    model = utils.unwrap_model(model)
    restoration_prefixes = ("restore_half.", "restore_half_out.", "restore_full.")
    for name, parameter in model.named_parameters():
        if not name.startswith(restoration_prefixes):
            parameter.requires_grad_(trainable)
    if not trainable:
        for name in (
            "encoder_s2", "encoder_s4", "encoder_s8", "encoder_s16",
            "encoder_s32", "context", "decode_s16", "decode_s8",
            "decode_s4", "decode_s2", "classifier",
        ):
            getattr(model, name).eval()


def main():
    opts = get_argparser().parse_args()
    if opts.output_stride is None:
        opts.output_stride = (
            32
            if opts.model in {
                "server_eraser", "joint_eraser", "layered_eraser"
            }
            else 16
        )
    expected_loss = {
        "joint_eraser": "joint",
        "layered_eraser": "layered",
    }.get(opts.model)
    if expected_loss is not None and opts.loss != expected_loss:
        raise ValueError(
            "%s must use --loss %s" % (opts.model, expected_loss)
        )
    if expected_loss is None and opts.loss in {"joint", "layered"}:
        raise ValueError(
            "--loss %s requires its matching restoration model" % opts.loss
        )
    if opts.ckpt and opts.init_segmentation_ckpt:
        raise ValueError("--ckpt and --init-segmentation-ckpt are mutually exclusive")
    if opts.init_segmentation_ckpt and opts.model != "joint_eraser":
        raise ValueError(
            "--init-segmentation-ckpt only supports joint_eraser; "
            "layered_eraser uses an ImageNet-pretrained ResNet-34 backbone"
        )
    if not 0.0 <= opts.color_negative_probability <= 1.0:
        raise ValueError("--color-negative-probability must be in [0, 1]")
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = resolve_device(opts.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed_everything(opts.random_seed)
    print("device: %s" % device)

    train_dataset, validation_datasets = get_datasets(opts)
    generator = torch.Generator().manual_seed(opts.random_seed)
    sampler = None
    dataset_weights = parse_dataset_weights(
        opts.dataset_weights, len(train_dataset.datasets)
    )
    if dataset_weights is not None and opts.dataset_sampling != "balanced":
        raise ValueError("--dataset-weights requires --dataset-sampling balanced")
    if opts.dataset_sampling == "balanced" and len(train_dataset.datasets) > 1:
        sampler = data.WeightedRandomSampler(
            train_dataset.balanced_sample_weights(dataset_weights),
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=opts.workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) >= opts.batch_size,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    # Full-resolution pages can have different shapes, so validation uses batch 1.
    validation_loaders = [
        (
            name,
            data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=opts.workers,
                pin_memory=device.type == "cuda",
            ),
        )
        for name, dataset in validation_datasets
    ]
    print("training sources: %s; sampling: %s" %
          (dict(zip(train_dataset.dataset_names,
                    [len(dataset) for dataset in train_dataset.datasets])),
           opts.dataset_sampling))

    model = build_model(
        opts,
        opts.pretrained_backbone
        and opts.ckpt is None
        and opts.init_segmentation_ckpt is None,
    )
    model.to(device)
    if opts.init_segmentation_ckpt:
        initialize_joint_from_segmentation(opts.init_segmentation_ckpt, model)
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
        opts.loss,
        class_weights,
        opts.dice_weight,
        tversky_weight=opts.tversky_weight,
        boundary_weight=opts.boundary_weight,
        tversky_alpha=opts.tversky_alpha,
        tversky_beta=opts.tversky_beta,
        tversky_gamma=opts.tversky_gamma,
        boundary_radius=opts.boundary_radius,
        label_smoothing=opts.label_smoothing,
        reconstruction_weight=opts.reconstruction_weight,
        ssim_weight=opts.ssim_weight,
        edge_weight=opts.edge_weight,
        color_weight=opts.color_weight,
        identity_weight=opts.identity_weight,
        print_structure_weight=opts.print_structure_weight,
        background_weight=opts.background_weight,
        background_low_frequency_weight=(
            opts.background_low_frequency_weight
        ),
        overlap_weight=opts.overlap_weight,
        overlap_ssim_weight=opts.overlap_ssim_weight,
        overlap_edge_weight=opts.overlap_edge_weight,
        composition_weight=opts.composition_weight,
    ).to(device)
    metrics = StreamSegMetrics(opts.num_classes)

    iteration, best_score = 0, -1.0
    checkpoint = None
    if opts.ckpt:
        checkpoint = utils.load_checkpoint(
            opts.ckpt,
            model,
            map_location=device,
            state_key="training_model_state" if opts.continue_training else None,
        )
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            iteration = checkpoint.get("iteration", checkpoint.get("cur_itrs", 0))
            best_score = checkpoint.get("best_score", -1.0)
        print("loaded: %s" % opts.ckpt)

    ema = None if opts.no_ema else utils.ModelEMA(
        model, decay=opts.ema_decay,
        updates=checkpoint.get("ema_updates", 0) if checkpoint else 0,
    )
    if ema is not None and checkpoint and "model_state" in checkpoint:
        ema.load_state_dict(
            checkpoint["model_state"], checkpoint.get("ema_updates", 0)
        )
    validation_model = ema.module if ema is not None else model

    checkpoint_dir = Path(opts.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = create_summary_writer(opts, checkpoint_dir)

    if opts.test_only:
        score, source_scores = validate_sources(
            validation_model, validation_loaders, device, metrics,
            opts.val_tile_size, opts.val_overlap,
        )
        print_validation_scores(score, source_scores, metrics)
        selection_score = validation_selection_score(source_scores)
        write_validation_scalars(writer, iteration, score, source_scores, selection_score)
        if writer is not None:
            writer.close()
        return

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    running_loss = 0.0
    running_components = {}
    model.train()
    segmentation_frozen = (
        opts.model == "joint_eraser"
        and opts.init_segmentation_ckpt is not None
        and iteration < opts.freeze_segmentation_itrs
    )
    if segmentation_frozen:
        set_joint_segmentation_trainable(model, False)
        print(
            "segmentation trunk frozen until iteration %d"
            % opts.freeze_segmentation_itrs
        )

    while iteration < opts.total_itrs:
        for (
            images, labels, clean_targets, clean_print_masks,
            restoration_valid,
        ) in train_loader:
            if iteration >= opts.total_itrs:
                break
            if segmentation_frozen and iteration >= opts.freeze_segmentation_itrs:
                set_joint_segmentation_trainable(model, True)
                model.train()
                segmentation_frozen = False
                print("segmentation trunk unfrozen at iteration %d" % iteration)
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            clean_targets = clean_targets.to(
                device, dtype=torch.float32, non_blocking=True
            )
            clean_print_masks = clean_print_masks.to(
                device, dtype=torch.float32, non_blocking=True
            )
            restoration_valid = restoration_valid.to(
                device, dtype=torch.float32, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                if opts.loss == "layered":
                    loss = criterion(
                        outputs, labels, clean_targets, clean_print_masks,
                        restoration_valid, images,
                    )
                elif opts.loss == "joint":
                    loss = criterion(
                        outputs, labels, clean_targets, restoration_valid, images
                    )
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            iteration += 1
            running_loss += loss.detach().item()
            for name, value in getattr(criterion, "last_components", {}).items():
                running_components[name] = (
                    running_components.get(name, 0.0) + float(value.item())
                )
            if iteration % opts.print_interval == 0:
                train_loss = running_loss / opts.print_interval
                memory_text = ""
                peak_reserved_gib = None
                if device.type == "cuda":
                    peak_reserved_gib = (
                        torch.cuda.max_memory_reserved() / (1024.0 ** 3)
                    )
                    memory_text = "  peak-vram %.2f GiB" % peak_reserved_gib
                print(
                    "iteration %d/%d  loss %.5f  lr %.2e%s"
                    % (iteration, opts.total_itrs,
                       train_loss,
                       optimizer.param_groups[-1]["lr"],
                       memory_text)
                )
                if writer is not None:
                    writer.add_scalar("train/loss", train_loss, iteration)
                    writer.add_scalar(
                        "train/lr", optimizer.param_groups[-1]["lr"], iteration
                    )
                    if peak_reserved_gib is not None:
                        writer.add_scalar(
                            "train/peak_vram_gib",
                            peak_reserved_gib,
                            iteration,
                        )
                    for name, value in running_components.items():
                        writer.add_scalar(
                            "train/loss_%s" % name,
                            value / opts.print_interval,
                            iteration,
                        )
                running_loss = 0.0
                running_components.clear()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()

            if iteration % opts.val_interval == 0 or iteration == opts.total_itrs:
                score, source_scores = validate_sources(
                    validation_model, validation_loaders, device, metrics,
                    opts.val_tile_size, opts.val_overlap,
                )
                print_validation_scores(score, source_scores, metrics)
                selection_score = validation_selection_score(source_scores)
                write_validation_scalars(
                    writer, iteration, score, source_scores, selection_score
                )
                improved = (
                    not np.isnan(selection_score) and selection_score > best_score
                )
                if improved:
                    best_score = selection_score
                save_checkpoint(
                    checkpoint_dir / "latest.pth", model, validation_model,
                    optimizer, scheduler, opts, iteration, best_score,
                    ema.updates if ema is not None else 0,
                )
                if improved:
                    save_checkpoint(
                        checkpoint_dir / "best.pth", model, validation_model,
                        optimizer, scheduler, opts, iteration, best_score,
                        ema.updates if ema is not None else 0,
                    )
                model.train()
                if segmentation_frozen:
                    set_joint_segmentation_trainable(model, False)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
