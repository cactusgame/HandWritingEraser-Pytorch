"""Build paired handwriting data from clean document images.

The clean document is the RGB target.  Handwriting comes from existing
Baidu-format datasets or a procedural scribble generator.  Outputs follow the
extended Baidu format used by layered restoration training.
"""

import argparse
import json
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.conversion_utils import (
    IMAGE_SUFFIXES,
    index_by_stem,
    prepare_output,
    write_metadata,
    write_splits,
)


DEFAULT_HANDWRITING_ROOTS = [
    "/Users/peng/Documents/data/HandWritingData/baidu",
    "/Users/peng/Documents/data/HandWritingData/SCUT-EnsExam-baidu-format",
    "/Users/peng/Documents/data/HandWritingData/SignaTR6K-baidu-format",
]

INK_PALETTE = np.asarray(
    [
        (24, 35, 52),      # black ballpoint
        (29, 66, 155),     # blue
        (43, 91, 177),     # lighter blue
        (156, 35, 46),     # red correction
        (75, 78, 84),      # pencil gray
        (28, 112, 77),     # green
    ],
    dtype=np.float32,
)

_WORKER_OPTIONS = None
_WORKER_PAIRS = None
_WORKER_OUTPUT = None


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-root", action="append", dest="clean_roots", required=True,
        help="directory of clean document images; repeat for multiple roots",
    )
    parser.add_argument(
        "--handwriting-root", action="append", dest="handwriting_roots",
        help=(
            "Baidu-format source containing Images/ and Labels/, or a parent "
            "whose immediate children have that structure; repeatable"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants-per-document", type=int, default=5)
    parser.add_argument("--clean-negative-ratio", type=float, default=0.25)
    parser.add_argument("--overlap-ratio", type=float, default=0.45)
    parser.add_argument(
        "--random-scribble-probability", type=float, default=0.25
    )
    parser.add_argument("--minimum-print-overlap", type=float, default=0.12)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1),
        help=(
            "document synthesis processes; each worker holds one full page "
            "and its masks in memory"
        ),
    )
    parser.add_argument(
        "--preview-count", type=int, default=24,
        help="number of generated samples included in preview.jpg; 0 disables",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--high-resolution-command",
        default=None,
        help=(
            "optional command template run separately on composite and clean "
            "images; must contain {input} and {output}, and is executed "
            "without a shell"
        ),
    )
    return parser


def collect_clean_images(roots):
    files = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        # Passing an already converted dataset should select only its clean
        # targets, not accidentally ingest Labels/ or CleanPrintMasks/.
        search_root = root / "CleanTargets"
        if not search_root.is_dir():
            search_root = root
        files.extend(
            path for path in search_root.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
                and not any(
                    part.lower() in {
                        "labels", "masks", "cleanprintmasks", "annotations",
                    }
                    for part in path.relative_to(search_root).parts[:-1]
                )
            )
        )
    files = sorted(set(path.resolve() for path in files))
    if not files:
        raise ValueError("no clean document images found")
    return files


def is_handwriting_dataset_root(root):
    root = Path(root)
    return (root / "Images").is_dir() and (root / "Labels").is_dir()


def resolve_handwriting_roots(roots):
    """Resolve explicit dataset roots or their common parent directory."""
    resolved = []
    for requested in roots:
        requested = Path(requested).expanduser().resolve()
        if not requested.is_dir():
            raise FileNotFoundError(requested)
        if is_handwriting_dataset_root(requested):
            candidates = [requested]
        else:
            candidates = sorted(
                child.resolve()
                for child in requested.iterdir()
                if child.is_dir() and is_handwriting_dataset_root(child)
            )
        if not candidates:
            raise ValueError(
                "%s is neither a Baidu-format dataset root nor a parent "
                "containing one; expected Images/ and Labels/ directories"
                % requested
            )
        resolved.extend(candidates)
    return sorted(set(resolved), key=lambda path: str(path).lower())


def validate_handwriting_labels(root, labels):
    """Reject raw masks that have not been converted to class IDs 0/1/2."""
    sample_paths = sorted(labels.values())[:3]
    for path in sample_paths:
        with Image.open(path) as source:
            values = set(np.unique(np.asarray(source.convert("L"))).tolist())
        invalid = values - {0, 1, 2}
        if invalid:
            raise ValueError(
                "%s is not a converted Baidu-format dataset: label %s "
                "contains class values outside 0/1/2 (%s)"
                % (root, path.name, sorted(invalid)[:8])
            )


def collect_handwriting_pairs(roots):
    pairs = []
    for root in roots:
        root = Path(root)
        images = index_by_stem(root / "Images")
        labels = index_by_stem(root / "Labels")
        validate_handwriting_labels(root, labels)
        common = sorted(set(images) & set(labels))
        if not common:
            raise ValueError("no image/label pairs under %s" % root)
        pairs.extend(
            (images[stem].resolve(), labels[stem].resolve())
            for stem in common
        )
    if not pairs:
        raise ValueError("no handwriting sources found")
    return pairs


def estimate_document_print_mask(image):
    """Find text, rules, and colored document foreground on local backgrounds."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luma = lab[..., 0]
    kernel_size = max(15, int(round(min(height, width) * 0.018)))
    kernel_size = min(kernel_size, 51)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    local_light = cv2.morphologyEx(luma, cv2.MORPH_CLOSE, kernel)
    dark_contrast = (
        local_light.astype(np.int16) - luma.astype(np.int16)
    ) >= 11

    sigma = max(2.0, min(height, width) / 120.0)
    local_color = cv2.GaussianBlur(
        lab.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma
    )
    color_delta = np.sqrt(
        np.square(lab.astype(np.float32) - local_color).sum(axis=2)
    )
    color_foreground = color_delta >= 16.0
    edges = cv2.Canny(luma, 45, 120) > 0
    edges = cv2.dilate(
        edges.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ) > 0
    mask = dark_contrast | color_foreground | edges

    # Remove isolated scanner noise while retaining 1-2 pixel glyph strokes.
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    valid_components = (
        stats[:, cv2.CC_STAT_AREA] >= 3
    )
    valid_components[0] = False
    return valid_components[components]


def split_documents(files, validation_ratio, test_ratio, seed):
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation-ratio must be in [0, 1)")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test-ratio must be in [0, 1)")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation-ratio + test-ratio must be below 1")
    if len(files) < 3 and (validation_ratio > 0 or test_ratio > 0):
        raise ValueError(
            "at least three clean documents are required for train/val/test"
        )
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    validation_count = (
        max(1, int(round(len(files) * validation_ratio)))
        if validation_ratio > 0 else 0
    )
    test_count = (
        max(1, int(round(len(files) * test_ratio)))
        if test_ratio > 0 else 0
    )
    while validation_count + test_count >= len(files):
        if test_count > validation_count and test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
        else:
            break
    validation = shuffled[:validation_count]
    test = shuffled[validation_count:validation_count + test_count]
    train = shuffled[validation_count + test_count:]
    return {"train": train, "validation": validation, "test": test}


def choose_ink_color(source_rgb, source_mask, rng):
    if rng.random() < 0.55:
        pixels = source_rgb[source_mask]
        if len(pixels):
            luma = (
                pixels[:, 0] * 0.299
                + pixels[:, 1] * 0.587
                + pixels[:, 2] * 0.114
            )
            darkest = pixels[luma <= np.percentile(luma, 45)]
            if len(darkest):
                color = np.median(darkest, axis=0)
                if color.mean() < 220:
                    return color.astype(np.float32)
    return INK_PALETTE[rng.randrange(len(INK_PALETTE))].copy()


def real_handwriting_layer(pairs, target_size, rng):
    for _ in range(20):
        image_path, label_path = rng.choice(pairs)
        with Image.open(image_path) as source:
            source_image = ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(label_path) as source:
            source_label = source.convert("L")
        if source_label.size != source_image.size:
            source_label = source_label.resize(
                source_image.size, Image.Resampling.NEAREST
            )
        mask = np.asarray(source_label, dtype=np.uint8) == 1
        coordinates = np.argwhere(mask)
        if not len(coordinates):
            continue
        center_y, center_x = coordinates[rng.randrange(len(coordinates))]
        source_min = min(source_image.size)
        if source_min < 12:
            continue
        side_low = min(source_min, max(12, source_min // 12))
        side_high = min(source_min, max(side_low, source_min // 3))
        side = rng.randint(side_low, side_high)
        left = max(0, min(source_image.size[0] - side, center_x - side // 2))
        top = max(0, min(source_image.size[1] - side, center_y - side // 2))
        box = (left, top, left + side, top + side)
        patch = np.asarray(source_image.crop(box), dtype=np.float32)
        patch_mask = mask[top:top + side, left:left + side]
        if int(patch_mask.sum()) < 12:
            continue

        extent = rng.randint(
            max(20, int(target_size * 0.07)),
            max(28, int(target_size * 0.24)),
        )
        scale = extent / float(max(patch.shape[:2]))
        target_width = max(1, int(round(patch.shape[1] * scale)))
        target_height = max(1, int(round(patch.shape[0] * scale)))
        resized_mask = cv2.resize(
            patch_mask.astype(np.uint8) * 255,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        color = choose_ink_color(
            patch.astype(np.uint8), patch_mask, rng
        )
        opacity = rng.uniform(0.68, 1.0)
        alpha = Image.fromarray(resized_mask).filter(
            ImageFilter.GaussianBlur(rng.uniform(0.35, 0.8))
        )
        angle = rng.uniform(-12.0, 12.0)
        alpha = alpha.rotate(
            angle, Image.Resampling.BILINEAR, expand=True, fillcolor=0
        )
        alpha_array = np.asarray(alpha, dtype=np.float32) / 255.0
        alpha_array *= opacity
        if np.count_nonzero(alpha_array > 0.12) < 10:
            continue
        return alpha_array, color
    return None


def random_scribble_layer(target_size, rng):
    extent = rng.randint(
        max(28, int(target_size * 0.08)),
        max(40, int(target_size * 0.25)),
    )
    width = rng.randint(extent, max(extent + 1, int(extent * 2.5)))
    height = rng.randint(max(20, extent // 2), max(24, int(extent * 1.3)))
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    stroke_count = rng.randint(1, 5)
    for _ in range(stroke_count):
        line_width = rng.randint(
            max(1, extent // 45), max(2, extent // 18)
        )
        point_count = rng.randint(3, 11)
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)
        points = [(x, y)]
        for _ in range(point_count - 1):
            x = max(0, min(width - 1, x + rng.randint(-extent, extent)))
            y = max(
                0, min(height - 1, y + rng.randint(-height // 2, height // 2))
            )
            points.append((x, y))
        draw.line(points, fill=255, width=line_width, joint="curve")
        radius = max(1, line_width // 2)
        for x, y in (points[0], points[-1]):
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=255,
            )
    if rng.random() < 0.35:
        y = rng.randint(height // 2, height - 1)
        draw.line(
            (0, y, width - 1, y + rng.randint(-3, 3)),
            fill=255,
            width=max(1, extent // 35),
        )
    canvas = canvas.rotate(
        rng.uniform(-15.0, 15.0),
        Image.Resampling.BILINEAR,
        expand=True,
        fillcolor=0,
    ).filter(ImageFilter.GaussianBlur(rng.uniform(0.25, 0.7)))
    alpha = np.asarray(canvas, dtype=np.float32) / 255.0
    alpha *= rng.uniform(0.65, 1.0)
    color = INK_PALETTE[rng.randrange(len(INK_PALETTE))].copy()
    return alpha, color


def find_placement(alpha, print_mask, mode, minimum_overlap, rng):
    hand = alpha > 0.12
    hand_coordinates = np.argwhere(hand)
    if not len(hand_coordinates):
        return None
    page_height, page_width = print_mask.shape
    layer_height, layer_width = hand.shape
    if layer_height > page_height or layer_width > page_width:
        return None
    print_coordinates = np.argwhere(print_mask)
    background_coordinates = np.argwhere(~print_mask)
    anchors = (
        print_coordinates if mode == "overlap" else background_coordinates
    )
    if not len(anchors):
        return None

    best = None
    for _ in range(80):
        page_y, page_x = anchors[rng.randrange(len(anchors))]
        hand_y, hand_x = hand_coordinates[rng.randrange(len(hand_coordinates))]
        left = int(page_x - hand_x + rng.randint(-4, 4))
        top = int(page_y - hand_y + rng.randint(-4, 4))
        left = max(0, min(page_width - layer_width, left))
        top = max(0, min(page_height - layer_height, top))
        region = print_mask[
            top:top + layer_height, left:left + layer_width
        ]
        ratio = float(np.count_nonzero(region & hand)) / max(
            1, int(hand.sum())
        )
        score = ratio if mode == "overlap" else -ratio
        if best is None or score > best[0]:
            best = (score, left, top, ratio)
        if mode == "overlap" and ratio >= minimum_overlap:
            break
        if mode == "background" and ratio <= 0.01:
            break
    if best is None:
        return None
    _, left, top, ratio = best
    if mode == "overlap" and ratio < minimum_overlap:
        return None
    return left, top, ratio


def composite_handwriting(clean, alpha, color, placement):
    clean_array = np.asarray(clean.convert("RGB"), dtype=np.float32)
    left, top, overlap_ratio = placement
    height, width = alpha.shape
    region = clean_array[top:top + height, left:left + width]
    alpha_rgb = alpha[..., None]
    region[:] = region * (1.0 - alpha_rgb) + color * alpha_rgb
    hand_mask = np.zeros(clean_array.shape[:2], dtype=bool)
    hand_mask[top:top + height, left:left + width] = alpha > 0.12
    return (
        Image.fromarray(np.clip(clean_array, 0, 255).astype(np.uint8)),
        hand_mask,
        overlap_ratio,
    )


def apply_high_resolution_command(composite, clean, command):
    if command is None:
        return composite, clean
    if "{input}" not in command or "{output}" not in command:
        raise ValueError(
            "--high-resolution-command needs {input} and {output}"
        )
    outputs = []
    with tempfile.TemporaryDirectory(prefix="hw-high-resolution-") as directory:
        directory = Path(directory)
        for name, image in (("input", composite), ("clean", clean)):
            source = directory / (name + "_source.png")
            destination = directory / (name + "_output.png")
            image.save(source)
            arguments = shlex.split(
                command.format(input=str(source), output=str(destination))
            )
            subprocess.run(arguments, check=True)
            if not destination.is_file():
                raise RuntimeError(
                    "high-resolution command did not create %s" % destination
                )
            with Image.open(destination) as result:
                outputs.append(result.convert("RGB").copy())
    if outputs[0].size != outputs[1].size:
        raise ValueError(
            "high-resolution input and clean outputs have different sizes"
        )
    return outputs[0], outputs[1]


def safe_name(path):
    value = re.sub(r"[^0-9A-Za-z_-]+", "_", path.stem).strip("_")
    return value[:48] or "document"


def save_variant(
    output,
    stem,
    composite,
    clean,
    label,
    print_mask,
):
    composite.save(output / "Images" / (stem + ".png"), compress_level=1)
    clean.save(
        output / "CleanTargets" / (stem + ".png"), compress_level=1
    )
    Image.fromarray(label.astype(np.uint8), mode="L").save(
        output / "Labels" / (stem + ".png"), compress_level=1
    )
    Image.fromarray(print_mask.astype(np.uint8) * 255, mode="L").save(
        output / "CleanPrintMasks" / (stem + ".png"), compress_level=1
    )


def synthesize_variant(
    clean,
    print_mask,
    pairs,
    mode,
    random_scribble_probability,
    minimum_overlap,
    high_resolution_command,
    rng,
):
    if mode == "negative":
        composite = clean.copy()
        hand_mask = np.zeros(print_mask.shape, dtype=bool)
        overlap_ratio = 0.0
        source_type = "none"
    else:
        result = None
        for _ in range(20):
            use_random = (
                not pairs
                or rng.random() < random_scribble_probability
            )
            layer = (
                random_scribble_layer(min(clean.size), rng)
                if use_random
                else real_handwriting_layer(pairs, min(clean.size), rng)
            )
            if layer is None:
                continue
            alpha, color = layer
            placement = find_placement(
                alpha, print_mask, mode, minimum_overlap, rng
            )
            if placement is None:
                continue
            composite, hand_mask, overlap_ratio = composite_handwriting(
                clean, alpha, color, placement
            )
            source_type = "procedural" if use_random else "real"
            result = True
            break
        if result is None:
            raise RuntimeError(
                "failed to place %s handwriting; check clean print masks"
                % mode
            )

    composite, processed_clean = apply_high_resolution_command(
        composite, clean, high_resolution_command
    )
    if composite.size != clean.size:
        new_size = composite.size
        hand_mask = np.asarray(
            Image.fromarray(hand_mask.astype(np.uint8)).resize(
                new_size, Image.Resampling.NEAREST
            )
        ) > 0
        print_mask = np.asarray(
            Image.fromarray(print_mask.astype(np.uint8)).resize(
                new_size, Image.Resampling.NEAREST
            )
        ) > 0
    label = print_mask.astype(np.uint8) * 2
    label[hand_mask] = 1
    return (
        composite, processed_clean, label, print_mask,
        overlap_ratio, source_type,
    )


def choose_mode(opts, rng):
    value = rng.random()
    if value < opts.clean_negative_ratio:
        return "negative"
    remaining = 1.0 - opts.clean_negative_ratio
    overlap_probability = opts.overlap_ratio / max(remaining, 1e-6)
    return "overlap" if rng.random() < overlap_probability else "background"


def validate_options(opts):
    probabilities = {
        "clean-negative-ratio": opts.clean_negative_ratio,
        "overlap-ratio": opts.overlap_ratio,
        "random-scribble-probability": opts.random_scribble_probability,
    }
    for name, value in probabilities.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError("--%s must be in [0, 1]" % name)
    if opts.clean_negative_ratio + opts.overlap_ratio > 1.0:
        raise ValueError(
            "clean-negative-ratio + overlap-ratio must not exceed 1"
        )
    if not 0.0 < opts.minimum_print_overlap <= 1.0:
        raise ValueError("--minimum-print-overlap must be in (0, 1]")
    if opts.variants_per_document <= 0:
        raise ValueError("--variants-per-document must be positive")
    if opts.preview_count < 0:
        raise ValueError("--preview-count must not be negative")
    if opts.workers <= 0:
        raise ValueError("--workers must be positive")


def write_provenance(output, records):
    content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records
    )
    (output / "provenance.jsonl").write_text(content, encoding="utf-8")


def write_preview(output, stems):
    if not stems:
        return
    thumbnail_size = (320, 220)
    rows = []
    label_palette = np.asarray(
        [[248, 248, 248], [36, 96, 210], [34, 34, 34]],
        dtype=np.uint8,
    )
    for stem in stems:
        with Image.open(output / "Images" / (stem + ".png")) as source:
            composite = source.convert("RGB")
        with Image.open(output / "CleanTargets" / (stem + ".png")) as source:
            clean = source.convert("RGB")
        with Image.open(output / "Labels" / (stem + ".png")) as source:
            labels = np.asarray(source.convert("L"), dtype=np.uint8)
        label_rgb = Image.fromarray(
            label_palette[np.clip(labels, 0, 2)], mode="RGB"
        )
        panels = []
        for image in (composite, clean, label_rgb):
            panel = ImageOps.contain(
                image, thumbnail_size, Image.Resampling.LANCZOS
            )
            canvas = Image.new("RGB", thumbnail_size, "white")
            canvas.paste(
                panel,
                (
                    (thumbnail_size[0] - panel.width) // 2,
                    (thumbnail_size[1] - panel.height) // 2,
                ),
            )
            panels.append(canvas)
        row = Image.new("RGB", (thumbnail_size[0] * 3, thumbnail_size[1]))
        for column, panel in enumerate(panels):
            row.paste(panel, (column * thumbnail_size[0], 0))
        rows.append(row)
    preview = Image.new(
        "RGB", (thumbnail_size[0] * 3, thumbnail_size[1] * len(rows)),
        "white",
    )
    for index, row in enumerate(rows):
        preview.paste(row, (0, index * thumbnail_size[1]))
    preview.save(output / "preview.jpg", quality=90)


def initialize_synthesis_worker(options, pairs, output):
    global _WORKER_OPTIONS, _WORKER_PAIRS, _WORKER_OUTPUT
    _WORKER_OPTIONS = argparse.Namespace(**options)
    _WORKER_PAIRS = [
        (Path(image_path), Path(label_path))
        for image_path, label_path in pairs
    ]
    _WORKER_OUTPUT = Path(output)
    # One OpenCV thread per process prevents workers from multiplying the CPU
    # thread count and making high-resolution pages slower.
    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)


def document_seed(base_seed, split, document_index):
    split_offset = {"train": 0, "validation": 1, "test": 2}[split]
    return (
        int(base_seed) * 1_000_003
        + split_offset * 100_000_007
        + int(document_index) * 97_409
    ) & 0xFFFFFFFF


def synthesize_document(task):
    if (
        _WORKER_OPTIONS is None
        or _WORKER_PAIRS is None
        or _WORKER_OUTPUT is None
    ):
        raise RuntimeError("synthesis worker was not initialized")
    split, document_index, path = task
    path = Path(path)
    opts = _WORKER_OPTIONS
    rng = random.Random(document_seed(opts.seed, split, document_index))
    with Image.open(path) as source:
        clean = ImageOps.exif_transpose(source).convert("RGB")
    print_mask = estimate_document_print_mask(clean)

    stems = []
    statistics = Counter()
    overlap_sum = 0.0
    overlap_count = 0
    provenance = []
    for variant in range(opts.variants_per_document):
        mode = choose_mode(opts, rng)
        (
            composite, target, label, variant_print,
            overlap_ratio, source_type,
        ) = synthesize_variant(
            clean,
            print_mask,
            _WORKER_PAIRS,
            mode,
            opts.random_scribble_probability,
            opts.minimum_print_overlap,
            opts.high_resolution_command,
            rng,
        )
        stem = "clean_%s_%05d_%02d_%s" % (
            split, document_index, variant, safe_name(path)
        )
        save_variant(
            _WORKER_OUTPUT,
            stem,
            composite,
            target,
            label,
            variant_print,
        )
        stems.append(stem)
        statistics["mode_" + mode] += 1
        statistics["source_" + source_type] += 1
        hand_pixels = int(np.count_nonzero(label == 1))
        print_pixels = int(np.count_nonzero(variant_print))
        hidden_print_pixels = int(np.count_nonzero(
            (label == 1) & variant_print
        ))
        statistics["pixels_handwriting"] += hand_pixels
        statistics["pixels_clean_print"] += print_pixels
        statistics["pixels_hidden_print"] += hidden_print_pixels
        if mode == "overlap":
            overlap_sum += float(overlap_ratio)
            overlap_count += 1
        provenance.append(
            {
                "stem": stem,
                "split": split,
                "clean_document": str(path),
                "mode": mode,
                "handwriting_source": source_type,
                "print_overlap": overlap_ratio,
                "handwriting_pixels": hand_pixels,
                "hidden_print_pixels": hidden_print_pixels,
            }
        )
    return {
        "split": split,
        "stems": stems,
        "statistics": dict(statistics),
        "overlap_sum": overlap_sum,
        "overlap_count": overlap_count,
        "provenance": provenance,
    }


def run_document_tasks(tasks, opts, pairs, output, total_samples):
    options = vars(opts).copy()
    serialized_pairs = [
        (str(image_path), str(label_path))
        for image_path, label_path in pairs
    ]
    results = []
    progress = tqdm(total=total_samples, desc="clean document synthesis")
    if opts.workers == 1:
        initialize_synthesis_worker(options, serialized_pairs, str(output))
        for task in tasks:
            result = synthesize_document(task)
            results.append(result)
            progress.update(len(result["stems"]))
    else:
        worker_count = min(opts.workers, len(tasks))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=initialize_synthesis_worker,
            initargs=(options, serialized_pairs, str(output)),
        ) as executor:
            futures = {
                executor.submit(synthesize_document, task): task
                for task in tasks
            }
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    progress.update(len(result["stems"]))
            except Exception:
                for future in futures:
                    future.cancel()
                raise
            finally:
                progress.close()
        return results
    progress.close()
    return results


def main():
    opts = get_argparser().parse_args()
    validate_options(opts)
    clean_files = collect_clean_images(opts.clean_roots)
    if opts.limit is not None:
        clean_files = clean_files[:opts.limit]
    requested_handwriting_roots = (
        opts.handwriting_roots or DEFAULT_HANDWRITING_ROOTS
    )
    if opts.handwriting_roots:
        handwriting_roots = resolve_handwriting_roots(
            requested_handwriting_roots
        )
    else:
        available_roots = [
            root for root in requested_handwriting_roots
            if Path(root).is_dir()
        ]
        handwriting_roots = (
            resolve_handwriting_roots(available_roots)
            if available_roots else []
        )
    output_path = Path(opts.output).expanduser().resolve()
    # When output lives under the supplied parent, do not accidentally reuse a
    # previous synthetic build as an "old handwriting" source on the next run.
    handwriting_roots = [
        root for root in handwriting_roots
        if root.resolve() != output_path
    ]
    pairs = (
        collect_handwriting_pairs(handwriting_roots)
        if handwriting_roots else []
    )
    if not pairs and opts.random_scribble_probability < 1.0:
        raise ValueError(
            "no old handwriting dataset is available; pass "
            "--handwriting-root or use --random-scribble-probability 1"
        )
    document_splits = split_documents(
        clean_files, opts.validation_ratio, opts.test_ratio, opts.seed
    )
    output = prepare_output(opts.output, overwrite=opts.overwrite)
    split_stems = {"train": [], "validation": [], "test": []}
    statistics = Counter()
    overlap_sum = 0.0
    overlap_count = 0
    provenance = []

    total_samples = sum(
        len(files) * opts.variants_per_document
        for files in document_splits.values()
    )
    tasks = []
    for split, files in document_splits.items():
        for document_index, path in enumerate(files):
            tasks.append((split, document_index, str(path)))
    results = run_document_tasks(
        tasks, opts, pairs, output, total_samples
    )
    for result in results:
        split_stems[result["split"]].extend(result["stems"])
        statistics.update(result["statistics"])
        overlap_sum += result["overlap_sum"]
        overlap_count += result["overlap_count"]
        provenance.extend(result["provenance"])
    for stems in split_stems.values():
        stems.sort()
    provenance.sort(key=lambda record: record["stem"])
    preview_stems = [
        record["stem"] for record in provenance[:opts.preview_count]
    ]

    write_splits(output, split_stems)
    write_provenance(output, provenance)
    write_preview(output, preview_stems)
    metadata = {
        "name": "CleanDocumentSynthetic",
        "clean_roots": [
            str(Path(root).resolve()) for root in opts.clean_roots
        ],
        "handwriting_roots": [
            str(Path(root).resolve()) for root in handwriting_roots
        ],
        "paired_clean_targets": True,
        "clean_target_directory": "CleanTargets",
        "clean_print_mask_directory": "CleanPrintMasks",
        "split_counts": {
            split: len(stems) for split, stems in split_stems.items()
        },
        "document_counts": {
            split: len(files) for split, files in document_splits.items()
        },
        "generation": {
            "variants_per_document": opts.variants_per_document,
            "clean_negative_ratio": opts.clean_negative_ratio,
            "overlap_ratio": opts.overlap_ratio,
            "random_scribble_probability": (
                opts.random_scribble_probability
            ),
            "minimum_print_overlap": opts.minimum_print_overlap,
            "high_resolution_command": opts.high_resolution_command,
            "seed": opts.seed,
            "workers": opts.workers,
            "effective_workers": min(opts.workers, len(tasks)),
        },
        "statistics": dict(statistics),
        "mean_actual_print_overlap": (
            overlap_sum / overlap_count if overlap_count else None
        ),
    }
    write_metadata(output, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("output: %s" % output)


if __name__ == "__main__":
    main()
