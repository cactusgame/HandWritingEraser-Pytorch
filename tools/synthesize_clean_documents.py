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
_HANDWRITING_CACHE = {}


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
    parser.add_argument("--clean-negative-ratio", type=float, default=0.10)
    parser.add_argument(
        "--overlap-ratio", type=float, default=0.35,
        help=(
            "probability that one requested annotation is a small check/mark "
            "touching print; total annotation count does not change"
        ),
    )
    parser.add_argument(
        "--random-scribble-probability", type=float, default=0.08,
        help="probability that an annotated page contains one random scribble",
    )
    parser.add_argument("--minimum-print-overlap", type=float, default=0.04)
    parser.add_argument(
        "--min-annotations", type=int, default=3,
        help="minimum handwriting items on each non-negative page",
    )
    parser.add_argument(
        "--max-annotations", type=int, default=7,
        help="maximum handwriting items on each non-negative page",
    )
    parser.add_argument(
        "--print-mask-method",
        choices=("doc3d-adaptive", "doc3d-otsu"),
        default="doc3d-adaptive",
        help=(
            "CleanPrintMasks binarization; adaptive is safer for colored or "
            "uneven document backgrounds"
        ),
    )
    parser.add_argument(
        "--print-mask-block-size",
        type=int,
        default=11,
        help="odd adaptive-threshold neighborhood size (default: 11)",
    )
    parser.add_argument(
        "--print-mask-c",
        type=float,
        default=2.0,
        help="constant subtracted by adaptive thresholding (default: 2)",
    )
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


def is_generated_dataset_root(root):
    metadata_path = Path(root) / "dataset.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    name = str(metadata.get("name", "")).lower()
    return "synthetic" in name


def resolve_handwriting_roots(roots):
    """Resolve explicit dataset roots or their common parent directory."""
    resolved = []
    for requested in roots:
        requested = Path(requested).expanduser().resolve()
        if not requested.is_dir():
            raise FileNotFoundError(requested)
        if (
            is_handwriting_dataset_root(requested)
            and not is_generated_dataset_root(requested)
        ):
            candidates = [requested]
        else:
            candidates = sorted(
                child.resolve()
                for child in requested.iterdir()
                if (
                    child.is_dir()
                    and is_handwriting_dataset_root(child)
                    and not is_generated_dataset_root(child)
                )
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


def estimate_document_print_mask(
    image,
    method="doc3d-adaptive",
    block_size=11,
    threshold_c=2.0,
):
    """Binarize clean print using the Doc3D text-segmentation approach.

    This follows ``doc_clean/data/doc3d/gen_seg_text.py``: grayscale, a
    horizontal Gaussian blur, thresholding, and inversion so printed strokes
    are True.  No dilation or local-residual fill is used, which avoids fuzzy
    block-shaped supervision around anti-aliased glyphs.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if method == "doc3d-adaptive":
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("print-mask block size must be an odd integer >= 3")
        # The 11x1 kernel is intentionally horizontal, matching gen_seg_text:
        # suppress scan noise without smearing thin glyph edges vertically.
        blurred = cv2.GaussianBlur(gray, (11, 1), 0)
        background_binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(block_size),
            float(threshold_c),
        )
        mask = background_binary == 0

        # The original Doc3D data mostly contains dark print. Preserve its
        # behavior while also supporting reverse-white glyphs on dark panels.
        inverse_binary = cv2.adaptiveThreshold(
            255 - blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(block_size),
            float(threshold_c),
        )
        local_luma = cv2.GaussianBlur(
            gray, (0, 0), sigmaX=max(3.0, block_size / 2.0)
        )
        mask |= (inverse_binary == 0) & (local_luma < 175)
        return mask
    if method == "doc3d-otsu":
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, background_binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return background_binary == 0
    raise ValueError("unknown print-mask method: %s" % method)


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
    if rng.random() < 0.82:
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
                if color.mean() < 185:
                    return color.astype(np.float32)
    index = rng.choices(
        range(len(INK_PALETTE)),
        weights=(5.0, 6.0, 3.0, 0.8, 3.0, 0.25),
        k=1,
    )[0]
    return INK_PALETTE[index].copy()


def _extract_handwriting_alpha(source_rgb, coarse_mask):
    """Recover stroke-shaped alpha from a sometimes coarse class-1 mask.

    Several converted legacy datasets contain class-1 regions that are wider
    than the visible pen strokes.  The mask is therefore used as a search
    region, while local colour/luminance contrast in the source image decides
    the actual alpha.  This prevents rectangular label blobs from being pasted
    into otherwise clean documents.
    """
    lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    height, width = coarse_mask.shape
    sigma = max(1.4, min(height, width) / 420.0)
    local = cv2.GaussianBlur(lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    difference = lab - local
    luminance_contrast = np.abs(difference[..., 0])
    chroma_contrast = np.sqrt(np.square(difference[..., 1:]).sum(axis=2))
    # Chroma is especially useful for red/blue corrections drawn over black
    # printed text.  A low luminance threshold still retains light pencil.
    evidence = np.maximum(
        luminance_contrast / 13.0,
        chroma_contrast / 8.0,
    )
    alpha = np.clip((evidence - 0.12) / 0.88, 0.0, 1.0)
    alpha *= coarse_mask.astype(np.float32)

    visible = alpha >= 0.10
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        visible.astype(np.uint8), connectivity=8
    )
    valid = stats[:, cv2.CC_STAT_AREA] >= 3
    valid[0] = False
    alpha *= valid[components]
    return alpha


def _handwriting_regions(mask):
    """Group handwriting strokes into characters, words, and short lines."""
    height, width = mask.shape
    minimum = min(height, width)
    regions = set()
    # Multiple grouping radii produce both individual answers and short text
    # lines. The source mask contains handwriting only, so neighboring print is
    # never copied into a crop.
    for horizontal_scale, vertical_scale in (
        (0.004, 0.003),
        (0.009, 0.004),
        (0.018, 0.006),
        (0.035, 0.008),
    ):
        kernel_width = max(3, int(round(minimum * horizontal_scale)))
        kernel_height = max(3, int(round(minimum * vertical_scale)))
        joined = cv2.dilate(
            mask.astype(np.uint8),
            np.ones((kernel_height, kernel_width), dtype=np.uint8),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            joined, connectivity=8
        )
        for component_id in range(1, count):
            left = int(stats[component_id, cv2.CC_STAT_LEFT])
            top = int(stats[component_id, cv2.CC_STAT_TOP])
            box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            padding = max(2, minimum // 500)
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(width, left + box_width + padding * 2)
            bottom = min(height, top + box_height + padding * 2)
            original = mask[top:bottom, left:right]
            ink_pixels = int(np.count_nonzero(original))
            box_area = max(1, original.size)
            aspect = original.shape[1] / max(1.0, original.shape[0])
            if (
                ink_pixels >= 12
                and original.shape[0] >= 4
                and original.shape[1] >= 4
                and box_area <= mask.size * 0.18
                and ink_pixels / box_area <= 0.52
                and 0.18 <= aspect <= 16.0
            ):
                regions.add((left, top, right, bottom))
    return sorted(regions)


def _crop_nonzero_alpha(alpha):
    coordinates = np.argwhere(alpha > 0.02)
    if not len(coordinates):
        return alpha
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    return alpha[top:bottom, left:right]


def _region_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if not intersection:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


def _load_handwriting_assets(image_path, label_path, content_type):
    """Load one legacy page and retain only compact, reusable stroke crops."""
    cache_key = (str(image_path), str(label_path), content_type)
    cached = _HANDWRITING_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with Image.open(image_path) as source:
        source_image = ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(label_path) as source:
        source_label = source.convert("L")
    if source_label.size != source_image.size:
        source_label = source_label.resize(
            source_image.size, Image.Resampling.NEAREST
        )
    source_rgb = np.asarray(source_image, dtype=np.uint8).copy()
    coarse_mask = np.asarray(source_label, dtype=np.uint8) == 1
    stroke_alpha = _extract_handwriting_alpha(source_rgb, coarse_mask)
    regions = _handwriting_regions(stroke_alpha >= 0.10)
    source_minimum = min(stroke_alpha.shape)
    scored_regions = []
    for region in regions:
        left, top, right, bottom = region
        region_width = right - left
        region_height = bottom - top
        aspect = region_width / max(1.0, region_height)
        region_alpha = stroke_alpha[top:bottom, left:right]
        region_ink = region_alpha >= 0.10
        ink_pixels = int(np.count_nonzero(region_ink))
        ink_density = ink_pixels / max(1, region_ink.size)
        if content_type == "text":
            _, _, component_stats, _ = cv2.connectedComponentsWithStats(
                region_ink.astype(np.uint8), connectivity=8
            )
            substantial_components = int(np.count_nonzero(
                component_stats[1:, cv2.CC_STAT_AREA] >= 3
            ))
            component_areas = component_stats[
                1:, cv2.CC_STAT_AREA
            ]
            component_areas = component_areas[component_areas >= 3]
            occupied_bins = sum(
                bool(np.count_nonzero(section))
                for section in np.array_split(region_ink, 8, axis=1)
            )
            line_join_width = max(3, int(round(region_width * 0.035)))
            joined_line = cv2.dilate(
                region_ink.astype(np.uint8),
                np.ones((3, line_join_width), dtype=np.uint8),
            )
            line_count, _, line_stats, _ = (
                cv2.connectedComponentsWithStats(
                    joined_line, connectivity=8
                )
            )
            substantial_lines = int(np.count_nonzero(
                line_stats[1:, cv2.CC_STAT_AREA]
                >= max(12, region_width * 0.08)
            ))
            # Prefer complete words, Chinese phrases, and short answer lines.
            # Tiny fragments and dense crossed-out blobs do not represent the
            # normal answer text requested for most synthetic annotations.
            if (
                aspect < 3.0
                or region_width < source_minimum * 0.065
                or region_height < 6
                or ink_pixels < 48
                or not 0.012 <= ink_density <= 0.29
                or substantial_components < 2
                or occupied_bins < 5
                or np.percentile(component_areas, 75) < 8
                or substantial_lines != 1
            ):
                continue
            relative_width = region_width / source_minimum
            score = (
                min(18.0, aspect) ** 1.7
                * min(0.45, relative_width) ** 1.3
                * np.sqrt(ink_pixels)
            )
        else:
            if ink_pixels < 12:
                continue
            score = np.sqrt(ink_pixels)
        scored_regions.append((float(score), region))

    # Multiple dilation radii often return the same line several times. Keep
    # a small non-duplicate library; this also releases the full source page
    # before thousands of synthetic samples are generated.
    selected = []
    for score, region in sorted(scored_regions, reverse=True):
        if any(_region_iou(region, old_region) >= 0.55
               for _, old_region in selected):
            continue
        selected.append((score, region))
        if len(selected) >= 10:
            break
    if selected:
        quality_floor = selected[0][0] * (
            0.35 if content_type == "text" else 0.0
        )
        selected = [
            item for item in selected if item[0] >= quality_floor
        ]
    assets = []
    for score, (left, top, right, bottom) in selected:
        assets.append((
            source_rgb[top:bottom, left:right].copy(),
            stroke_alpha[top:bottom, left:right].copy(),
            score,
        ))
    if len(_HANDWRITING_CACHE) >= 32:
        _HANDWRITING_CACHE.pop(next(iter(_HANDWRITING_CACHE)))
    _HANDWRITING_CACHE[cache_key] = assets
    return assets


def real_handwriting_layer(pairs, target_size, rng, content_type="text"):
    if content_type == "text":
        text_pairs = [
            pair for pair in pairs
            if "signatr" not in str(pair[0]).lower()
        ]
        candidate_pairs = text_pairs or pairs
    else:
        candidate_pairs = pairs
    for _ in range(30):
        image_path, label_path = rng.choice(candidate_pairs)
        assets = _load_handwriting_assets(
            image_path, label_path, content_type
        )
        if not assets:
            continue
        patch, patch_alpha, _ = rng.choices(
            assets, weights=[asset[2] for asset in assets], k=1
        )[0]
        patch = patch.astype(np.float32)
        if np.count_nonzero(patch_alpha >= 0.10) < 12:
            continue

        if content_type == "text":
            minimum_height = max(22, int(target_size * 0.014))
            maximum_height = max(minimum_height + 1, int(target_size * 0.040))
            angle = rng.uniform(-3.5, 3.5)
        else:
            minimum_height = max(20, int(target_size * 0.012))
            maximum_height = max(minimum_height + 1, int(target_size * 0.032))
            angle = rng.uniform(-9.0, 9.0)
        target_height = rng.randint(minimum_height, maximum_height)
        scale = target_height / float(max(1, patch.shape[0]))
        target_width = max(1, int(round(patch.shape[1] * scale)))
        maximum_width = max(48, int(target_size * 0.48))
        if target_width > maximum_width:
            scale *= maximum_width / float(target_width)
            target_width = maximum_width
            target_height = max(1, int(round(patch.shape[0] * scale)))
        resized_mask = cv2.resize(
            np.uint8(np.clip(patch_alpha * 255.0, 0, 255)),
            (target_width, target_height),
            interpolation=(
                cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            ),
        )
        color = choose_ink_color(
            patch.astype(np.uint8), patch_alpha >= 0.10, rng
        )
        opacity = rng.uniform(0.68, 1.0)
        alpha = Image.fromarray(resized_mask).filter(
            ImageFilter.GaussianBlur(rng.uniform(0.20, 0.55))
        )
        alpha = alpha.rotate(
            angle, Image.Resampling.BILINEAR, expand=True, fillcolor=0
        )
        alpha_array = _crop_nonzero_alpha(
            np.asarray(alpha, dtype=np.float32) / 255.0
        )
        alpha_array *= opacity
        if np.count_nonzero(alpha_array > 0.12) < 10:
            continue
        return alpha_array, color
    return None


def random_scribble_layer(target_size, rng):
    extent = rng.randint(
        max(24, int(target_size * 0.025)),
        max(32, int(target_size * 0.070)),
    )
    width = rng.randint(extent, max(extent + 1, int(extent * 2.0)))
    height = rng.randint(max(16, extent // 2), max(20, int(extent * 1.1)))
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    stroke_count = rng.randint(1, 3)
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


def random_mark_layer(target_size, rng):
    height = rng.randint(
        max(22, int(target_size * 0.014)),
        max(30, int(target_size * 0.034)),
    )
    width = rng.randint(height, max(height + 1, int(height * 2.2)))
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, int(round(target_size * rng.uniform(0.0010, 0.0022))))
    mark_type = rng.choice(("check", "check", "cross", "underline", "circle"))
    if mark_type == "check":
        draw.line(
            [
                (int(width * 0.08), int(height * 0.52)),
                (int(width * 0.36), int(height * 0.86)),
                (int(width * 0.92), int(height * 0.10)),
            ],
            fill=255,
            width=line_width,
            joint="curve",
        )
    elif mark_type == "cross":
        draw.line(
            (int(width * 0.15), int(height * 0.12),
             int(width * 0.85), int(height * 0.88)),
            fill=255, width=line_width,
        )
        draw.line(
            (int(width * 0.82), int(height * 0.10),
             int(width * 0.18), int(height * 0.90)),
            fill=255, width=line_width,
        )
    elif mark_type == "underline":
        draw.line(
            (0, int(height * 0.65), width - 1, int(height * 0.75)),
            fill=255, width=line_width,
        )
    else:
        draw.ellipse(
            (
                line_width,
                line_width,
                width - line_width - 1,
                height - line_width - 1,
            ),
            outline=255,
            width=line_width,
        )
    canvas = canvas.rotate(
        rng.uniform(-8.0, 8.0),
        Image.Resampling.BILINEAR,
        expand=True,
        fillcolor=0,
    ).filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.45)))
    alpha = _crop_nonzero_alpha(
        np.asarray(canvas, dtype=np.float32) / 255.0
    )
    alpha *= rng.uniform(0.72, 1.0)
    return alpha, INK_PALETTE[rng.randrange(len(INK_PALETTE))].copy()


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


def _integral_sum(integral, left, top, right, bottom):
    return (
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    )


def _content_bounds(print_mask):
    height, width = print_mask.shape
    coordinates = np.argwhere(print_mask)
    if not len(coordinates):
        return (
            int(width * 0.06),
            int(height * 0.05),
            int(width * 0.94),
            int(height * 0.95),
        )
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    margin_x = max(int(width * 0.035), 8)
    margin_y = max(int(height * 0.025), 8)
    return (
        max(int(width * 0.035), int(left) - margin_x),
        max(int(height * 0.025), int(top) - margin_y),
        min(int(width * 0.965), int(right) + margin_x),
        min(int(height * 0.975), int(bottom) + margin_y),
    )


def analyze_document_layout(print_mask):
    height, width = print_mask.shape
    minimum_line = max(84, int(round(width * 0.050)))
    horizontal = cv2.morphologyEx(
        print_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((1, minimum_line), dtype=np.uint8),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        horizontal, connectivity=8
    )
    answer_lines = []
    for component_id in range(1, count):
        left = int(stats[component_id, cv2.CC_STAT_LEFT])
        top = int(stats[component_id, cv2.CC_STAT_TOP])
        line_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        line_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        band_height = max(16, int(round(height * 0.018)))
        above = print_mask[
            max(0, top - band_height):top,
            left:left + line_width,
        ]
        above_density = float(above.mean()) if above.size else 1.0
        if (
            line_width >= minimum_line
            and line_width <= width * 0.78
            and line_height <= max(6, int(height * 0.003))
            and line_width / max(1.0, line_height) >= 15.0
            and above_density <= 0.055
        ):
            answer_lines.append(
                (left, top + line_height // 2, left + line_width)
            )
    return {
        "bounds": _content_bounds(print_mask),
        "print_integral": cv2.integral(print_mask.astype(np.uint8)),
        "answer_lines": answer_lines,
    }


def find_answer_line_placement(
    alpha,
    print_mask,
    occupied_mask,
    layout,
    rng,
):
    """Place handwriting on detected underlines, as a student would answer."""
    alpha = _crop_nonzero_alpha(alpha)
    if not layout["answer_lines"]:
        return None
    candidates = list(layout["answer_lines"])
    rng.shuffle(candidates)
    best = None
    for line_left, line_y, line_right in candidates:
        line_width = line_right - line_left
        candidate_alpha = alpha
        if candidate_alpha.shape[1] > line_width * 0.92:
            scale = line_width * 0.92 / candidate_alpha.shape[1]
            if scale < 0.45:
                continue
            new_width = max(1, int(round(candidate_alpha.shape[1] * scale)))
            new_height = max(1, int(round(candidate_alpha.shape[0] * scale)))
            candidate_alpha = cv2.resize(
                candidate_alpha,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )
        layer_height, layer_width = candidate_alpha.shape
        page_height, page_width = print_mask.shape
        if layer_width >= page_width or layer_height >= page_height:
            continue
        horizontal_room = max(0, line_width - layer_width)
        left = line_left + (
            rng.randint(0, horizontal_room) if horizontal_room else 0
        )
        left = max(0, min(page_width - layer_width, left))
        # The handwriting baseline rests just above and slightly touches the
        # printed answer line.
        top = int(round(line_y - layer_height * rng.uniform(0.96, 1.08)))
        top = max(0, min(page_height - layer_height, top))
        hand = candidate_alpha > 0.12
        hand_pixels = int(hand.sum())
        if hand_pixels < 5:
            continue
        region_print = print_mask[
            top:top + layer_height, left:left + layer_width
        ]
        region_occupied = occupied_mask[
            top:top + layer_height, left:left + layer_width
        ]
        overlap = float(np.count_nonzero(region_print & hand)) / hand_pixels
        occupied = float(
            np.count_nonzero(region_occupied & hand)
        ) / hand_pixels
        if overlap > 0.10 or occupied > 0.02:
            continue
        fit = min(1.0, line_width / max(1.0, layer_width))
        score = fit - overlap * 4.0 - occupied * 10.0
        if best is None or score > best[0]:
            best = (
                score,
                candidate_alpha,
                (left, top, overlap),
            )
    if best is None:
        return None
    return best[1], best[2]


def find_contextual_placement(
    alpha,
    print_mask,
    occupied_mask,
    kind,
    minimum_overlap,
    rng,
    layout=None,
):
    """Place answers in nearby whitespace and reserve overlap for small marks."""
    alpha = _crop_nonzero_alpha(alpha)
    hand = alpha > 0.12
    hand_pixels = int(hand.sum())
    if hand_pixels < 5:
        return None
    page_height, page_width = print_mask.shape
    layer_height, layer_width = hand.shape
    if layer_height >= page_height or layer_width >= page_width:
        return None
    layout = layout or analyze_document_layout(print_mask)
    bound_left, bound_top, bound_right, bound_bottom = layout["bounds"]
    maximum_left = min(page_width - layer_width, bound_right - layer_width)
    maximum_top = min(page_height - layer_height, bound_bottom - layer_height)
    if maximum_left < bound_left or maximum_top < bound_top:
        return None

    print_integral = layout["print_integral"]
    occupied_integral = cv2.integral(occupied_mask.astype(np.uint8))
    candidates = []
    for _ in range(320):
        left = rng.randint(bound_left, maximum_left)
        top = rng.randint(bound_top, maximum_top)
        right = left + layer_width
        bottom = top + layer_height
        box_area = max(1, layer_width * layer_height)
        box_print = _integral_sum(
            print_integral, left, top, right, bottom
        ) / box_area
        box_occupied = _integral_sum(
            occupied_integral, left, top, right, bottom
        ) / box_area
        pad_x = max(12, layer_width // 3)
        pad_y = max(12, layer_height)
        outer_left = max(0, left - pad_x)
        outer_top = max(0, top - pad_y)
        outer_right = min(page_width, right + pad_x)
        outer_bottom = min(page_height, bottom + pad_y)
        outer_area = (
            (outer_right - outer_left) * (outer_bottom - outer_top)
        )
        context_area = max(1, outer_area - box_area)
        context_print = (
            _integral_sum(
                print_integral,
                outer_left,
                outer_top,
                outer_right,
                outer_bottom,
            )
            - box_print * box_area
        ) / context_area

        if kind == "answer_text":
            # Real answers sit in whitespace, but usually near a question,
            # underline, table cell, or other printed context.
            score = (
                context_print * 8.0
                - box_print * 14.0
                - box_occupied * 20.0
            )
            if box_print > 0.075 or box_occupied > 0.025:
                continue
        elif kind == "mark":
            score = (
                context_print * 4.0
                - abs(box_print - 0.10) * 5.0
                - box_occupied * 20.0
            )
            if box_print > 0.35 or box_occupied > 0.04:
                continue
        else:
            score = (
                context_print * 5.0
                - abs(box_print - 0.035) * 4.0
                - box_occupied * 20.0
            )
            if box_print > 0.22 or box_occupied > 0.04:
                continue
        candidates.append((score, left, top))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    best = None
    for coarse_score, left, top in candidates[:48]:
        region_print = print_mask[
            top:top + layer_height, left:left + layer_width
        ]
        region_occupied = occupied_mask[
            top:top + layer_height, left:left + layer_width
        ]
        overlap_ratio = float(np.count_nonzero(
            region_print & hand
        )) / hand_pixels
        occupied_ratio = float(np.count_nonzero(
            region_occupied & hand
        )) / hand_pixels
        if occupied_ratio > 0.02:
            continue
        if kind == "answer_text":
            if overlap_ratio > 0.045:
                continue
            score = coarse_score - overlap_ratio * 18.0
        elif kind == "mark":
            if overlap_ratio < minimum_overlap or overlap_ratio > 0.55:
                continue
            score = coarse_score - abs(overlap_ratio - 0.16) * 4.0
        else:
            if overlap_ratio > 0.28:
                continue
            score = coarse_score - abs(overlap_ratio - 0.04) * 3.0
        if best is None or score > best[0]:
            best = (score, left, top, overlap_ratio)
    if best is None:
        return None
    _, left, top, overlap_ratio = best
    return alpha, (left, top, overlap_ratio)


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


def synthesize_page_variant(clean, print_mask, pairs, opts, rng):
    """Create a realistic page with several answers and occasional marks."""
    if rng.random() < opts.clean_negative_ratio:
        composite = clean.copy()
        hand_mask = np.zeros(print_mask.shape, dtype=bool)
        annotation_types = []
        source_types = []
    else:
        annotation_count = rng.randint(
            opts.min_annotations, opts.max_annotations
        )
        annotation_types = [
            "answer_text" if pairs else "scribble"
            for _ in range(annotation_count)
        ]
        if rng.random() < opts.overlap_ratio:
            annotation_types[rng.randrange(len(annotation_types))] = "mark"
        if rng.random() < opts.random_scribble_probability:
            replaceable = [
                index for index, kind in enumerate(annotation_types)
                if kind == "answer_text"
            ]
            if replaceable:
                annotation_types[rng.choice(replaceable)] = "scribble"

        composite_array = np.asarray(
            clean.convert("RGB"), dtype=np.float32
        ).copy()
        hand_mask = np.zeros(print_mask.shape, dtype=bool)
        layout = analyze_document_layout(print_mask)
        placed_types = []
        source_types = []
        for requested_kind in annotation_types:
            placed = None
            for _ in range(10):
                if requested_kind == "answer_text":
                    layer = (
                        real_handwriting_layer(
                            pairs, min(clean.size), rng, content_type="text"
                        )
                        if pairs else None
                    )
                    source_type = "real_text"
                elif requested_kind == "mark":
                    layer = random_mark_layer(min(clean.size), rng)
                    source_type = "procedural_mark"
                else:
                    layer = random_scribble_layer(min(clean.size), rng)
                    source_type = "procedural_scribble"
                if layer is None:
                    continue
                alpha, color = layer
                placement_result = None
                if (
                    requested_kind == "answer_text"
                    and layout["answer_lines"]
                ):
                    placement_result = find_answer_line_placement(
                        alpha,
                        print_mask,
                        hand_mask,
                        layout,
                        rng,
                    )
                if placement_result is None:
                    placement_result = find_contextual_placement(
                        alpha,
                        print_mask,
                        hand_mask,
                        requested_kind,
                        opts.minimum_print_overlap,
                        rng,
                        layout=layout,
                    )
                if placement_result is None:
                    continue
                alpha, placement = placement_result
                left, top, _ = placement
                layer_height, layer_width = alpha.shape
                region = composite_array[
                    top:top + layer_height, left:left + layer_width
                ]
                alpha_rgb = alpha[..., None]
                region[:] = (
                    region * (1.0 - alpha_rgb) + color * alpha_rgb
                )
                local_hand = alpha > 0.12
                hand_mask[
                    top:top + layer_height, left:left + layer_width
                ] |= local_hand
                placed = True
                placed_types.append(requested_kind)
                source_types.append(source_type)
                break
            # A page may expose fewer suitable answer regions than requested.
            # Skipping is safer than putting handwriting over arbitrary text.
            if placed is None:
                continue
        annotation_types = placed_types
        composite = Image.fromarray(
            np.clip(composite_array, 0, 255).astype(np.uint8)
        )

    composite, processed_clean = apply_high_resolution_command(
        composite, clean, opts.high_resolution_command
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
    hand_pixels = int(hand_mask.sum())
    overlap_ratio = (
        float(np.count_nonzero(hand_mask & print_mask)) / hand_pixels
        if hand_pixels else 0.0
    )
    mode = "annotated" if hand_pixels else "negative"
    return (
        composite,
        processed_clean,
        label,
        print_mask,
        overlap_ratio,
        source_types,
        annotation_types,
        mode,
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
    if not 0.0 < opts.minimum_print_overlap <= 1.0:
        raise ValueError("--minimum-print-overlap must be in (0, 1]")
    if opts.variants_per_document <= 0:
        raise ValueError("--variants-per-document must be positive")
    if opts.min_annotations <= 0:
        raise ValueError("--min-annotations must be positive")
    if opts.max_annotations < opts.min_annotations:
        raise ValueError(
            "--max-annotations must be at least --min-annotations"
        )
    if (
        opts.print_mask_block_size < 3
        or opts.print_mask_block_size % 2 == 0
    ):
        raise ValueError(
            "--print-mask-block-size must be an odd integer >= 3"
        )
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
    _HANDWRITING_CACHE.clear()
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
    print_mask = estimate_document_print_mask(
        clean,
        method=opts.print_mask_method,
        block_size=opts.print_mask_block_size,
        threshold_c=opts.print_mask_c,
    )

    stems = []
    statistics = Counter()
    overlap_sum = 0.0
    overlap_count = 0
    provenance = []
    for variant in range(opts.variants_per_document):
        (
            composite, target, label, variant_print,
            overlap_ratio, source_types, annotation_types, mode,
        ) = synthesize_page_variant(
            clean,
            print_mask,
            _WORKER_PAIRS,
            opts,
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
        for source_type in source_types:
            statistics["source_" + source_type] += 1
        for annotation_type in annotation_types:
            statistics["annotation_" + annotation_type] += 1
        statistics["annotations_total"] += len(annotation_types)
        hand_pixels = int(np.count_nonzero(label == 1))
        print_pixels = int(np.count_nonzero(variant_print))
        hidden_print_pixels = int(np.count_nonzero(
            (label == 1) & variant_print
        ))
        statistics["pixels_handwriting"] += hand_pixels
        statistics["pixels_clean_print"] += print_pixels
        statistics["pixels_hidden_print"] += hidden_print_pixels
        if hand_pixels:
            overlap_sum += float(overlap_ratio)
            overlap_count += 1
        provenance.append(
            {
                "stem": stem,
                "split": split,
                "clean_document": str(path),
                "mode": mode,
                "handwriting_sources": source_types,
                "annotation_types": annotation_types,
                "annotation_count": len(annotation_types),
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
            "mark_page_probability": opts.overlap_ratio,
            "scribble_page_probability": opts.random_scribble_probability,
            "min_annotations": opts.min_annotations,
            "max_annotations": opts.max_annotations,
            "minimum_print_overlap": opts.minimum_print_overlap,
            "print_mask_method": opts.print_mask_method,
            "print_mask_block_size": opts.print_mask_block_size,
            "print_mask_c": opts.print_mask_c,
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
