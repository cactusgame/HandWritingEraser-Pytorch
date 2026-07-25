"""Conservative document restoration refinements for inference.

The neural model remains responsible for handwriting segmentation.  These
operations only touch its handwriting mask and use the source page as
additional evidence:

* replace implausible white fill with a locally inpainted paper background;
* preserve neutral dark source pixels connected to predicted print;
* bridge short print gaps along four common stroke directions.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class PostprocessDebug:
    hand_mask: np.ndarray
    estimated_background: np.ndarray
    background_repair: np.ndarray
    protected_source_ink: np.ndarray
    bridged_print: np.ndarray


def _as_rgb_array(image):
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an RGB image")
    return array


def _disk(radius):
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _dilate(mask, radius):
    if radius <= 0:
        return mask.astype(bool, copy=True)
    return cv2.dilate(mask.astype(np.uint8), _disk(radius)) > 0


def _line_kernels(max_gap):
    # Odd kernels make the current pixel the center of every direction.
    size = max(3, int(max_gap) + 2)
    if size % 2 == 0:
        size += 1
    horizontal = np.ones((1, size), dtype=np.uint8)
    vertical = horizontal.T
    diagonal = np.eye(size, dtype=np.uint8)
    anti_diagonal = np.fliplr(diagonal)
    return horizontal, vertical, diagonal, anti_diagonal


def _estimate_background(original, hand_mask, radius):
    """Fill the hand mask from its boundary while preserving local paper color."""
    if not np.any(hand_mask):
        return original.copy()
    bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    estimated = cv2.inpaint(
        bgr,
        hand_mask.astype(np.uint8) * 255,
        float(radius),
        cv2.INPAINT_TELEA,
    )
    return cv2.cvtColor(estimated, cv2.COLOR_BGR2RGB)


def _luma(rgb):
    value = rgb.astype(np.float32)
    return value[..., 0] * 0.299 + value[..., 1] * 0.587 + value[..., 2] * 0.114


def _background_repair_mask(restored, background, hand_mask):
    """Select bright generated pixels that disagree with a non-white background."""
    restored_luma = _luma(restored)
    background_luma = _luma(background)
    restored_chroma = (
        restored.astype(np.int16).max(axis=2)
        - restored.astype(np.int16).min(axis=2)
    )
    color_difference = np.max(
        np.abs(restored.astype(np.int16) - background.astype(np.int16)), axis=2
    )

    generated_white = (restored_luma >= 238.0) & (restored_chroma <= 18)
    background_is_not_white = (
        (background_luma <= 225.0)
        | (
            background.astype(np.int16).max(axis=2)
            - background.astype(np.int16).min(axis=2)
            >= 16
        )
    )
    return (
        hand_mask
        & generated_white
        & background_is_not_white
        & (color_difference >= 18)
    )


def _source_print_evidence(original, background, labels, hand_mask, print_class):
    original_luma = _luma(original)
    # The inpainted image intentionally keeps source pixels outside the hand
    # mask, including printed ink.  A grayscale closing supplies the local
    # paper tone on both sides of the mask so those print pixels have contrast.
    local_background_luma = cv2.morphologyEx(
        np.clip(original_luma, 0, 255).astype(np.uint8),
        cv2.MORPH_CLOSE,
        _disk(7),
    ).astype(np.float32)
    background_luma = np.maximum(_luma(background), local_background_luma)
    chroma = (
        original.astype(np.int16).max(axis=2)
        - original.astype(np.int16).min(axis=2)
    )
    dark = (background_luma - original_luma) >= 22.0
    neutral = chroma <= 28
    source_ink = dark & neutral
    print_seed = (labels == print_class) & source_ink & ~hand_mask
    return source_ink, print_seed


def _connected_source_ink(source_ink, print_seed, hand_mask, max_gap):
    """Grow print seeds through neutral source ink, bounded by a short distance."""
    if max_gap <= 0 or not np.any(print_seed):
        return np.zeros_like(hand_mask)
    permitted = source_ink & (hand_mask | print_seed)
    reached = print_seed.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(max_gap):
        grown = cv2.dilate(reached.astype(np.uint8), kernel) > 0
        reached |= grown & permitted
    return reached & hand_mask


def _directional_bridges(print_seed, hand_mask, max_gap):
    """Bridge only short gaps that lie inside the predicted handwriting mask."""
    if max_gap <= 0 or not np.any(print_seed):
        return np.zeros_like(hand_mask)
    seed = print_seed.astype(np.uint8)
    bridges = np.zeros_like(seed)
    for kernel in _line_kernels(max_gap):
        closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)
        bridges |= closed & (1 - seed)
    return (bridges > 0) & hand_mask


def _paint_bridge_colors(result, original, print_seed, bridge, max_gap):
    """Use nearby predicted-print color instead of introducing hard-coded black."""
    if not np.any(bridge):
        return result
    painted = result.copy()
    component_count, components = cv2.connectedComponents(
        bridge.astype(np.uint8), connectivity=8
    )
    search_kernel = _disk(max(1, max_gap + 1))
    for component_id in range(1, component_count):
        component = components == component_id
        neighborhood = (
            cv2.dilate(component.astype(np.uint8), search_kernel) > 0
        )
        samples = original[neighborhood & print_seed]
        if not len(samples):
            continue
        color = np.median(samples, axis=0).round().astype(np.uint8)
        painted[component] = color
    return painted


def _region_boxes(hand_mask, margin):
    """Group nearby mask components and return padded local processing boxes."""
    grouped = _dilate(hand_mask, margin)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        grouped.astype(np.uint8), connectivity=8
    )
    height, width = hand_mask.shape
    boxes = []
    for component_id in range(1, count):
        x, y, box_width, box_height, _ = stats[component_id]
        left = max(0, int(x) - margin)
        top = max(0, int(y) - margin)
        right = min(width, int(x + box_width) + margin)
        bottom = min(height, int(y + box_height) + margin)
        boxes.append((left, top, right, bottom))
    return boxes


def _refine_region(
    original,
    restored,
    labels,
    hand_mask,
    print_class,
    inpaint_radius,
    max_print_gap,
    mode,
):
    background = _estimate_background(original, hand_mask, inpaint_radius)
    background_repair = _background_repair_mask(
        restored, background, hand_mask
    )
    result = restored.copy()
    result[background_repair] = background[background_repair]

    protected_source = np.zeros_like(hand_mask)
    bridge = np.zeros_like(hand_mask)
    if mode == "balanced" and np.any(labels == print_class):
        source_ink, print_seed = _source_print_evidence(
            original, background, labels, hand_mask, print_class
        )
        protected_source = _connected_source_ink(
            source_ink, print_seed, hand_mask, max_print_gap
        )
        bridge = _directional_bridges(
            print_seed, hand_mask, max_print_gap
        )
        result[protected_source] = original[protected_source]
        bridge &= ~protected_source
        result = _paint_bridge_colors(
            result, original, print_seed, bridge, max_print_gap
        )
    return result, background, background_repair, protected_source, bridge


def refine_document_restoration(
    original,
    restored,
    labels,
    handwriting_class=1,
    print_class=2,
    mask_dilate=1,
    inpaint_radius=5,
    max_print_gap=7,
    mode="balanced",
    return_debug=False,
):
    """Refine a model result without changing model weights.

    ``background`` mode only corrects white fill over colored paper.
    ``balanced`` additionally protects source print and bridges short gaps.
    """
    if mode not in {"background", "balanced"}:
        raise ValueError("postprocess mode must be background or balanced")
    if mask_dilate < 0:
        raise ValueError("mask_dilate must be non-negative")
    if inpaint_radius <= 0:
        raise ValueError("inpaint_radius must be positive")
    if max_print_gap < 0:
        raise ValueError("max_print_gap must be non-negative")

    original_array = _as_rgb_array(original)
    restored_array = _as_rgb_array(restored)
    label_array = np.asarray(labels, dtype=np.uint8)
    if (
        original_array.shape != restored_array.shape
        or label_array.shape != original_array.shape[:2]
    ):
        raise ValueError("original, restored, and labels must share a size")

    raw_hand_mask = label_array == handwriting_class
    hand_mask = _dilate(raw_hand_mask, mask_dilate)
    if not np.any(hand_mask):
        debug = PostprocessDebug(
            hand_mask,
            original_array.copy(),
            np.zeros_like(hand_mask),
            np.zeros_like(hand_mask),
            np.zeros_like(hand_mask),
        )
        output = Image.fromarray(restored_array.copy())
        return (output, debug) if return_debug else output

    result = restored_array.copy()
    background = original_array.copy()
    background_repair = np.zeros_like(hand_mask)
    protected_source = np.zeros_like(hand_mask)
    bridge = np.zeros_like(hand_mask)
    margin = max(inpaint_radius + 2, max_print_gap + 2, 4)
    for left, top, right, bottom in _region_boxes(hand_mask, margin):
        slices = np.s_[top:bottom, left:right]
        (
            region_result,
            region_background,
            region_repair,
            region_protected,
            region_bridge,
        ) = _refine_region(
            original_array[slices],
            result[slices],
            label_array[slices],
            hand_mask[slices],
            print_class,
            inpaint_radius,
            max_print_gap,
            mode,
        )
        result[slices] = region_result
        background[slices] = region_background
        background_repair[slices] |= region_repair
        protected_source[slices] |= region_protected
        bridge[slices] |= region_bridge

    debug = PostprocessDebug(
        hand_mask,
        background,
        background_repair,
        protected_source,
        bridge,
    )
    output = Image.fromarray(result)
    return (output, debug) if return_debug else output
