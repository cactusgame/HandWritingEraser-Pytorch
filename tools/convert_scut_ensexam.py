"""Convert paired SCUT-EnsExam pages to Baidu-style three-class masks."""

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

try:
    from .conversion_utils import (
        copy_image,
        index_by_stem,
        natural_key,
        prepare_output,
        write_metadata,
        write_splits,
    )
except ImportError:  # Allow running the file directly.
    from conversion_utils import (
        copy_image,
        index_by_stem,
        natural_key,
        prepare_output,
        write_metadata,
        write_splits,
    )


DEFAULT_SOURCE = "/Users/peng/Documents/data/HandWritingData/SCUT-EnsExam"
DEFAULT_OUTPUT = "/Users/peng/Documents/data/HandWritingData/SCUT-EnsExam-baidu-format"


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="fraction carved from the official training split")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print-local-threshold", type=int, default=12)
    parser.add_argument("--print-global-threshold", type=int, default=128)
    parser.add_argument("--background-radius", type=int, default=15)
    parser.add_argument("--hand-luma-threshold", type=int, default=10)
    parser.add_argument("--hand-color-threshold", type=int, default=22)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="reuse completed image/label pairs in a partial output")
    parser.add_argument("--limit", type=int, default=None,
                        help="convert at most N samples per source split (for testing)")
    return parser


def rgb_luma(rgb):
    rgb = rgb.astype(np.int32, copy=False)
    return (rgb[..., 0] * 299 + rgb[..., 1] * 587 + rgb[..., 2] * 114) // 1000


def load_box_mask(path, size):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    categories = Counter()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 9:
            raise ValueError("malformed annotation %s:%d" % (path, line_number))
        try:
            points = [(int(parts[index]), int(parts[index + 1])) for index in range(0, 8, 2)]
            category = int(parts[-1])
        except ValueError as error:
            raise ValueError("malformed annotation %s:%d" % (path, line_number)) from error
        if category not in (1, 2):
            raise ValueError("unsupported SCUT category %d in %s" % (category, path))
        draw.polygon(points, fill=255)
        categories[category] += 1
    return np.asarray(mask) > 0, categories


def count_annotation_categories(path):
    categories = Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            category = int(line.rsplit(",", 1)[-1].strip())
            if category not in (1, 2):
                raise ValueError("unsupported SCUT category %d in %s" % (category, path))
            categories[category] += 1
    return categories


def make_three_class_label(
    original,
    erased,
    box_mask,
    print_local_threshold=12,
    print_global_threshold=128,
    background_radius=15,
    hand_luma_threshold=10,
    hand_color_threshold=22,
    return_print_mask=False,
):
    # int16 is sufficient for channel differences and substantially reduces
    # peak memory on multi-megapixel pages; rgb_luma promotes before multiply.
    original_rgb = np.asarray(original.convert("RGB"), dtype=np.int16)
    erased_rgb = np.asarray(erased.convert("RGB"), dtype=np.int16)
    if original_rgb.shape != erased_rgb.shape:
        raise ValueError("original and erased page sizes differ")
    if box_mask.shape != original_rgb.shape[:2]:
        raise ValueError("box mask and page sizes differ")

    original_gray = rgb_luma(original_rgb)
    erased_gray = rgb_luma(erased_rgb)
    background = np.asarray(
        Image.fromarray(erased_gray.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(background_radius)
        ),
        dtype=np.int32,
    )
    printed = ((background - erased_gray) > print_local_threshold) | (
        erased_gray < print_global_threshold
    )

    removed_luma = erased_gray - original_gray
    color_difference = np.abs(erased_rgb - original_rgb).max(axis=2)
    handwriting = box_mask & (
        (removed_luma > hand_luma_threshold)
        | ((color_difference > hand_color_threshold) & (removed_luma > 2))
    )
    # Close one-pixel JPEG holes without expanding whole quadrilateral boxes.
    handwriting = np.asarray(
        Image.fromarray(handwriting.astype(np.uint8) * 255)
        .filter(ImageFilter.MaxFilter(3))
        .filter(ImageFilter.MinFilter(3))
    ) > 0

    label = np.zeros(original_rgb.shape[:2], dtype=np.uint8)
    label[printed] = 2
    label[handwriting] = 1  # Handwriting has priority on overlaps.
    if return_print_mask:
        return label, printed.astype(np.uint8)
    return label


def split_official_train(stems, val_ratio, seed):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val-ratio must be between 0 and 1")
    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * val_ratio))))
    validation = set(shuffled[:count])
    return {stem: ("validation" if stem in validation else "train") for stem in stems}


def convert_dataset(source, output, val_ratio=0.15, seed=1, overwrite=False,
                    resume=False, limit=None, **thresholds):
    source = Path(source)
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    output = prepare_output(output, overwrite, resume)
    split_stems = {"train": [], "validation": [], "test": []}
    class_pixels = Counter()
    annotation_categories = Counter()

    official_train = index_by_stem(source / "train" / "all_images")
    train_stems = sorted(official_train, key=lambda value: natural_key(Path(value)))
    if limit is not None:
        train_stems = train_stems[:limit]
    assigned_split = split_official_train(train_stems, val_ratio, seed)

    for source_split in ("train", "test"):
        images = index_by_stem(source / source_split / "all_images")
        erased = index_by_stem(source / source_split / "all_labels")
        annotations = source / source_split / "box_label_txt"
        if set(images) != set(erased):
            raise ValueError("%s all_images/all_labels stems do not match" % source_split)
        stems = sorted(images, key=lambda value: natural_key(Path(value)))
        if limit is not None:
            stems = stems[:limit]

        for stem in tqdm(stems, desc="SCUT-EnsExam %s" % source_split):
            destination_split = assigned_split[stem] if source_split == "train" else "test"
            output_stem = "scut_ensexam_%s_%s" % (source_split, stem)
            image_destination = output / "Images" / (
                output_stem + images[stem].suffix.lower()
            )
            label_destination = output / "Labels" / (output_stem + ".png")
            clean_destination = output / "CleanTargets" / (
                output_stem + erased[stem].suffix.lower()
            )
            print_destination = (
                output / "CleanPrintMasks" / (output_stem + ".png")
            )
            annotation_path = annotations / (stem + ".txt")
            if (
                resume
                and image_destination.is_file()
                and label_destination.is_file()
            ):
                if not clean_destination.is_file():
                    copy_image(erased[stem], clean_destination)
                if not print_destination.is_file():
                    original = Image.open(images[stem])
                    clean = Image.open(erased[stem])
                    box_mask, _ = load_box_mask(
                        annotation_path, original.size
                    )
                    _, clean_print = make_three_class_label(
                        original, clean, box_mask,
                        return_print_mask=True, **thresholds
                    )
                    Image.fromarray(clean_print, mode="L").save(
                        print_destination, compress_level=1
                    )
                with Image.open(label_destination) as saved_label:
                    label = np.asarray(saved_label.convert("L"))
                if label.size == 0 or label.min() < 0 or label.max() > 2:
                    raise ValueError("invalid existing label while resuming: %s" % label_destination)
                counts = np.bincount(label.ravel(), minlength=3)
                class_pixels.update(
                    {index: int(value) for index, value in enumerate(counts)}
                )
                annotation_categories.update(count_annotation_categories(annotation_path))
                split_stems[destination_split].append(output_stem)
                continue
            original = Image.open(images[stem])
            clean = Image.open(erased[stem])
            if original.size != clean.size:
                raise ValueError("size mismatch for SCUT sample %s" % stem)
            box_mask, categories = load_box_mask(
                annotation_path, original.size
            )
            annotation_categories.update(categories)
            label, clean_print = make_three_class_label(
                original, clean, box_mask,
                return_print_mask=True, **thresholds
            )

            copy_image(images[stem], image_destination)
            copy_image(erased[stem], clean_destination)
            Image.fromarray(label, mode="L").save(
                label_destination, compress_level=1
            )
            Image.fromarray(clean_print, mode="L").save(
                print_destination, compress_level=1
            )
            counts = np.bincount(label.ravel(), minlength=3)
            class_pixels.update({index: int(value) for index, value in enumerate(counts)})
            split_stems[destination_split].append(output_stem)

    write_splits(output, split_stems)
    write_metadata(
        output,
        {
            "name": "SCUT-EnsExam",
            "source": str(source.resolve()),
            "paired_clean_targets": True,
            "clean_target_directory": "CleanTargets",
            "clean_print_mask_directory": "CleanPrintMasks",
            "split_counts": {key: len(value) for key, value in split_stems.items()},
            "class_pixels": {str(key): value for key, value in sorted(class_pixels.items())},
            "annotation_categories": {
                "1_student_answer": annotation_categories[1],
                "2_teacher_correction": annotation_categories[2],
            },
            "label_generation": {
                "handwriting": "paired original/erased difference restricted by quadrilaterals",
                "print": "local contrast and global darkness in erased image",
                "thresholds": thresholds,
                "seed": seed,
                "validation_ratio": val_ratio,
            },
        },
    )
    return split_stems


def main():
    opts = get_argparser().parse_args()
    thresholds = {
        "print_local_threshold": opts.print_local_threshold,
        "print_global_threshold": opts.print_global_threshold,
        "background_radius": opts.background_radius,
        "hand_luma_threshold": opts.hand_luma_threshold,
        "hand_color_threshold": opts.hand_color_threshold,
    }
    splits = convert_dataset(
        opts.source,
        opts.output,
        opts.val_ratio,
        opts.seed,
        opts.overwrite,
        opts.resume,
        opts.limit,
        **thresholds,
    )
    print("converted SCUT-EnsExam: %s" % {key: len(value) for key, value in splits.items()})
    print("output: %s" % opts.output)


if __name__ == "__main__":
    main()
