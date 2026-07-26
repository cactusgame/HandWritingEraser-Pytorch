"""Convert SignaTR6K RGB masks to the Baidu three-class label format."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
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


DEFAULT_SOURCE = "/Users/peng/Documents/data/HandWritingData/SignaTR6K"
DEFAULT_OUTPUT = "/Users/peng/Documents/data/HandWritingData/SignaTR6K-baidu-format"

BACKGROUND = (0, 0, 255)
HANDWRITING = (0, 255, 0)
PRINT = (255, 0, 0)
OVERLAP = (255, 255, 0)


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overlap-policy",
        choices=["handwriting", "print", "error"],
        default="handwriting",
        help="map yellow handwritten+printed overlap pixels",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="reuse completed image/label pairs in a partial output")
    parser.add_argument("--limit", type=int, default=None,
                        help="convert at most N samples per split (for testing)")
    return parser


def convert_mask(mask, overlap_policy="handwriting"):
    rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    output = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    color_to_id = {
        BACKGROUND: 0,
        HANDWRITING: 1,
        PRINT: 2,
    }
    if overlap_policy == "handwriting":
        color_to_id[OVERLAP] = 1
    elif overlap_policy == "print":
        color_to_id[OVERLAP] = 2

    for color, class_id in color_to_id.items():
        output[np.all(rgb == color, axis=2)] = class_id
    unknown = output == 255
    if np.any(unknown):
        colors, counts = np.unique(rgb[unknown], axis=0, return_counts=True)
        detail = sorted(
            zip(colors.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True
        )[:10]
        if overlap_policy == "error" and np.any(np.all(rgb == OVERLAP, axis=2)):
            raise ValueError("mask contains overlap pixels and policy is 'error'")
        raise ValueError("mask contains unsupported RGB colors: %s" % detail)
    return output


def convert_dataset(source, output, overlap_policy="handwriting", overwrite=False,
                    resume=False, limit=None):
    source = Path(source)
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    output = prepare_output(output, overwrite, resume)
    split_stems = {"train": [], "validation": [], "test": []}
    class_pixels = Counter()

    for split in ("train", "validation", "test"):
        images = index_by_stem(source / split / "crop")
        labels = index_by_stem(source / split / "label")
        if set(images) != set(labels):
            raise ValueError("%s crop/label stems do not match" % split)
        stems = sorted(images, key=lambda value: natural_key(Path(value)))
        if limit is not None:
            stems = stems[:limit]

        for stem in tqdm(stems, desc="SignaTR6K %s" % split):
            output_stem = "signatr6k_%s_%s" % (split, stem)
            image_destination = output / "Images" / (
                output_stem + images[stem].suffix.lower()
            )
            label_destination = output / "Labels" / (output_stem + ".png")
            if resume and image_destination.is_file() and label_destination.is_file():
                with Image.open(label_destination) as saved_label:
                    label = np.asarray(saved_label.convert("L"))
                if label.size == 0 or label.min() < 0 or label.max() > 2:
                    raise ValueError("invalid existing label while resuming: %s" % label_destination)
                counts = np.bincount(label.ravel(), minlength=3)
                class_pixels.update(
                    {index: int(value) for index, value in enumerate(counts)}
                )
                split_stems[split].append(output_stem)
                continue
            copy_image(images[stem], image_destination)
            label = convert_mask(Image.open(labels[stem]), overlap_policy)
            Image.fromarray(label, mode="L").save(label_destination, compress_level=1)
            counts = np.bincount(label.ravel(), minlength=3)
            class_pixels.update({index: int(value) for index, value in enumerate(counts)})
            split_stems[split].append(output_stem)

    write_splits(output, split_stems)
    write_metadata(
        output,
        {
            "name": "SignaTR6K",
            "source": str(source.resolve()),
            "split_counts": {key: len(value) for key, value in split_stems.items()},
            "class_pixels": {str(key): value for key, value in sorted(class_pixels.items())},
            "source_color_mapping": {
                "0,0,255": "background",
                "0,255,0": "handwriting",
                "255,0,0": "print",
                "255,255,0": "overlap -> %s" % overlap_policy,
            },
        },
    )
    return split_stems


def main():
    opts = get_argparser().parse_args()
    splits = convert_dataset(
        opts.source, opts.output, opts.overlap_policy, opts.overwrite,
        opts.resume, opts.limit
    )
    print("converted SignaTR6K: %s" % {key: len(value) for key, value in splits.items()})
    print("output: %s" % opts.output)


if __name__ == "__main__":
    main()
