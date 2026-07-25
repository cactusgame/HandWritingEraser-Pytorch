"""Materialize forced hand-over-print pairs for layered restoration training."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.restoration import HWRestorationDataset
from tools.conversion_utils import prepare_output, write_metadata, write_splits


DEFAULT_ROOTS = [
    "/Users/peng/Documents/data/HandWritingData/baidu",
    "/Users/peng/Documents/data/HandWritingData/SignaTR6K-baidu-format",
]


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", action="append", dest="source_roots",
        help="unpaired Baidu-format source; repeat for multiple domains",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument("--validation-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _build_sources(roots, split):
    sources = []
    for root in roots:
        dataset = HWRestorationDataset(
            root=root,
            split=split,
            transform=None,
            synthetic_probability=1.0,
            split_seed=1,
        )
        # Validation synthesis is intentional here: it draws base crops only
        # from each source's validation split, keeping it separate from train.
        dataset.synthetic_enabled = True
        sources.append(dataset)
    return sources


def _save_sample(output, stem, sample):
    image, label, clean, clean_print, valid = sample
    if float(valid) <= 0:
        return False
    if torch.is_tensor(label):
        label = Image.fromarray(label.cpu().numpy().astype(np.uint8))
    if torch.is_tensor(clean_print):
        clean_print = Image.fromarray(
            (clean_print.cpu().numpy() > 0).astype(np.uint8) * 255
        )
    image.save(output / "Images" / (stem + ".png"), compress_level=1)
    label.save(output / "Labels" / (stem + ".png"), compress_level=1)
    clean.save(
        output / "CleanTargets" / (stem + ".png"), compress_level=1
    )
    clean_print.point(lambda value: 255 if value > 0 else 0).save(
        output / "CleanPrintMasks" / (stem + ".png"), compress_level=1
    )
    return True


def generate_split(output, sources, split, count, random_state):
    stems = []
    attempts = 0
    maximum_attempts = max(100, count * 30)
    progress = tqdm(total=count, desc="layered synthetic/%s" % split)
    while len(stems) < count and attempts < maximum_attempts:
        attempts += 1
        source = sources[(attempts - 1) % len(sources)]
        index = random_state.randrange(len(source))
        sample = source[index]
        stem = "layered_%s_%07d" % (split, len(stems))
        if _save_sample(output, stem, sample):
            stems.append(stem)
            progress.update(1)
    progress.close()
    if len(stems) != count:
        raise RuntimeError(
            "only generated %d/%d %s samples after %d attempts"
            % (len(stems), count, split, attempts)
        )
    return stems


def main():
    opts = get_argparser().parse_args()
    if opts.overwrite and opts.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if opts.train_count <= 0 or opts.validation_count <= 0:
        raise ValueError("train and validation counts must be positive")
    roots = opts.source_roots or DEFAULT_ROOTS
    output = prepare_output(opts.output, opts.overwrite, opts.resume)
    random.seed(opts.seed)
    np.random.seed(opts.seed)

    split_stems = {"train": [], "validation": [], "test": []}
    for split, count in (
        ("train", opts.train_count),
        ("validation", opts.validation_count),
    ):
        sources = _build_sources(roots, split)
        split_stems[split] = generate_split(
            output, sources, split, count,
            random.Random(opts.seed + (0 if split == "train" else 1)),
        )
    write_splits(output, split_stems)
    write_metadata(
        output,
        {
            "name": "LayeredSynthetic",
            "source_roots": [str(Path(root).resolve()) for root in roots],
            "paired_clean_targets": True,
            "clean_target_directory": "CleanTargets",
            "clean_print_mask_directory": "CleanPrintMasks",
            "split_counts": {
                name: len(stems) for name, stems in split_stems.items()
            },
            "generation": {
                "method": (
                    "real handwriting alpha over real clean print crop, "
                    "forced hand-over-print overlap"
                ),
                "seed": opts.seed,
            },
        },
    )
    print(json.dumps(
        {name: len(stems) for name, stems in split_stems.items()},
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
