import json
import random
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch.utils.data as data
from PIL import Image


class HWSegmentation(data.Dataset):
    """One Baidu-format handwriting dataset source.

    A source contains ``Images/`` and ``Labels/``. If ``splits/<split>.txt``
    exists it is authoritative; otherwise a deterministic grouped train/val
    split is generated. Grouping keeps Baidu variants such as ``*_00`` and
    ``*_01`` together and prevents augmented siblings leaking into validation.
    """

    HandWClass = namedtuple(
        "HandWClass",
        [
            "name",
            "id",
            "train_id",
            "category",
            "category_id",
            "has_instances",
            "ignore_in_eval",
            "color",
        ],
    )

    classes = [
        HandWClass("background", 0, 0, "void", 0, False, False, (255, 255, 255)),
        HandWClass("hand", 1, 1, "void", 0, False, False, (128, 64, 128)),
        HandWClass("print", 2, 2, "void", 0, False, False, (244, 35, 232)),
    ]
    train_id_to_color = np.array([c.color for c in classes])
    id_to_train_id = np.array([c.train_id for c in classes])
    valid_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    variant_suffix = re.compile(r"^(?P<group>.+)_\d{2}$")

    def __init__(
        self,
        root,
        transform=None,
        split=None,
        train=None,
        val_ratio=0.15,
        split_seed=1,
        val_list=None,
    ):
        self.root = Path(root)
        if split is None:
            split = "train" if train is not False else "validation"
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.split = split
        self.transform = transform
        self.dataset_name = self._read_dataset_name()

        image_dir = self.root / "Images"
        label_dir = self.root / "Labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                "dataset must contain Images/ and Labels/ under %s" % root
            )
        images = self._index_by_stem(image_dir)
        labels = self._index_by_stem(label_dir)
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        if missing_labels or missing_images:
            raise ValueError(
                "image/label stems do not match under %s; missing labels=%s, "
                "missing images=%s"
                % (root, missing_labels[:5], missing_images[:5])
            )
        if len(images) < 2:
            raise ValueError("at least two paired samples are required under %s" % root)

        group_variants = any(stem.startswith("dehw_train_") for stem in images)
        selected = self._select_stems(
            set(images), split, val_ratio, split_seed, val_list, group_variants
        )
        self.pairs = [(images[stem], labels[stem]) for stem in sorted(selected)]
        if not self.pairs:
            raise ValueError("%s split is empty for %s" % (split, root))
        self.images = [str(pair[0]) for pair in self.pairs]
        self.targets = [str(pair[1]) for pair in self.pairs]
        self.sample_ids = [pair[0].stem for pair in self.pairs]
        print("%s/%s: %d paired samples" %
              (self.dataset_name, self.split, len(self.pairs)))

    def _read_dataset_name(self):
        metadata_path = self.root / "dataset.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("name"):
                    return str(metadata["name"])
            except (OSError, ValueError):
                pass
        return self.root.name

    @classmethod
    def _index_by_stem(cls, directory):
        result = {}
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in cls.valid_suffixes:
                continue
            if path.stem in result:
                raise ValueError("duplicate file stem in %s: %s" % (directory, path.stem))
            result[path.stem] = path
        if not result:
            raise ValueError("no supported images found in %s" % directory)
        return result

    @staticmethod
    def _read_stem_list(path):
        return {
            Path(line.strip()).stem
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    @classmethod
    def _group_key(cls, stem, group_variants):
        if not group_variants:
            return stem
        match = cls.variant_suffix.match(stem)
        return match.group("group") if match else stem

    @classmethod
    def _generated_validation_stems(
        cls, all_stems, val_ratio, split_seed, group_variants=False
    ):
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1")
        groups = {}
        for stem in sorted(all_stems):
            groups.setdefault(cls._group_key(stem, group_variants), []).append(stem)
        if len(groups) < 2:
            raise ValueError("at least two source groups are required for train/validation")
        group_names = sorted(groups)
        random.Random(split_seed).shuffle(group_names)
        target = max(1, int(round(len(all_stems) * val_ratio)))
        selected = set()
        for group_name in group_names:
            if selected and len(selected) >= target:
                break
            selected.update(groups[group_name])
        if len(selected) == len(all_stems):
            selected.difference_update(groups[group_names[-1]])
        return selected

    def _select_stems(
        self, all_stems, split, val_ratio, split_seed, val_list, group_variants
    ):
        split_path = self.root / "splits" / (split + ".txt")
        if split_path.is_file():
            selected = self._read_stem_list(split_path)
        elif val_list:
            validation = self._read_stem_list(val_list)
            if split == "validation":
                selected = validation
            elif split == "train":
                selected = all_stems - validation
            else:
                raise ValueError("test split needs splits/test.txt")
        elif split in {"train", "validation"}:
            validation = self._generated_validation_stems(
                all_stems, val_ratio, split_seed, group_variants
            )
            selected = all_stems - validation if split == "train" else validation
        else:
            raise ValueError("test split needs %s" % split_path)

        unknown = sorted(selected - all_stems)
        if unknown:
            raise ValueError("%s contains unknown stems: %s" %
                             (split_path if split_path.is_file() else split, unknown[:5]))
        return selected

    @classmethod
    def encode_target(cls, target):
        target = np.asarray(target, dtype=np.int64)
        if target.size and (target.min() < 0 or target.max() >= len(cls.id_to_train_id)):
            raise ValueError(
                "label contains unsupported class ids: min=%d max=%d"
                % (target.min(), target.max())
            )
        return cls.id_to_train_id[target]

    @classmethod
    def decode_target(cls, target):
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        image_path, target_path = self.pairs[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(target_path) as source:
            target = source.convert("L")
        if image.size != target.size:
            raise ValueError(
                "image and label sizes differ for %s: %s vs %s"
                % (image_path.name, image.size, target.size)
            )
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, self.encode_target(target)

    def __len__(self):
        return len(self.pairs)


class MultiSourceHWSegmentation(data.ConcatDataset):
    """Concatenation with optional equal-probability domain sampling weights."""

    def __init__(self, datasets):
        if not datasets:
            raise ValueError("at least one dataset source is required")
        super().__init__(datasets)
        self.dataset_names = [
            getattr(dataset, "dataset_name", "source_%d" % index)
            for index, dataset in enumerate(datasets)
        ]

    def balanced_sample_weights(self, dataset_weights=None):
        if dataset_weights is None:
            dataset_weights = [1.0] * len(self.datasets)
        if len(dataset_weights) != len(self.datasets):
            raise ValueError("dataset weight count must match data-root count")
        if any(weight <= 0 for weight in dataset_weights):
            raise ValueError("dataset weights must be positive")
        sample_weights = []
        for dataset, source_weight in zip(self.datasets, dataset_weights):
            sample_weights.extend([source_weight / len(dataset)] * len(dataset))
        return sample_weights
