import random
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch.utils.data as data
from PIL import Image


class HWSegmentation(data.Dataset):
    """Paired document images and three-class segmentation labels."""

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

    def __init__(
        self,
        root,
        transform=None,
        train=True,
        val_ratio=0.15,
        split_seed=1,
        val_list=None,
    ):
        image_dir = Path(root) / "Images"
        label_dir = Path(root) / "Labels"
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
                "image/label stems do not match; missing labels=%s, missing images=%s"
                % (missing_labels[:5], missing_images[:5])
            )

        pairs = [(images[stem], labels[stem]) for stem in sorted(images)]
        if len(pairs) < 2:
            raise ValueError("at least two paired samples are required")
        if val_list:
            with open(val_list, "r", encoding="utf-8") as file:
                val_stems = {
                    Path(line.strip()).stem for line in file if line.strip()
                }
            unknown = sorted(val_stems - set(images))
            if unknown:
                raise ValueError("validation list contains unknown stems: %s" % unknown[:5])
            if not val_stems or len(val_stems) == len(pairs):
                raise ValueError("validation list must select some, but not all, samples")
            self.pairs = [
                pair for pair in pairs
                if ((pair[0].stem not in val_stems) if train else
                    (pair[0].stem in val_stems))
            ]
        else:
            if not 0.0 < val_ratio < 1.0:
                raise ValueError("val_ratio must be between 0 and 1")
            rng = random.Random(split_seed)
            rng.shuffle(pairs)
            val_count = max(1, min(len(pairs) - 1,
                                   int(round(len(pairs) * val_ratio))))
            self.pairs = pairs[val_count:] if train else pairs[:val_count]
        self.images = [str(pair[0]) for pair in self.pairs]
        self.targets = [str(pair[1]) for pair in self.pairs]
        self.transform = transform
        split_name = "train" if train else "validation"
        print("%s: %d paired samples" % (split_name, len(self.pairs)))

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

    @classmethod
    def encode_target(cls, target):
        target = np.asarray(target, dtype=np.int64)
        if target.size and target.max() >= len(cls.id_to_train_id):
            raise ValueError("label contains an unsupported class id: %d" % target.max())
        return cls.id_to_train_id[target]

    @classmethod
    def decode_target(cls, target):
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        image_path, target_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        target = Image.open(target_path).convert("L")
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
