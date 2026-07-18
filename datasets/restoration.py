"""Joint segmentation/restoration datasets and aligned document transforms."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils import data
from torchvision.transforms import functional as TF

from .handwriting import HWSegmentation
from utils.ext_transforms import ExtRandomDocumentDegradation


class JointDocumentTransform:
    """Apply identical geometry/color changes to input and clean target."""

    def __init__(
        self,
        crop_size=640,
        scale_range=(0.75, 1.5),
        foreground_probability=0.75,
        train=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.crop_size = int(crop_size)
        self.scale_range = scale_range
        self.foreground_probability = foreground_probability
        self.train = train
        self.mean = mean
        self.std = std
        self.degradation = ExtRandomDocumentDegradation(p=0.6)

    @staticmethod
    def _color_pair(image, clean):
        brightness = random.uniform(0.75, 1.25)
        contrast = random.uniform(0.75, 1.25)
        saturation = random.uniform(0.85, 1.15)
        operations = [
            lambda value: TF.adjust_brightness(value, brightness),
            lambda value: TF.adjust_contrast(value, contrast),
            lambda value: TF.adjust_saturation(value, saturation),
        ]
        random.shuffle(operations)
        for operation in operations:
            image, clean = operation(image), operation(clean)
        return image, clean

    def _random_crop(self, image, label, clean):
        size = self.crop_size
        width, height = image.size
        pad_w, pad_h = max(0, size - width), max(0, size - height)
        if pad_w or pad_h:
            padding = (
                pad_w // 2,
                pad_h // 2,
                pad_w - pad_w // 2,
                pad_h - pad_h // 2,
            )
            image = ImageOps.expand(image, padding, fill=(255, 255, 255))
            clean = ImageOps.expand(clean, padding, fill=(255, 255, 255))
            label = ImageOps.expand(label, padding, fill=0)
            width, height = image.size

        def candidate():
            top = random.randint(0, height - size)
            left = random.randint(0, width - size)
            return top, left

        top, left = candidate()
        if random.random() < self.foreground_probability:
            for _ in range(10):
                try_top, try_left = candidate()
                crop = np.asarray(
                    label.crop((try_left, try_top, try_left + size, try_top + size))
                )
                top, left = try_top, try_left
                if np.count_nonzero(crop == 1) >= size * size * 0.001:
                    break
        box = (left, top, left + size, top + size)
        return image.crop(box), label.crop(box), clean.crop(box)

    def __call__(self, image, label, clean):
        if self.train:
            image, clean = self._color_pair(image, clean)
            if random.random() < 0.35:
                angle = random.uniform(-2.0, 2.0)
                image = image.rotate(
                    angle, Image.BILINEAR, fillcolor=(255, 255, 255)
                )
                clean = clean.rotate(
                    angle, Image.BILINEAR, fillcolor=(255, 255, 255)
                )
                label = label.rotate(angle, Image.NEAREST, fillcolor=0)

            scale = random.uniform(*self.scale_range)
            width, height = image.size
            target = (max(1, int(round(width * scale))),
                      max(1, int(round(height * scale))))
            image = image.resize(target, Image.BILINEAR)
            clean = clean.resize(target, Image.BILINEAR)
            label = label.resize(target, Image.NEAREST)

            width, height = image.size
            min_scale = max(
                self.crop_size / float(width),
                self.crop_size / float(height),
                1.0,
            )
            if min_scale > 1.0:
                target = (
                    int(round(width * min_scale)),
                    int(round(height * min_scale)),
                )
                image = image.resize(target, Image.BILINEAR)
                clean = clean.resize(target, Image.BILINEAR)
                label = label.resize(target, Image.NEAREST)
            image, label, clean = self._random_crop(image, label, clean)
            image, _ = self.degradation(image, label)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, self.mean, self.std)
        clean_tensor = TF.to_tensor(clean)
        label_tensor = torch.from_numpy(
            np.asarray(label, dtype=np.int64).copy()
        )
        return image_tensor, label_tensor, clean_tensor


class HWRestorationDataset(data.Dataset):
    """Baidu-format segmentation data with optional paired clean RGB targets.

    Sources without ``CleanTargets/`` remain useful for segmentation. Their
    clean target is a placeholder and ``restoration_valid`` is zero, so they
    never contribute to image-reconstruction losses.
    """

    def __init__(
        self, root, transform=None, synthetic_probability=0.5, **split_options
    ):
        self.base = HWSegmentation(root, transform=None, **split_options)
        self.root = Path(root)
        self.transform = transform
        if not 0.0 <= synthetic_probability <= 1.0:
            raise ValueError("synthetic_probability must be in [0, 1]")
        self.synthetic_probability = synthetic_probability
        self.dataset_name = self.base.dataset_name
        clean_dir = self.root / "CleanTargets"
        self.clean_targets = {}
        if clean_dir.is_dir():
            for path in clean_dir.iterdir():
                if path.is_file() and path.suffix.lower() in self.base.valid_suffixes:
                    if path.stem in self.clean_targets:
                        raise ValueError("duplicate clean target stem: %s" % path.stem)
                    self.clean_targets[path.stem] = path
        self.paired_count = sum(
            image_path.stem in self.clean_targets
            for image_path, _ in self.base.pairs
        )
        if self.paired_count not in (0, len(self.base)):
            missing = [
                image_path.stem for image_path, _ in self.base.pairs
                if image_path.stem not in self.clean_targets
            ]
            raise ValueError(
                "CleanTargets is incomplete for %s/%s: %s"
                % (self.dataset_name, self.base.split, missing[:5])
            )
        print(
            "%s/%s: %d paired clean targets"
            % (self.dataset_name, self.base.split, self.paired_count)
        )
        self.synthetic_enabled = (
            self.base.split == "train"
            and self.paired_count == 0
            and self.synthetic_probability > 0
        )
        if self.synthetic_enabled:
            print(
                "%s/train: online clean-region handwriting synthesis enabled (p=%.2f)"
                % (self.dataset_name, self.synthetic_probability)
            )

    @property
    def split(self):
        return self.base.split

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _best_clean_crop(image, label, attempts=100):
        width, height = image.size
        # Smaller clean windows are easier to find on densely answered pages;
        # they are later scaled to the requested training crop size.
        side = max(48, int(round(min(width, height) * 0.25)))
        side = min(side, width, height)
        label_array = np.asarray(label)
        best = None
        for _ in range(attempts):
            left = random.randint(0, width - side)
            top = random.randint(0, height - side)
            hand_count = np.count_nonzero(
                label_array[top:top + side, left:left + side] == 1
            )
            print_count = np.count_nonzero(
                label_array[top:top + side, left:left + side] == 2
            )
            score = (hand_count, -print_count)
            if best is None or score < best[:2]:
                best = (hand_count, -print_count, left, top, side)
        if (
            best is None
            or best[0] > side * side * 0.0005
            or -best[1] < side * side * 0.002
        ):
            return None
        _, _, left, top, side = best
        box = (left, top, left + side, top + side)
        return image.crop(box), label.crop(box)

    def _load_random_ink_patch(self):
        for _ in range(12):
            image_path, label_path = random.choice(self.base.pairs)
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            with Image.open(label_path) as source:
                label = source.convert("L")
            mask = np.asarray(label) == 1
            coordinates = np.argwhere(mask)
            if not len(coordinates):
                continue
            center_y, center_x = coordinates[random.randrange(len(coordinates))]
            min_dimension = min(image.size)
            side = random.randint(
                max(24, min_dimension // 10), max(32, min_dimension // 3)
            )
            left = max(0, min(image.size[0] - side, int(center_x - side // 2)))
            top = max(0, min(image.size[1] - side, int(center_y - side // 2)))
            box = (left, top, left + side, top + side)
            patch = image.crop(box)
            patch_mask = label.crop(box).point(lambda value: 255 if value == 1 else 0)
            if np.count_nonzero(np.asarray(patch_mask)) >= 8:
                return patch, patch_mask
        return None

    def _make_synthetic_pair(self, image, label):
        clean_crop = self._best_clean_crop(image, label)
        ink = self._load_random_ink_patch()
        if clean_crop is None or ink is None:
            return None
        clean, clean_label = clean_crop
        patch, patch_mask = ink
        target_side = random.randint(
            max(24, clean.size[0] // 8), max(32, clean.size[0] // 3)
        )
        scale = target_side / float(max(patch.size))
        target_size = (
            max(1, int(round(patch.size[0] * scale))),
            max(1, int(round(patch.size[1] * scale))),
        )
        patch = patch.resize(target_size, Image.BILINEAR)
        patch_mask = patch_mask.resize(target_size, Image.NEAREST)
        angle = random.uniform(-8.0, 8.0)
        patch = patch.rotate(angle, Image.BILINEAR, expand=True, fillcolor=(255, 255, 255))
        patch_mask = patch_mask.rotate(angle, Image.NEAREST, expand=True, fillcolor=0)
        if patch.size[0] > clean.size[0] or patch.size[1] > clean.size[1]:
            return None
        print_coordinates = np.argwhere(np.asarray(clean_label) == 2)
        if not len(print_coordinates):
            return None
        center_y, center_x = print_coordinates[
            random.randrange(len(print_coordinates))
        ]
        jitter = max(1, patch.size[0] // 4)
        left = int(center_x - patch.size[0] // 2 + random.randint(-jitter, jitter))
        top = int(center_y - patch.size[1] // 2 + random.randint(-jitter, jitter))
        left = max(0, min(clean.size[0] - patch.size[0], left))
        top = max(0, min(clean.size[1] - patch.size[1], top))

        alpha = patch_mask.filter(ImageFilter.GaussianBlur(0.5))
        synthetic = clean.copy()
        synthetic.paste(patch, (left, top), alpha)
        new_label = clean_label.copy()
        label_region = new_label.crop(
            (left, top, left + patch.size[0], top + patch.size[1])
        )
        label_array = np.asarray(label_region, dtype=np.uint8).copy()
        label_array[np.asarray(patch_mask) > 64] = 1
        new_label.paste(Image.fromarray(label_array), (left, top))
        return synthetic, new_label, clean

    def __getitem__(self, index):
        image_path, label_path = self.base.pairs[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(label_path) as source:
            label = source.convert("L")
        clean_path = self.clean_targets.get(image_path.stem)
        synthetic = None
        if (
            clean_path is None
            and self.synthetic_enabled
            and random.random() < self.synthetic_probability
        ):
            for attempt in range(8):
                if attempt == 0:
                    target_image, target_label = image, label
                else:
                    target_image_path, target_label_path = random.choice(
                        self.base.pairs
                    )
                    with Image.open(target_image_path) as source:
                        target_image = source.convert("RGB")
                    with Image.open(target_label_path) as source:
                        target_label = source.convert("L")
                synthetic = self._make_synthetic_pair(
                    target_image, target_label
                )
                if synthetic is not None:
                    break
        if synthetic is not None:
            image, label, clean = synthetic
            restoration_valid = 1.0
        elif clean_path is None:
            clean = image.copy()
            restoration_valid = 0.0
        else:
            with Image.open(clean_path) as source:
                clean = source.convert("RGB")
            restoration_valid = 1.0
        if image.size != label.size or image.size != clean.size:
            raise ValueError(
                "joint sample sizes differ for %s: image=%s label=%s clean=%s"
                % (image_path.name, image.size, label.size, clean.size)
            )
        if self.transform is not None:
            image, label, clean = self.transform(image, label, clean)
        return image, label, clean, torch.tensor(restoration_valid)


class MultiSourceHWRestoration(data.ConcatDataset):
    def __init__(self, datasets):
        if not datasets:
            raise ValueError("at least one dataset source is required")
        super().__init__(datasets)
        self.dataset_names = [dataset.dataset_name for dataset in datasets]

    def balanced_sample_weights(self, dataset_weights=None):
        if dataset_weights is None:
            dataset_weights = [1.0] * len(self.datasets)
        if len(dataset_weights) != len(self.datasets):
            raise ValueError("dataset weight count must match data-root count")
        weights = []
        for dataset, source_weight in zip(self.datasets, dataset_weights):
            if source_weight <= 0:
                raise ValueError("dataset weights must be positive")
            weights.extend([source_weight / len(dataset)] * len(dataset))
        return weights
