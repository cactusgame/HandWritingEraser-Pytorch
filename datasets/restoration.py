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


def estimate_clean_print_mask(clean, label):
    """Estimate amodal print, retaining explicit visible-print annotations.

    Paired data only needs estimation under handwriting; visible print remains
    authoritative from class 2.  Synthetic data supplies an exact mask instead.
    """
    clean_rgb = np.asarray(clean.convert("RGB"), dtype=np.uint8)
    label_array = np.asarray(label, dtype=np.uint8)
    gray = (
        clean_rgb[..., 0].astype(np.float32) * 0.299
        + clean_rgb[..., 1].astype(np.float32) * 0.587
        + clean_rgb[..., 2].astype(np.float32) * 0.114
    ).astype(np.uint8)
    local_background = np.asarray(
        Image.fromarray(gray).filter(ImageFilter.MaxFilter(15)),
        dtype=np.int16,
    )
    contrast = (local_background - gray.astype(np.int16)) >= 14
    dark = gray < 145
    estimated_overlap = (label_array == 1) & (contrast | dark)
    return Image.fromarray(
        ((label_array == 2) | estimated_overlap).astype(np.uint8), mode="L"
    )


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
        color_negative_probability=0.55,
    ):
        self.crop_size = int(crop_size)
        self.scale_range = scale_range
        self.foreground_probability = foreground_probability
        self.train = train
        self.mean = mean
        self.std = std
        self.color_negative_probability = color_negative_probability
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

    @staticmethod
    def _color_negative(image, clean, label, clean_print):
        """Create colored panels and colored print without changing semantics."""
        image_array = np.asarray(image, dtype=np.float32).copy()
        clean_array = np.asarray(clean, dtype=np.float32).copy()
        label_array = np.asarray(label, dtype=np.uint8)
        print_array = np.asarray(clean_print, dtype=np.uint8) > 0
        height, width = label_array.shape

        palette = np.asarray(
            [
                (188, 218, 194), (244, 226, 161), (191, 218, 239),
                (231, 197, 210), (210, 207, 232), (224, 215, 190),
            ],
            dtype=np.float32,
        )
        color = palette[random.randrange(len(palette))]
        left = random.randint(0, max(0, width - 1))
        top = random.randint(0, max(0, height - 1))
        right = random.randint(max(left + 1, width // 3), width)
        bottom = random.randint(max(top + 1, height // 4), height)
        alpha = random.uniform(0.18, 0.48)
        image_array[top:bottom, left:right] = (
            image_array[top:bottom, left:right] * (1.0 - alpha)
            + color * alpha
        )
        clean_array[top:bottom, left:right] = (
            clean_array[top:bottom, left:right] * (1.0 - alpha)
            + color * alpha
        )

        if random.random() < 0.65:
            ink_colors = np.asarray(
                [(22, 54, 145), (145, 35, 45), (25, 105, 76),
                 (111, 47, 135), (36, 100, 125)],
                dtype=np.float32,
            )
            ink_color = ink_colors[random.randrange(len(ink_colors))]
            strength = random.uniform(0.55, 0.95)
            visible_print = label_array == 2
            image_array[visible_print] = (
                image_array[visible_print] * (1.0 - strength)
                + ink_color * strength
            )
            clean_array[print_array] = (
                clean_array[print_array] * (1.0 - strength)
                + ink_color * strength
            )
        return (
            Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8)),
            Image.fromarray(np.clip(clean_array, 0, 255).astype(np.uint8)),
        )

    def _random_crop(self, image, label, clean, clean_print):
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
            clean_print = ImageOps.expand(clean_print, padding, fill=0)
            width, height = image.size

        def candidate():
            top = random.randint(0, height - size)
            left = random.randint(0, width - size)
            return top, left

        top, left = candidate()
        prefer_overlap = random.random() < 0.65
        if random.random() < self.foreground_probability:
            for _ in range(16):
                try_top, try_left = candidate()
                crop = np.asarray(
                    label.crop((try_left, try_top, try_left + size, try_top + size))
                )
                print_crop = np.asarray(
                    clean_print.crop(
                        (try_left, try_top, try_left + size, try_top + size)
                    )
                ) > 0
                top, left = try_top, try_left
                hand = crop == 1
                enough_hand = np.count_nonzero(hand) >= size * size * 0.001
                enough_overlap = np.count_nonzero(hand & print_crop) >= 8
                if enough_hand and (enough_overlap or not prefer_overlap):
                    break
        box = (left, top, left + size, top + size)
        return (
            image.crop(box), label.crop(box), clean.crop(box),
            clean_print.crop(box),
        )

    def __call__(self, image, label, clean, clean_print):
        if self.train:
            if random.random() < self.color_negative_probability:
                image, clean = self._color_negative(
                    image, clean, label, clean_print
                )
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
                clean_print = clean_print.rotate(
                    angle, Image.NEAREST, fillcolor=0
                )

            scale = random.uniform(*self.scale_range)
            width, height = image.size
            target = (max(1, int(round(width * scale))),
                      max(1, int(round(height * scale))))
            image = image.resize(target, Image.BILINEAR)
            clean = clean.resize(target, Image.BILINEAR)
            label = label.resize(target, Image.NEAREST)
            clean_print = clean_print.resize(target, Image.NEAREST)

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
                clean_print = clean_print.resize(target, Image.NEAREST)
            image, label, clean, clean_print = self._random_crop(
                image, label, clean, clean_print
            )
            image, _ = self.degradation(image, label)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, self.mean, self.std)
        clean_tensor = TF.to_tensor(clean)
        label_tensor = torch.from_numpy(
            np.asarray(label, dtype=np.int64).copy()
        )
        print_tensor = torch.from_numpy(
            (np.asarray(clean_print, dtype=np.uint8) > 0).astype(
                np.float32, copy=False
            ).copy()
        )
        return image_tensor, label_tensor, clean_tensor, print_tensor


class HWRestorationDataset(data.Dataset):
    """Baidu-format data with clean RGB and amodal print supervision.

    Sources without ``CleanTargets/`` remain useful for segmentation. Their
    clean target is a placeholder and ``restoration_valid`` is zero, so they
    never contribute to image-reconstruction losses unless a forced-overlap
    pair is synthesized online.
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
        clean_print_dir = self.root / "CleanPrintMasks"
        self.clean_print_targets = {}
        if clean_print_dir.is_dir():
            for path in clean_print_dir.iterdir():
                if path.is_file() and path.suffix.lower() in self.base.valid_suffixes:
                    if path.stem in self.clean_print_targets:
                        raise ValueError(
                            "duplicate clean print mask stem: %s" % path.stem
                        )
                    self.clean_print_targets[path.stem] = path
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
        clean_print_array = np.asarray(clean_label) == 2
        print_coordinates = np.argwhere(clean_print_array)
        if not len(print_coordinates):
            return None
        patch_mask_array = np.asarray(patch_mask) > 64
        hand_pixels = max(1, int(patch_mask_array.sum()))
        best = None
        for _ in range(24):
            center_y, center_x = print_coordinates[
                random.randrange(len(print_coordinates))
            ]
            jitter = max(1, patch.size[0] // 6)
            left = int(
                center_x - patch.size[0] // 2
                + random.randint(-jitter, jitter)
            )
            top = int(
                center_y - patch.size[1] // 2
                + random.randint(-jitter, jitter)
            )
            left = max(0, min(clean.size[0] - patch.size[0], left))
            top = max(0, min(clean.size[1] - patch.size[1], top))
            print_region = clean_print_array[
                top:top + patch.size[1], left:left + patch.size[0]
            ]
            overlap = int(np.count_nonzero(print_region & patch_mask_array))
            if best is None or overlap > best[0]:
                best = (overlap, left, top)
            if overlap >= max(8, int(round(hand_pixels * 0.12))):
                break
        if best is None or best[0] < 8:
            return None
        _, left, top = best

        alpha = patch_mask.filter(ImageFilter.GaussianBlur(0.5))
        synthetic = clean.copy()
        synthetic.paste(patch, (left, top), alpha)
        new_label = clean_label.copy()
        label_region = new_label.crop(
            (left, top, left + patch.size[0], top + patch.size[1])
        )
        label_array = np.asarray(label_region, dtype=np.uint8).copy()
        label_array[patch_mask_array] = 1
        new_label.paste(Image.fromarray(label_array), (left, top))
        clean_print = Image.fromarray(
            clean_print_array.astype(np.uint8), mode="L"
        )
        return synthetic, new_label, clean, clean_print

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
            image, label, clean, clean_print = synthetic
            restoration_valid = 1.0
        elif clean_path is None:
            clean = image.copy()
            clean_print = Image.fromarray(
                (np.asarray(label, dtype=np.uint8) == 2).astype(np.uint8),
                mode="L",
            )
            restoration_valid = 0.0
        else:
            with Image.open(clean_path) as source:
                clean = source.convert("RGB")
            clean_print_path = self.clean_print_targets.get(image_path.stem)
            if clean_print_path is None:
                clean_print = estimate_clean_print_mask(clean, label)
            else:
                with Image.open(clean_print_path) as source:
                    clean_print = source.convert("L").point(
                        lambda value: 1 if value > 0 else 0
                    )
            restoration_valid = 1.0
        if (
            image.size != label.size
            or image.size != clean.size
            or image.size != clean_print.size
        ):
            raise ValueError(
                "joint sample sizes differ for %s: image=%s label=%s "
                "clean=%s clean_print=%s"
                % (
                    image_path.name, image.size, label.size, clean.size,
                    clean_print.size,
                )
            )
        if self.transform is not None:
            image, label, clean, clean_print = self.transform(
                image, label, clean, clean_print
            )
        return (
            image, label, clean, clean_print,
            torch.tensor(restoration_valid),
        )


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
