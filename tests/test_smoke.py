import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import network
from datasets import HWSegmentation, MultiSourceHWSegmentation
from predict import tile_starts
from tools.convert_scut_ensexam import make_three_class_label
from tools.convert_signatr6k import convert_mask
from utils.ext_transforms import ExtEnsureMinSize
from utils.loss import HybridSegmentationLoss


class UpgradeSmokeTests(unittest.TestCase):
    def test_lite_model_cpu_forward_and_loss(self):
        model = network.modeling.lite_eraser(
            num_classes=3, output_stride=16, pretrained_backbone=False
        )
        inputs = torch.randn(2, 3, 65, 91)
        targets = torch.randint(0, 3, (2, 65, 91))
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2, 3, 65, 91))
        loss = HybridSegmentationLoss([1, 4, 2])(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(sum(p.numel() for p in model.parameters()), 2_000_000)

    def test_tiles_cover_the_last_pixel(self):
        starts = tile_starts(1500, 768, 128)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1] + 768, 1500)

    def test_dataset_pairs_by_stem_and_honors_val_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Images").mkdir()
            (root / "Labels").mkdir()
            for stem in ("a", "b", "c"):
                Image.new("RGB", (16, 12), "white").save(root / "Images" / (stem + ".jpg"))
                Image.fromarray(np.zeros((12, 16), dtype=np.uint8)).save(
                    root / "Labels" / (stem + ".png")
                )
            val_list = root / "validation.txt"
            val_list.write_text("b.jpg\n", encoding="utf-8")
            train = HWSegmentation(root, train=True, val_list=val_list)
            val = HWSegmentation(root, train=False, val_list=val_list)
            self.assertEqual(len(train), 2)
            self.assertEqual(len(val), 1)
            self.assertEqual(Path(val.images[0]).stem, "b")

    def test_baidu_variants_stay_in_one_generated_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Images").mkdir()
            (root / "Labels").mkdir()
            stems = [
                "dehw_train_00000_00", "dehw_train_00000_01",
                "dehw_train_00001_00", "dehw_train_00001_10",
                "dehw_train_00002_00", "dehw_train_00002_11",
            ]
            for stem in stems:
                Image.new("RGB", (8, 8), "white").save(root / "Images" / (stem + ".png"))
                Image.new("L", (8, 8), 0).save(root / "Labels" / (stem + ".png"))
            train = HWSegmentation(root, split="train", split_seed=3, val_ratio=0.34)
            val = HWSegmentation(root, split="validation", split_seed=3, val_ratio=0.34)
            train_groups = {stem.rsplit("_", 1)[0] for stem in train.sample_ids}
            val_groups = {stem.rsplit("_", 1)[0] for stem in val.sample_ids}
            self.assertTrue(train_groups.isdisjoint(val_groups))

    def test_multi_source_balancing(self):
        first = list(range(2))
        second = list(range(6))
        combined = MultiSourceHWSegmentation([first, second])
        weights = combined.balanced_sample_weights()
        self.assertAlmostEqual(sum(weights[:2]), sum(weights[2:]))
        self.assertEqual(len(combined), 8)

    def test_signatr_color_mapping_prioritizes_overlap_as_handwriting(self):
        colors = np.array(
            [[[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]],
            dtype=np.uint8,
        )
        converted = convert_mask(Image.fromarray(colors), "handwriting")
        np.testing.assert_array_equal(converted, [[0, 1, 2, 1]])

    def test_scut_paired_difference_conversion(self):
        erased = np.full((32, 32, 3), 255, dtype=np.uint8)
        erased[15:17, 3:29] = 0
        original = erased.copy()
        original[5:12, 8:24] = [40, 40, 180]
        box_mask = np.zeros((32, 32), dtype=bool)
        box_mask[3:14, 6:26] = True
        label = make_three_class_label(
            Image.fromarray(original),
            Image.fromarray(erased),
            box_mask,
            background_radius=3,
        )
        self.assertEqual(label[8, 12], 1)
        self.assertEqual(label[16, 12], 2)
        self.assertEqual(label[0, 0], 0)

    def test_small_source_is_upscaled_without_padding(self):
        image = Image.new("RGB", (32, 16), "white")
        label = Image.new("L", (32, 16), 0)
        image, label = ExtEnsureMinSize(64)(image, label)
        self.assertEqual(image.size, (128, 64))
        self.assertEqual(label.size, image.size)


if __name__ == "__main__":
    unittest.main()
