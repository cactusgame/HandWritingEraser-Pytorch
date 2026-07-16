import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import network
from datasets import HWSegmentation
from predict import tile_starts
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


if __name__ == "__main__":
    unittest.main()
