import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import network
from main import write_validation_scalars
from datasets import (
    HWSegmentation,
    HWRestorationDataset,
    JointDocumentTransform,
    MultiSourceHWSegmentation,
)
from datasets.restoration import estimate_clean_print_mask
from predict import tile_starts
from tools.convert_scut_ensexam import make_three_class_label
from tools.convert_signatr6k import convert_mask
from tools.synthesize_clean_documents import (
    collect_handwriting_pairs,
    document_seed,
    estimate_document_print_mask,
    find_placement,
    random_scribble_layer,
    resolve_handwriting_roots,
    split_documents,
    synthesize_variant,
)
from utils.ext_transforms import ExtEnsureMinSize
from utils import ModelEMA
from utils.document_postprocess import refine_document_restoration
from utils.loss import (
    HybridSegmentationLoss,
    JointRestorationLoss,
    LayeredRestorationLoss,
    StructureAwareLoss,
)


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

    def test_server_model_cpu_forward_with_odd_shape(self):
        model = network.modeling.server_eraser(
            num_classes=3, output_stride=32, pretrained_backbone=False
        ).eval()
        inputs = torch.randn(1, 3, 65, 91)
        with torch.inference_mode():
            logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (1, 3, 65, 91))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertLess(parameter_count, 12_000_000)

    def test_structure_loss_and_ema(self):
        logits = torch.randn(2, 3, 33, 41, requires_grad=True)
        targets = torch.randint(0, 3, (2, 33, 41))
        criterion = StructureAwareLoss([1, 3, 2])
        loss = criterion(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(criterion.last_components),
            {"cross_entropy", "tversky", "boundary"},
        )

        model = torch.nn.Conv2d(3, 3, 1)
        ema = ModelEMA(model, decay=0.9)
        before = ema.module.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(1.0)
        ema.update(model)
        self.assertFalse(torch.equal(before, ema.module.weight))
        self.assertFalse(any(p.requires_grad for p in ema.module.parameters()))

    def test_joint_model_restores_rgb_and_backpropagates(self):
        model = network.modeling.joint_eraser(
            num_classes=3, output_stride=32, pretrained_backbone=False
        )
        inputs = torch.randn(2, 3, 65, 83)
        labels = torch.randint(0, 3, (2, 65, 83))
        clean = torch.rand(2, 3, 65, 83)
        valid = torch.tensor([1.0, 0.0])
        outputs = model(inputs)
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(tuple(item.shape[-2:]) == (65, 83) for item in outputs))
        self.assertTrue(torch.all((outputs[-1] >= 0) & (outputs[-1] <= 1)))
        criterion = JointRestorationLoss([1, 3, 2])
        loss = criterion(outputs, labels, clean, valid, inputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("reconstruction", criterion.last_components)

    def test_layered_model_has_amodal_print_and_separate_losses(self):
        model = network.modeling.layered_eraser(
            num_classes=3, output_stride=32, pretrained_backbone=False
        )
        inputs = torch.randn(1, 3, 65, 83)
        labels = torch.zeros(1, 65, 83, dtype=torch.long)
        labels[:, 20:45, 28:55] = 1
        clean_print = torch.zeros(1, 65, 83)
        clean_print[:, 31:34, 10:72] = 1
        clean = torch.rand(1, 3, 65, 83)
        outputs = model(inputs)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(tuple(outputs[0].shape), (1, 3, 65, 83))
        self.assertEqual(tuple(outputs[1].shape), (1, 1, 65, 83))
        self.assertTrue(torch.all((outputs[-1] >= 0) & (outputs[-1] <= 1)))

        criterion = LayeredRestorationLoss([1, 3, 2])
        loss = criterion(
            outputs, labels, clean, clean_print, torch.ones(1), inputs
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("background", criterion.last_components)
        self.assertIn("overlap_reconstruction", criterion.last_components)
        self.assertLess(
            sum(parameter.numel() for parameter in model.parameters()),
            30_000_000,
        )

    def test_restoration_dataset_reads_paired_clean_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("Images", "Labels", "CleanTargets"):
                (root / name).mkdir()
            for index in range(2):
                stem = str(index)
                Image.new("RGB", (40, 32), (230, 225, 220)).save(
                    root / "Images" / (stem + ".png")
                )
                label = np.zeros((32, 40), dtype=np.uint8)
                label[8:12, 10:25] = 1
                Image.fromarray(label).save(root / "Labels" / (stem + ".png"))
                Image.new("RGB", (40, 32), (245, 240, 235)).save(
                    root / "CleanTargets" / (stem + ".png")
                )
            dataset = HWRestorationDataset(
                root,
                transform=JointDocumentTransform(32, train=False),
                split="train",
                val_ratio=0.5,
            )
            image, label, clean, clean_print, valid = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 32, 40))
            self.assertEqual(tuple(clean.shape), (3, 32, 40))
            self.assertEqual(tuple(label.shape), (32, 40))
            self.assertEqual(tuple(clean_print.shape), (32, 40))
            self.assertEqual(float(valid), 1.0)

    def test_clean_print_mask_keeps_print_hidden_by_handwriting(self):
        clean = np.full((40, 64, 3), 245, dtype=np.uint8)
        clean[19:22, 8:56] = [35, 55, 120]
        labels = np.zeros((40, 64), dtype=np.uint8)
        labels[19:22, 8:24] = 2
        labels[14:28, 24:40] = 1
        mask = np.asarray(estimate_clean_print_mask(
            Image.fromarray(clean), Image.fromarray(labels)
        ))
        self.assertEqual(mask[20, 12], 1)
        self.assertEqual(mask[20, 32], 1)
        self.assertEqual(mask[5, 5], 0)

    def test_clean_document_synthesis_forces_print_overlap(self):
        clean_array = np.full((128, 192, 3), [196, 220, 202], dtype=np.uint8)
        clean_array[54:62, 12:180] = [22, 35, 29]
        clean = Image.fromarray(clean_array)
        print_mask = estimate_document_print_mask(clean)
        self.assertGreater(np.count_nonzero(print_mask[54:62]), 0)

        result = synthesize_variant(
            clean=clean,
            print_mask=print_mask,
            pairs=[],
            mode="overlap",
            random_scribble_probability=1.0,
            minimum_overlap=0.08,
            high_resolution_command=None,
            rng=random.Random(31),
        )
        composite, target, label, complete_print, overlap, source = result
        self.assertEqual(composite.size, clean.size)
        self.assertEqual(target.size, clean.size)
        self.assertEqual(source, "procedural")
        self.assertGreaterEqual(overlap, 0.08)
        self.assertGreater(np.count_nonzero(label == 1), 0)
        self.assertGreater(
            np.count_nonzero((label == 1) & complete_print), 0
        )
        np.testing.assert_array_equal(np.asarray(target), clean_array)

    def test_synthesis_placement_and_document_splits_do_not_leak(self):
        alpha = np.zeros((20, 30), dtype=np.float32)
        alpha[8:12, 2:28] = 1.0
        print_mask = np.zeros((60, 90), dtype=bool)
        print_mask[28:32, 5:85] = True
        placement = find_placement(
            alpha, print_mask, "overlap", 0.5, random.Random(7)
        )
        self.assertIsNotNone(placement)
        self.assertGreaterEqual(placement[2], 0.5)

        documents = [Path("document_%02d.png" % index) for index in range(10)]
        splits = split_documents(documents, 0.2, 0.2, seed=5)
        self.assertEqual(sum(map(len, splits.values())), len(documents))
        self.assertTrue(set(splits["train"]).isdisjoint(splits["validation"]))
        self.assertTrue(set(splits["train"]).isdisjoint(splits["test"]))
        self.assertTrue(
            set(splits["validation"]).isdisjoint(splits["test"])
        )

    def test_handwriting_parent_root_discovery_and_worker_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            dataset = parent / "converted"
            (dataset / "Images").mkdir(parents=True)
            (dataset / "Labels").mkdir()
            Image.new("RGB", (24, 16), "white").save(
                dataset / "Images" / "sample.png"
            )
            label = np.zeros((16, 24), dtype=np.uint8)
            label[5:9, 6:18] = 1
            Image.fromarray(label).save(
                dataset / "Labels" / "sample.png"
            )
            roots = resolve_handwriting_roots([parent])
            self.assertEqual(roots, [dataset.resolve()])
            self.assertEqual(len(collect_handwriting_pairs(roots)), 1)

        self.assertEqual(
            document_seed(17, "train", 4),
            document_seed(17, "train", 4),
        )
        self.assertNotEqual(
            document_seed(17, "train", 4),
            document_seed(17, "validation", 4),
        )

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

    def test_validation_metrics_are_written_to_tensorboard_writer(self):
        class DummyWriter:
            def __init__(self):
                self.scalars = {}
                self.flushed = False

            def add_scalar(self, tag, value, step):
                self.scalars[tag] = (value, step)

            def flush(self):
                self.flushed = True

        writer = DummyWriter()
        overall = {
            "Overall Acc": 0.9,
            "Handwriting IoU": 0.7,
            "Class IoU": {0: 0.95, 1: 0.7, 2: np.nan},
        }
        source_scores = {
            "SCUT/EnsExam": {
                "Overall Acc": 0.8,
                "Handwriting IoU": 0.6,
                "Class IoU": {0: 0.9, 1: 0.6, 2: 0.5},
            }
        }

        write_validation_scalars(writer, 12, overall, source_scores, 0.6)

        self.assertEqual(writer.scalars["validation/all/Overall_Acc"], (0.9, 12))
        self.assertEqual(writer.scalars["validation/all/Class_IoU/class_1"], (0.7, 12))
        self.assertNotIn("validation/all/Class_IoU/class_2", writer.scalars)
        self.assertEqual(
            writer.scalars["validation/SCUT_EnsExam/Handwriting_IoU"], (0.6, 12)
        )
        self.assertEqual(writer.scalars["validation/macro/Handwriting_IoU"], (0.6, 12))
        self.assertTrue(writer.flushed)

    def test_postprocess_repairs_white_fill_on_colored_paper(self):
        original = np.full((48, 96, 3), [198, 220, 204], dtype=np.uint8)
        labels = np.zeros((48, 96), dtype=np.uint8)
        labels[14:36, 38:62] = 1
        restored = original.copy()
        restored[labels == 1] = 255

        result, debug = refine_document_restoration(
            Image.fromarray(original),
            Image.fromarray(restored),
            labels,
            mask_dilate=0,
            mode="background",
            return_debug=True,
        )
        result = np.asarray(result)

        self.assertGreater(np.count_nonzero(debug.background_repair), 0)
        self.assertLess(
            np.abs(result[24, 48].astype(int) - original[24, 48]).max(), 5
        )
        np.testing.assert_array_equal(result[0, 0], restored[0, 0])

    def test_postprocess_bridges_short_print_gap_without_restoring_color_ink(self):
        original = np.full((60, 160, 3), 255, dtype=np.uint8)
        original[29:32, 15:145] = 20
        # A colored handwritten stroke covers a short section of the print line.
        original[10:50, 77:83] = [35, 80, 210]
        labels = np.zeros((60, 160), dtype=np.uint8)
        labels[29:32, 15:145] = 2
        labels[10:50, 77:83] = 1
        restored = original.copy()
        restored[labels == 1] = 255

        result, debug = refine_document_restoration(
            original,
            restored,
            labels,
            mask_dilate=0,
            max_print_gap=7,
            mode="balanced",
            return_debug=True,
        )
        result = np.asarray(result)

        self.assertTrue(np.all(result[30, 77:83].mean(axis=1) < 80))
        self.assertTrue(np.all(result[15, 77:83] > 240))
        self.assertGreater(np.count_nonzero(debug.bridged_print), 0)


if __name__ == "__main__":
    unittest.main()
