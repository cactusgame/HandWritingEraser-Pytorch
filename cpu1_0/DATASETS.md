# CPU 1.0 dataset compatibility

CPU 1.0 accepts one or more dataset roots. Every root must have:

```text
dataset-root/
├── Images/
├── Labels/
└── splits/             # optional but recommended
    ├── train.txt
    ├── validation.txt
    └── test.txt
```

Image and label file extensions may differ, but their stems and dimensions must
match. Labels are single-channel class-ID images:

```text
0 = background
1 = handwriting
2 = print
```

`CleanTargets/` and `CleanPrintMasks/` are accepted as extra directories but
are ignored by CPU 1.0, which trains only three-class segmentation.

## Datasets available on this machine

| Dataset root | Directly loadable | Train / validation / test |
| --- | --- | --- |
| `baidu` | yes | 4 / 1 / no explicit test |
| `SCUT-EnsExam-baidu-format` | yes | 4 / 1 / 5 |
| `SignaTR6K-baidu-format` | yes | 5 / 5 / 5 |
| `exersice_images-baidu-format` | yes | 90 / 12 / 12 |
| `SCUT-EnsExam` | no; raw source | convert first |
| `SignaTR6K` | no; raw source | convert first |

The local legacy datasets are small samples. Use the complete copies for formal
training.

## Converting complete datasets on another server

Run from the repository root. SCUT-EnsExam must retain
`train|test/all_images`, `all_labels`, and `box_label_txt`:

```bash
python tools/convert_scut_ensexam.py \
  --source /data/HandWritingData/SCUT-EnsExam \
  --output /data/HandWritingData/SCUT-EnsExam-baidu-format \
  --val-ratio 0.15 \
  --resume
```

SignaTR6K must retain `train|validation|test/crop` and `label`:

```bash
python tools/convert_signatr6k.py \
  --source /data/HandWritingData/SignaTR6K \
  --output /data/HandWritingData/SignaTR6K-baidu-format \
  --overlap-policy handwriting \
  --resume
```

Use `--overwrite` instead of `--resume` to rebuild an output from scratch.

The complete Baidu dataset needs no conversion if it already contains matching
`Images/` and `Labels/` with class IDs 0/1/2. If it only contains legacy
`mask_label/`, create one `Labels/<image-stem>.png` for every image. Augmented
siblings such as `dehw_train_00000_00` and `dehw_train_00000_01` use the same
base mask only when that is how the legacy augmentation was produced. Keep all
siblings from one base page in the same split.

## Training all four sources

```bash
python cpu1_0/main.py \
  --data-root /data/HandWritingData/baidu \
  --data-root /data/HandWritingData/SCUT-EnsExam-baidu-format \
  --data-root /data/HandWritingData/SignaTR6K-baidu-format \
  --data-root /data/HandWritingData/exersice_images-baidu-format \
  --model lite_eraser \
  --dataset-sampling balanced \
  --crop-size 512 \
  --batch-size 8 \
  --total-itrs 30000 \
  --device cuda \
  --checkpoint-dir checkpoints/cpu1_0
```

`--data-root` must point to each dataset root separately; CPU 1.0 does not
discover datasets by passing their common parent directory.
