# cpu1_0 snapshot

This directory is a self-contained copy of the CPU 1.0 training, prediction,
model, dataset, metric, and export code.

- Source branch: `feat/cpu1.0`
- Source commit: `c23e36a`
- Copied into: `feat/gpu1.0`

Run commands from the repository root so checkpoints and output paths remain
explicit:

```bash
python cpu1_0/main.py --help
python cpu1_0/predict.py --help
```

The code accepts one or more Baidu-format dataset roots. Each root must contain
matching `Images/` and `Labels/` files by stem, and labels must use class IDs
`0=background`, `1=handwriting`, `2=print`.
