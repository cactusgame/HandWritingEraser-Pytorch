import json
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(path):
    stem = path.stem if isinstance(path, Path) else str(path)
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def index_by_stem(directory):
    directory = Path(directory)
    result = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in result:
            raise ValueError("duplicate stem in %s: %s" % (directory, path.stem))
        result[path.stem] = path
    if not result:
        raise ValueError("no images found in %s" % directory)
    return result


def prepare_output(output_root, overwrite=False, resume=False):
    output_root = Path(output_root)
    managed = [
        output_root / "Images",
        output_root / "Labels",
        output_root / "CleanTargets",
        output_root / "CleanPrintMasks",
        output_root / "splits",
    ]
    if output_root.exists() and any(path.exists() for path in managed):
        if resume:
            for path in managed:
                path.mkdir(parents=True, exist_ok=True)
            return output_root
        if not overwrite:
            raise FileExistsError(
                "%s already contains converted data; pass --overwrite to rebuild"
                % output_root
            )
        for path in managed:
            if path.exists():
                shutil.rmtree(path)
    for path in managed:
        path.mkdir(parents=True, exist_ok=True)
    return output_root


def copy_image(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_splits(output_root, split_stems):
    split_dir = Path(output_root) / "splits"
    for split, stems in split_stems.items():
        content = "".join("%s\n" % stem for stem in sorted(stems))
        (split_dir / (split + ".txt")).write_text(content, encoding="utf-8")


def write_metadata(output_root, metadata):
    metadata = dict(metadata)
    metadata["format_version"] = 1
    metadata["classes"] = {"0": "background", "1": "handwriting", "2": "print"}
    (Path(output_root) / "dataset.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
