"""Render every PDF under a directory to high-resolution document images.

The default output is lossless color PNG at 300 DPI. Directory structure is
preserved and each PDF receives its own directory:

    input/contracts/sample.pdf
    output/contracts/sample/page_0001.png

Poppler's ``pdfinfo`` and ``pdftoppm`` executables are required.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm


PAGE_PATTERN = re.compile(r"-(\d+)$")
MANAGED_SUFFIXES = {".png", ".jpg", ".jpeg"}


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", required=True,
        help="directory containing PDF files",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="directory that will receive rendered page images",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="rendering resolution; 300 is recommended for training data",
    )
    parser.add_argument(
        "--format", choices=("png", "jpeg"), default="png",
        help="PNG is lossless; JPEG saves disk space",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95,
        help="JPEG quality when --format jpeg is selected",
    )
    parser.add_argument(
        "--jobs", type=int, default=min(4, os.cpu_count() or 1),
        help="number of PDFs rendered in parallel",
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="only process PDFs directly inside --input-dir",
    )
    parser.add_argument(
        "--use-cropbox", action="store_true",
        help="render each PDF page's CropBox instead of its MediaBox",
    )
    parser.add_argument(
        "--grayscale", action="store_true",
        help="export grayscale images; color is retained by default",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="re-render complete outputs as well as incomplete/stale outputs",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="stop scheduling new work after the first failed PDF",
    )
    return parser


def require_executable(name):
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(
            "%s was not found. Install Poppler first: "
            "'brew install poppler' on macOS or "
            "'sudo apt-get install poppler-utils' on Ubuntu." % name
        )
    return executable


def collect_pdfs(input_dir, recursive=True):
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    pdfs = sorted(
        (
            path for path in iterator
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: str(path.relative_to(input_dir)).lower(),
    )
    if not pdfs:
        raise ValueError("no PDF files found under %s" % input_dir)
    return input_dir, pdfs


def output_directory(input_dir, output_dir, pdf_path):
    relative = pdf_path.relative_to(input_dir)
    return Path(output_dir).expanduser().resolve() / relative.with_suffix("")


def validate_unique_outputs(input_dir, output_dir, pdfs):
    destinations = {}
    for pdf_path in pdfs:
        destination = output_directory(
            input_dir, output_dir, pdf_path
        )
        key = str(destination).casefold()
        previous = destinations.get(key)
        if previous is not None:
            raise ValueError(
                "PDF output collision: %s and %s both map to %s"
                % (previous, pdf_path, destination)
            )
        destinations[key] = pdf_path


def page_count(pdfinfo, pdf_path):
    process = subprocess.run(
        [pdfinfo, str(pdf_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(os.environ, LC_ALL="C"),
    )
    if process.returncode != 0:
        raise RuntimeError(
            "pdfinfo failed for %s:\n%s"
            % (pdf_path, process.stderr.strip())
        )
    match = re.search(r"^Pages:\s+(\d+)\s*$", process.stdout, re.MULTILINE)
    if match is None:
        raise RuntimeError("could not read page count from %s" % pdf_path)
    count = int(match.group(1))
    if count <= 0:
        raise RuntimeError("%s contains no pages" % pdf_path)
    return count


def managed_pages(destination, suffix):
    return sorted(
        path for path in destination.glob("page_*" + suffix)
        if path.is_file()
    )


def inspect_image(path):
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    if width <= 0 or height <= 0:
        raise RuntimeError("invalid image dimensions: %s" % path)
    return {"file": path.name, "width": width, "height": height, "mode": mode}


def complete_output(destination, suffix, expected_pages, source_mtime):
    pages = managed_pages(destination, suffix)
    if len(pages) != expected_pages:
        return None
    if any(path.stat().st_mtime < source_mtime for path in pages):
        return None
    try:
        details = [inspect_image(path) for path in pages]
    except Exception:
        return None
    return details


def rendered_page_number(path):
    match = PAGE_PATTERN.search(path.stem)
    if match is None:
        raise RuntimeError("unexpected Poppler output filename: %s" % path)
    return int(match.group(1))


def render_command(opts, pdftoppm, pdf_path, prefix):
    command = [pdftoppm, "-r", str(opts.dpi)]
    if opts.use_cropbox:
        command.append("-cropbox")
    if opts.grayscale:
        command.append("-gray")
    if opts.format == "png":
        command.append("-png")
    else:
        command.extend([
            "-jpeg",
            "-jpegopt",
            "quality=%d,optimize=y,progressive=n" % opts.jpeg_quality,
        ])
    command.extend([str(pdf_path), str(prefix)])
    return command


def remove_managed_pages(destination):
    if not destination.is_dir():
        return
    for path in destination.iterdir():
        if (
            path.is_file()
            and path.name.startswith("page_")
            and path.suffix.lower() in MANAGED_SUFFIXES
        ):
            path.unlink()


def render_one(
    opts,
    input_dir,
    output_dir,
    pdf_path,
    pdfinfo,
    pdftoppm,
):
    expected_pages = page_count(pdfinfo, pdf_path)
    destination = output_directory(input_dir, output_dir, pdf_path)
    suffix = ".png" if opts.format == "png" else ".jpg"
    source_mtime = pdf_path.stat().st_mtime
    if not opts.overwrite:
        existing = complete_output(
            destination, suffix, expected_pages, source_mtime
        )
        if existing is not None:
            return {
                "source": str(pdf_path),
                "source_relative": str(pdf_path.relative_to(input_dir)),
                "output_directory": str(destination),
                "page_count": expected_pages,
                "pages": existing,
                "status": "skipped_complete",
            }

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pdf-render-", dir=destination.parent
    ) as temporary:
        temporary = Path(temporary)
        prefix = temporary / "render"
        command = render_command(opts, pdftoppm, pdf_path, prefix)
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.returncode != 0:
            raise RuntimeError(
                "pdftoppm failed for %s:\n%s"
                % (pdf_path, process.stderr.strip())
            )
        rendered_suffix = ".png" if opts.format == "png" else ".jpg"
        rendered = sorted(
            temporary.glob("render-*" + rendered_suffix),
            key=rendered_page_number,
        )
        if len(rendered) != expected_pages:
            raise RuntimeError(
                "%s: expected %d pages, Poppler rendered %d"
                % (pdf_path, expected_pages, len(rendered))
            )
        temporary_details = [inspect_image(path) for path in rendered]

        remove_managed_pages(destination)
        digits = max(4, len(str(expected_pages)))
        final_pages = []
        for index, (source, detail) in enumerate(
            zip(rendered, temporary_details), start=1
        ):
            filename = ("page_%0*d%s" % (digits, index, suffix))
            final_path = destination / filename
            os.replace(source, final_path)
            detail = dict(detail)
            detail["file"] = filename
            final_pages.append(detail)

    verified = complete_output(
        destination, suffix, expected_pages, source_mtime=0
    )
    if verified is None:
        raise RuntimeError("final output verification failed for %s" % pdf_path)
    return {
        "source": str(pdf_path),
        "source_relative": str(pdf_path.relative_to(input_dir)),
        "output_directory": str(destination),
        "page_count": expected_pages,
        "pages": final_pages,
        "status": "rendered",
    }


def write_manifest(output_dir, opts, records, failures):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "input_directory": str(Path(opts.input_dir).expanduser().resolve()),
        "output_directory": str(output_dir),
        "rendering": {
            "dpi": opts.dpi,
            "format": opts.format,
            "jpeg_quality": (
                opts.jpeg_quality if opts.format == "jpeg" else None
            ),
            "grayscale": opts.grayscale,
            "use_cropbox": opts.use_cropbox,
        },
        "pdf_count": len(records) + len(failures),
        "successful_pdf_count": len(records),
        "page_count": sum(record["page_count"] for record in records),
        "rendered_count": sum(
            record["status"] == "rendered" for record in records
        ),
        "skipped_count": sum(
            record["status"] == "skipped_complete" for record in records
        ),
        "failure_count": len(failures),
        "documents": sorted(records, key=lambda item: item["source_relative"]),
        "failures": failures,
    }
    temporary = output_dir / ".export_manifest.json.tmp"
    final = output_dir / "export_manifest.json"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, final)
    return manifest


def validate_options(opts):
    if not 72 <= opts.dpi <= 1200:
        raise ValueError("--dpi must be between 72 and 1200")
    if not 1 <= opts.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if opts.jobs <= 0:
        raise ValueError("--jobs must be positive")


def main():
    opts = get_argparser().parse_args()
    validate_options(opts)
    pdfinfo = require_executable("pdfinfo")
    pdftoppm = require_executable("pdftoppm")
    input_dir, pdfs = collect_pdfs(
        opts.input_dir, recursive=not opts.no_recursive
    )
    output_dir = Path(opts.output_dir).expanduser().resolve()
    validate_unique_outputs(input_dir, output_dir, pdfs)

    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=min(opts.jobs, len(pdfs))) as executor:
        futures = {
            executor.submit(
                render_one,
                opts,
                input_dir,
                output_dir,
                pdf_path,
                pdfinfo,
                pdftoppm,
            ): pdf_path
            for pdf_path in pdfs
        }
        progress = tqdm(total=len(futures), desc="PDF export")
        for future in as_completed(futures):
            pdf_path = futures[future]
            try:
                records.append(future.result())
            except Exception as error:
                failures.append(
                    {
                        "source": str(pdf_path),
                        "error": str(error),
                    }
                )
                if opts.fail_fast:
                    for pending in futures:
                        pending.cancel()
                    progress.close()
                    write_manifest(output_dir, opts, records, failures)
                    raise
            finally:
                progress.update(1)
        progress.close()

    manifest = write_manifest(output_dir, opts, records, failures)
    print(json.dumps(
        {
            "pdfs": manifest["pdf_count"],
            "pages": manifest["page_count"],
            "rendered": manifest["rendered_count"],
            "skipped": manifest["skipped_count"],
            "failed": manifest["failure_count"],
            "output": str(output_dir),
        },
        ensure_ascii=False,
        indent=2,
    ))
    if failures:
        for failure in failures:
            print(
                "FAILED: %s\n%s"
                % (failure["source"], failure["error"]),
                file=sys.stderr,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
