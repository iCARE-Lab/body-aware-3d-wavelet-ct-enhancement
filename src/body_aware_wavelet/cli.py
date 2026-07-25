"""Command-line interface for single-case and batch enhancement."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

from .config import EnhancementConfig
from .dataset import list_nifti_files, load_manifest_split
from .runner import process_paths, write_run_reports


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", required=True, help="Enhanced NIfTI directory.")
    common.add_argument(
        "--report-dir",
        default=None,
        help="Report directory. Default: <output-dir>/reports.",
    )
    common.add_argument("--hu-min", type=float, default=-1000.0)
    common.add_argument("--hu-max", type=float, default=1000.0)
    common.add_argument("--levels", type=int, default=2048)
    common.add_argument("--wavelet", default="haar")
    common.add_argument("--body-threshold-hu", type=float, default=-600.0)
    common.add_argument("--entropy-bins", type=int, default=256)
    common.add_argument("--eps", type=float, default=1.0e-8)
    common.add_argument("--save-hu-min", type=float, default=-1024.0)
    common.add_argument("--save-hu-max", type=float, default=3071.0)
    common.add_argument(
        "--preserve-outside-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    common.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    common.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Optional expected number of input cases.",
    )
    common.add_argument(
        "--preview-count",
        type=int,
        default=0,
        help="Save matched orthogonal previews for the first N successful cases.",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        prog="body-aware-wavelet",
        description=(
            "Body-aware, exposure-adaptive 3D Haar-wavelet enhancement for CT."
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    common = _common_parser()

    single = subparsers.add_parser(
        "single",
        parents=[common],
        help="Enhance one .nii or .nii.gz volume.",
    )
    single.add_argument("--input-file", required=True)

    directory = subparsers.add_parser(
        "directory",
        parents=[common],
        help="Enhance every .nii.gz file directly inside one directory.",
    )
    directory.add_argument("--input-dir", required=True)

    manifest = subparsers.add_parser(
        "json",
        parents=[common],
        help="Enhance image paths listed in a dataset JSON.",
    )
    manifest.add_argument("--dataset-json", required=True)
    manifest.add_argument(
        "--splits",
        default="training,test",
        help="Comma-separated JSON split names. Default: training,test.",
    )
    return parser


def _configuration_from_args(args: argparse.Namespace) -> EnhancementConfig:
    return EnhancementConfig(
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        levels=args.levels,
        wavelet=args.wavelet,
        body_threshold_hu=args.body_threshold_hu,
        preserve_outside_mask=args.preserve_outside_mask,
        entropy_bins=args.entropy_bins,
        eps=args.eps,
        save_hu_min=args.save_hu_min,
        save_hu_max=args.save_hu_max,
    )


def run(args: argparse.Namespace) -> int:
    """Execute a parsed command and return a process exit code."""

    configuration = _configuration_from_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_dir = (
        Path(args.report_dir).expanduser().resolve()
        if args.report_dir
        else output_dir / "reports"
    )
    preview_dir = report_dir / "previews"
    results: list[dict[str, Any]] = []

    if args.mode == "single":
        results.extend(
            process_paths(
                [args.input_file],
                output_dir,
                configuration,
                overwrite=args.overwrite,
                split_name="single",
                expected_count=1,
                preview_count=args.preview_count,
                preview_dir=preview_dir,
            )
        )
    elif args.mode == "directory":
        paths = list_nifti_files(args.input_dir)
        results.extend(
            process_paths(
                paths,
                output_dir,
                configuration,
                overwrite=args.overwrite,
                split_name="directory",
                expected_count=args.expected_count,
                preview_count=args.preview_count,
                preview_dir=preview_dir,
            )
        )
    elif args.mode == "json":
        split_names = [
            split.strip() for split in args.splits.split(",") if split.strip()
        ]
        if not split_names:
            raise ValueError("At least one JSON split name is required.")
        for split_name in split_names:
            paths = load_manifest_split(args.dataset_json, split_name)
            results.extend(
                process_paths(
                    paths,
                    output_dir / split_name,
                    configuration,
                    overwrite=args.overwrite,
                    split_name=split_name,
                    preview_count=args.preview_count,
                    preview_dir=preview_dir,
                )
            )
        if args.expected_count is not None and len(results) != args.expected_count:
            logging.getLogger(__name__).warning(
                "Expected %d total JSON cases, but processed %d entries.",
                args.expected_count,
                len(results),
            )
    else:  # protected by argparse
        raise ValueError(f"Unsupported mode: {args.mode}")

    csv_path, json_path = write_run_reports(
        results,
        configuration,
        report_dir,
        mode=args.mode,
        command_arguments=vars(args),
    )

    status_counts = {
        status: sum(result["status"] == status for result in results)
        for status in ("success", "skipped", "error")
    }
    logging.getLogger(__name__).info(
        "Completed: %d success, %d skipped, %d errors.",
        status_counts["success"],
        status_counts["skipped"],
        status_counts["error"],
    )
    logging.getLogger(__name__).info("CSV report: %s", csv_path)
    logging.getLogger(__name__).info("JSON report: %s", json_path)
    return 1 if status_counts["error"] else 0


def main() -> None:
    """Console-script entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    try:
        exit_code = run(args)
    except Exception:
        logging.getLogger(__name__).exception("Enhancement run failed.")
        exit_code = 2
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
