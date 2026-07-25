"""Case processing, batch execution, and reproducibility reports."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import EnhancementConfig
from .enhancement import enhance_ct_volume_3d
from .io import compare_nifti_geometry, load_ct_nifti, save_ct_nifti
from .qc import save_orthogonal_preview

LOGGER = logging.getLogger(__name__)


def nifti_stem(path: str | Path) -> str:
    """Return a filename without ``.nii`` or ``.nii.gz``."""

    name = Path(path).name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def _compact_debug(debug: dict[str, Any]) -> dict[str, Any]:
    body_mask = debug["body_mask"]
    return {
        "entropy_before": float(debug["entropy_before"]),
        "entropy_after": float(debug["entropy_after"]),
        "entropy_change": float(
            debug["entropy_after"] - debug["entropy_before"]
        ),
        "body_voxels": int(np.count_nonzero(body_mask)),
        "body_fraction": float(np.mean(body_mask)),
        "k": int(debug["k"]),
        "K": int(debug["K"]),
        "exposure_E": None if debug.get("E") is None else float(debug["E"]),
        "histogram_threshold_T": (
            None if debug.get("T") is None else int(debug["T"])
        ),
        "split_point_X": None if debug.get("X") is None else int(debug["X"]),
        "approximation_shape": tuple(debug["approximation_shape"]),
    }


def process_single_case(
    input_path: str | Path,
    output_path: str | Path,
    config: EnhancementConfig,
    *,
    split_name: str = "data",
    preview_path: str | Path | None = None,
) -> dict[str, Any]:
    """Enhance one case, save it, and verify its NIfTI geometry."""

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    started = time.perf_counter()

    volume_hu, spacing_zyx, reference_image = load_ct_nifti(source)
    enhanced_hu, debug = enhance_ct_volume_3d(volume_hu, config)
    save_ct_nifti(
        enhanced_hu,
        reference_image,
        destination,
        save_hu_min=config.save_hu_min,
        save_hu_max=config.save_hu_max,
    )
    geometry = compare_nifti_geometry(source, destination)

    if preview_path is not None:
        save_orthogonal_preview(
            volume_hu,
            enhanced_hu,
            preview_path,
            window_min=config.hu_min,
            window_max=config.hu_max,
        )

    result: dict[str, Any] = {
        "status": "success",
        "split": split_name,
        "filename": source.name,
        "input_path": str(source),
        "output_path": str(destination),
        "shape_zyx": tuple(int(v) for v in volume_hu.shape),
        "spacing_zyx": tuple(float(v) for v in spacing_zyx),
        "elapsed_sec": float(time.perf_counter() - started),
        "output_size_bytes": int(destination.stat().st_size),
        "error": None,
    }
    result.update(_compact_debug(debug))
    result.update(geometry)
    return result


def process_paths(
    file_paths: Iterable[str | Path],
    output_dir: str | Path,
    config: EnhancementConfig,
    *,
    overwrite: bool,
    split_name: str,
    expected_count: int | None = None,
    preview_count: int = 0,
    preview_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Process an ordered collection of NIfTI paths."""

    paths = [Path(path).expanduser().resolve() for path in file_paths]
    if not paths:
        raise ValueError(f"No input paths were supplied for split '{split_name}'.")
    if expected_count is not None and len(paths) != expected_count:
        LOGGER.warning(
            "Expected %d cases for '%s', but found %d.",
            expected_count,
            split_name,
            len(paths),
        )

    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    previews = (
        Path(preview_dir).expanduser().resolve()
        if preview_dir is not None
        else None
    )

    results: list[dict[str, Any]] = []
    total = len(paths)
    successful_previews = 0

    for index, source in enumerate(paths, start=1):
        destination = destination_dir / source.name
        if destination.exists() and not overwrite:
            LOGGER.info(
                "[%03d/%03d] Skipped existing %s",
                index,
                total,
                destination.name,
            )
            results.append(
                {
                    "status": "skipped",
                    "split": split_name,
                    "filename": source.name,
                    "input_path": str(source),
                    "output_path": str(destination),
                    "error": None,
                }
            )
            continue

        preview_path = None
        if previews is not None and successful_previews < preview_count:
            preview_path = (
                previews
                / split_name
                / f"{nifti_stem(source)}_orthogonal_preview.png"
            )

        try:
            result = process_single_case(
                source,
                destination,
                config,
                split_name=split_name,
                preview_path=preview_path,
            )
            results.append(result)
            successful_previews += int(preview_path is not None)
            LOGGER.info(
                "[%03d/%03d] Saved %s | entropy %.4f -> %.4f | %.2f s",
                index,
                total,
                destination.name,
                result["entropy_before"],
                result["entropy_after"],
                result["elapsed_sec"],
            )
            if not result["geometry_preserved"]:
                LOGGER.warning(
                    "Geometry verification failed for %s. Inspect the run report.",
                    source.name,
                )
        except Exception as error:  # continue the batch and report the case
            LOGGER.exception(
                "[%03d/%03d] Failed %s",
                index,
                total,
                source.name,
            )
            results.append(
                {
                    "status": "error",
                    "split": split_name,
                    "filename": source.name,
                    "input_path": str(source),
                    "output_path": str(destination),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    return results


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _csv_ready_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = dict(result)
        for key in ("shape_zyx", "spacing_zyx", "approximation_shape"):
            if key in row and isinstance(row[key], (tuple, list)):
                row[key] = "x".join(str(value) for value in row[key])
        rows.append(row)
    return rows


def write_run_reports(
    results: list[dict[str, Any]],
    config: EnhancementConfig,
    report_dir: str | Path,
    *,
    mode: str,
    command_arguments: dict[str, Any],
) -> tuple[Path, Path]:
    """Write case-level CSV and structured JSON run reports."""

    destination = Path(report_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "case_results.csv"
    json_path = destination / "run_summary.json"

    pd.DataFrame(_csv_ready_rows(results)).to_csv(csv_path, index=False)
    status_counts = Counter(result["status"] for result in results)
    payload = {
        "repository": "body-aware-3d-wavelet-ct-enhancement",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "configuration": config.to_dict(),
        "command_arguments": command_arguments,
        "counts": {
            "total": len(results),
            "success": status_counts.get("success", 0),
            "skipped": status_counts.get("skipped", 0),
            "error": status_counts.get("error", 0),
        },
        "results": results,
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(payload), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return csv_path, json_path
