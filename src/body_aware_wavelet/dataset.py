"""Input discovery for single-file, directory, and JSON-manifest runs."""

from __future__ import annotations

import json
from pathlib import Path


def list_nifti_files(folder: str | Path) -> list[Path]:
    """Return sorted, non-recursive ``.nii.gz`` files from a directory."""

    input_directory = Path(folder).expanduser().resolve()
    if not input_directory.is_dir():
        raise NotADirectoryError(f"Input directory was not found: {input_directory}")

    paths = sorted(input_directory.glob("*.nii.gz"))
    if not paths:
        raise FileNotFoundError(
            f"No .nii.gz files were found directly inside {input_directory}."
        )
    return paths


def _resolve_manifest_image(
    value: str,
    manifest_directory: Path,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    return path.resolve()


def load_manifest_split(
    dataset_json_path: str | Path,
    split_name: str,
) -> list[Path]:
    """Load image paths from one split in a MONAI-style dataset JSON.

    Each split entry may be either ``{"image": "path/to/case.nii.gz"}`` or a
    direct string path. Relative paths are resolved from the JSON file's
    directory.
    """

    manifest_path = Path(dataset_json_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset JSON was not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as stream:
        dataset = json.load(stream)

    if split_name not in dataset:
        raise KeyError(
            f"Split '{split_name}' was not found in {manifest_path.name}. "
            f"Available keys: {sorted(dataset)}"
        )
    if not isinstance(dataset[split_name], list):
        raise TypeError(f"Split '{split_name}' must contain a JSON list.")

    paths: list[Path] = []
    for index, item in enumerate(dataset[split_name]):
        if isinstance(item, str):
            image_value = item
        elif isinstance(item, dict) and isinstance(item.get("image"), str):
            image_value = item["image"]
        else:
            raise TypeError(
                f"Entry {index} in split '{split_name}' must be a path string "
                "or an object containing an 'image' string."
            )
        paths.append(_resolve_manifest_image(image_value, manifest_path.parent))

    if not paths:
        raise ValueError(f"Split '{split_name}' contains no image paths.")
    return paths
