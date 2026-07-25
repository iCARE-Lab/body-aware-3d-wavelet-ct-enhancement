"""NIfTI input/output with explicit geometry preservation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def _is_nifti_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def load_ct_nifti(
    path: str | Path,
) -> tuple[np.ndarray, tuple[float, float, float], nib.Nifti1Image]:
    """Load a 3D CT volume and return it in ``[z, y, x]`` order.

    NiBabel exposes arrays in ``[x, y, z]`` order. The enhancement code uses
    ``[z, y, x]`` internally, matching the original notebook implementation.
    """

    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input NIfTI file was not found: {input_path}")
    if not _is_nifti_path(input_path):
        raise ValueError(f"Expected a .nii or .nii.gz file: {input_path}")

    reference_image = nib.load(str(input_path))
    volume_xyz = np.asarray(reference_image.dataobj).astype(np.float32)
    if volume_xyz.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, but {input_path.name} has shape "
            f"{volume_xyz.shape}."
        )

    volume_zyx = np.transpose(volume_xyz, (2, 1, 0))
    spacing_xyz = tuple(float(v) for v in reference_image.header.get_zooms()[:3])
    spacing_zyx = tuple(reversed(spacing_xyz))
    return volume_zyx, spacing_zyx, reference_image


def save_ct_nifti(
    volume_hu_zyx: np.ndarray,
    reference_image: nib.Nifti1Image,
    output_path: str | Path,
    *,
    save_hu_min: float = -1024.0,
    save_hu_max: float = 3071.0,
) -> Path:
    """Save an enhanced CT using the source affine, header, q-form, and s-form."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if volume_hu_zyx.ndim != 3:
        raise ValueError(
            f"Expected a 3D enhanced volume, but received {volume_hu_zyx.shape}."
        )

    volume_hu_zyx = np.clip(volume_hu_zyx, save_hu_min, save_hu_max)
    volume_int16_zyx = np.rint(volume_hu_zyx).astype(np.int16)
    output_xyz = np.transpose(volume_int16_zyx, (2, 1, 0))

    header = reference_image.header.copy()
    header.set_data_dtype(np.int16)
    output_image = nib.Nifti1Image(output_xyz, reference_image.affine, header)

    qform_code = int(reference_image.header["qform_code"])
    sform_code = int(reference_image.header["sform_code"])
    output_image.set_qform(reference_image.get_qform(), qform_code)
    output_image.set_sform(reference_image.get_sform(), sform_code)

    nib.save(output_image, str(destination))
    return destination


def _same_matrix(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.allclose(left, right, rtol=0.0, atol=1.0e-5))


def compare_nifti_geometry(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare source and output NIfTI geometry after saving."""

    source = nib.load(str(Path(input_path).expanduser().resolve()))
    output = nib.load(str(Path(output_path).expanduser().resolve()))

    source_qform, source_qcode = source.get_qform(coded=True)
    output_qform, output_qcode = output.get_qform(coded=True)
    source_sform, source_scode = source.get_sform(coded=True)
    output_sform, output_scode = output.get_sform(coded=True)

    shape_preserved = tuple(source.shape) == tuple(output.shape)
    affine_preserved = bool(
        np.allclose(source.affine, output.affine, rtol=0.0, atol=1.0e-5)
    )
    spacing_preserved = bool(
        np.allclose(
            source.header.get_zooms()[:3],
            output.header.get_zooms()[:3],
            rtol=0.0,
            atol=1.0e-6,
        )
    )
    qform_preserved = (
        int(source_qcode) == int(output_qcode)
        and _same_matrix(source_qform, output_qform)
    )
    sform_preserved = (
        int(source_scode) == int(output_scode)
        and _same_matrix(source_sform, output_sform)
    )

    return {
        "shape_preserved": shape_preserved,
        "affine_preserved": affine_preserved,
        "spacing_preserved": spacing_preserved,
        "qform_preserved": qform_preserved,
        "sform_preserved": sform_preserved,
        "geometry_preserved": all(
            (
                shape_preserved,
                affine_preserved,
                spacing_preserved,
                qform_preserved,
                sform_preserved,
            )
        ),
    }
