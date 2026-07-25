"""Synthetic tests that require no patient or benchmark data."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import nibabel as nib
import numpy as np

from body_aware_wavelet import EnhancementConfig, enhance_ct_volume_3d
from body_aware_wavelet.io import compare_nifti_geometry
from body_aware_wavelet.runner import process_single_case


def synthetic_ct(shape: tuple[int, int, int] = (17, 19, 21)) -> np.ndarray:
    """Create an odd-shaped CT-like volume in internal [z, y, x] order."""

    z_grid, y_grid, x_grid = np.indices(shape)
    center = (np.asarray(shape) - 1) / 2
    radius = (
        ((z_grid - center[0]) / 7.0) ** 2
        + ((y_grid - center[1]) / 8.0) ** 2
        + ((x_grid - center[2]) / 9.0) ** 2
    )
    volume = np.full(shape, -1000.0, dtype=np.float32)
    body = radius <= 1.0
    volume[body] = (
        -100.0
        + 12.0 * (x_grid[body] - center[2])
        + 8.0 * (z_grid[body] - center[0])
    )
    volume[7:11, 8:12, 9:13] = 650.0
    return volume


class EnhancementTests(unittest.TestCase):
    def test_enhancement_preserves_shape_and_outside_body_values(self) -> None:
        original = synthetic_ct()
        enhanced, debug = enhance_ct_volume_3d(
            original,
            EnhancementConfig(),
        )

        self.assertEqual(enhanced.shape, original.shape)
        self.assertTrue(np.isfinite(enhanced).all())
        self.assertTrue(
            np.array_equal(
                enhanced[~debug["body_mask"]],
                original[~debug["body_mask"]],
            )
        )
        self.assertGreater(int(debug["body_mask"].sum()), 0)

    def test_constant_background_volume_is_handled(self) -> None:
        original = np.full((9, 11, 13), -1000.0, dtype=np.float32)
        enhanced, debug = enhance_ct_volume_3d(original, EnhancementConfig())

        self.assertTrue(np.array_equal(enhanced, original))
        self.assertEqual(debug["entropy_before"], 0.0)
        self.assertEqual(debug["entropy_after"], 0.0)

    def test_nifti_roundtrip_preserves_geometry_and_filename(self) -> None:
        volume_zyx = synthetic_ct()
        volume_xyz = np.transpose(np.rint(volume_zyx).astype(np.int16), (2, 1, 0))
        affine = np.array(
            [
                [0.8, 0.0, 0.0, -120.0],
                [0.0, 0.9, 0.0, -90.0],
                [0.0, 0.0, 1.5, 35.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            input_path = temporary / "case_0001.nii.gz"
            output_path = temporary / "enhanced" / input_path.name

            image = nib.Nifti1Image(volume_xyz, affine)
            image.set_qform(affine, 1)
            image.set_sform(affine, 2)
            nib.save(image, input_path)

            result = process_single_case(
                input_path,
                output_path,
                EnhancementConfig(),
            )
            geometry = compare_nifti_geometry(input_path, output_path)

            self.assertEqual(output_path.name, input_path.name)
            self.assertTrue(output_path.is_file())
            self.assertTrue(result["geometry_preserved"])
            self.assertTrue(geometry["geometry_preserved"])


if __name__ == "__main__":
    unittest.main()
