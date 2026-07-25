"""Optional visual quality-control outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_orthogonal_preview(
    original_hu: np.ndarray,
    enhanced_hu: np.ndarray,
    output_path: str | Path,
    *,
    window_min: float = -1000.0,
    window_max: float = 1000.0,
) -> Path:
    """Save matched middle axial, coronal, and sagittal slices."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    z_index = original_hu.shape[0] // 2
    y_index = original_hu.shape[1] // 2
    x_index = original_hu.shape[2] // 2

    original_slices = (
        original_hu[z_index, :, :],
        original_hu[:, y_index, :],
        original_hu[:, :, x_index],
    )
    enhanced_slices = (
        enhanced_hu[z_index, :, :],
        enhanced_hu[:, y_index, :],
        enhanced_hu[:, :, x_index],
    )
    view_names = ("Axial", "Coronal", "Sagittal")

    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for column, view_name in enumerate(view_names):
        axes[0, column].imshow(
            original_slices[column],
            cmap="gray",
            vmin=window_min,
            vmax=window_max,
        )
        axes[0, column].set_title(f"Original - {view_name}")
        axes[0, column].axis("off")

        axes[1, column].imshow(
            enhanced_slices[column],
            cmap="gray",
            vmin=window_min,
            vmax=window_max,
        )
        axes[1, column].set_title(f"Enhanced - {view_name}")
        axes[1, column].axis("off")

    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
