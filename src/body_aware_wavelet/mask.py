"""Body-mask construction for CT volumes."""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary 3D mask."""

    labeled, number_of_components = ndi.label(mask)
    if number_of_components == 0:
        return mask.astype(bool)

    sizes = ndi.sum(
        mask,
        labeled,
        index=np.arange(1, number_of_components + 1),
    )
    largest_label = int(np.argmax(sizes) + 1)
    return labeled == largest_label


def make_body_mask_ct(
    volume_hu: np.ndarray,
    threshold_hu: float = -600.0,
) -> np.ndarray:
    """Create the paper's body mask from a clipped 3D CT volume.

    The sequence matches the manuscript and original notebook:

    1. threshold above ``threshold_hu``;
    2. 3x3x3 binary opening;
    3. 5x5x5 binary closing;
    4. 3D hole filling; and
    5. largest-connected-component selection.
    """

    if volume_hu.ndim != 3:
        raise ValueError(f"Expected a 3D volume, received {volume_hu.shape}.")

    mask = volume_hu > threshold_hu
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3, 3), dtype=bool))
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5, 5), dtype=bool))
    mask = ndi.binary_fill_holes(mask)
    return largest_connected_component(mask).astype(bool)
