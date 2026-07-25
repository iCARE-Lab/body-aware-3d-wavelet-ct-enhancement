"""Exact body-aware 3D Haar-wavelet enhancement implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pywt
import scipy.ndimage as ndi

from .config import EnhancementConfig
from .mask import make_body_mask_ct


def normalize_hu_to_levels(
    volume_hu: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    levels: int = 2048,
) -> np.ndarray:
    """Clip HU and linearly map the volume to ``[0, levels - 1]``."""

    clipped = np.clip(volume_hu, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    normalized *= levels - 1
    return normalized.astype(np.float32)


def denormalize_levels_to_hu(
    volume_normalized: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    levels: int = 2048,
) -> np.ndarray:
    """Map values from ``[0, levels - 1]`` back to HU."""

    volume_hu = volume_normalized / (levels - 1)
    volume_hu = volume_hu * (hu_max - hu_min) + hu_min
    return volume_hu.astype(np.float32)


def pad_to_even_3d(
    volume: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Pad each odd dimension by one voxel at its positive edge."""

    pad_width = [(0, 1 if size % 2 else 0) for size in volume.shape]
    padded = np.pad(volume, pad_width=pad_width, mode="edge")
    return padded, pad_width


def crop_from_pad_3d(
    volume: np.ndarray,
    pad_width: list[tuple[int, int]],
) -> np.ndarray:
    """Crop a reconstructed volume back to its original dimensions."""

    z_end = volume.shape[0] - pad_width[0][1]
    y_end = volume.shape[1] - pad_width[1][1]
    x_end = volume.shape[2] - pad_width[2][1]
    return volume[:z_end, :y_end, :x_end]


def resize_mask_to_shape(
    mask: np.ndarray,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Resize a binary mask with nearest-neighbor interpolation."""

    zoom_factors = [target / source for target, source in zip(target_shape, mask.shape)]
    resized = ndi.zoom(mask.astype(np.float32), zoom=zoom_factors, order=0)
    if tuple(resized.shape) != tuple(target_shape):
        raise RuntimeError(
            f"Wavelet-domain mask has shape {resized.shape}; expected {target_shape}."
        )
    return resized > 0.5


def shannon_entropy(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
    num_bins: int = 256,
) -> float:
    """Compute Shannon entropy from a fixed-bin histogram."""

    values = volume[mask] if mask is not None else volume.ravel()
    values = values.astype(np.float32)
    if values.size == 0:
        return 0.0

    value_min = float(values.min())
    value_max = float(values.max())
    if value_max <= value_min:
        return 0.0

    histogram, _ = np.histogram(
        values,
        bins=num_bins,
        range=(value_min, value_max),
        density=False,
    )
    probability = histogram.astype(np.float64) / histogram.sum()
    probability = probability[probability > 0]
    return float(-np.sum(probability * np.log2(probability)))


def entropy_enhance_3d_wavelet(
    volume_normalized: np.ndarray,
    body_mask: np.ndarray | None = None,
    *,
    wavelet: str = "haar",
    levels: int = 2048,
    eps: float = 1.0e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remap only the approximation band of a single-level 3D DWT.

    This function implements manuscript Eqs. (5)-(18). All seven detail bands
    remain unchanged.
    """

    volume_normalized = volume_normalized.astype(np.float32)
    padded_volume, pad_width = pad_to_even_3d(volume_normalized)

    if body_mask is None:
        padded_mask = np.ones_like(padded_volume, dtype=bool)
    else:
        padded_mask, _ = pad_to_even_3d(body_mask.astype(np.uint8))
        padded_mask = padded_mask.astype(bool)

    coefficients = pywt.dwtn(
        padded_volume,
        wavelet=wavelet,
        axes=(0, 1, 2),
        mode="periodization",
    )

    approximation_key = "aaa"
    approximation = coefficients[approximation_key].astype(np.float32)
    approximation_mask = resize_mask_to_shape(
        padded_mask,
        approximation.shape,
    )

    quantized = np.rint(approximation).astype(np.int32)
    valid_values = quantized[approximation_mask]
    if valid_values.size == 0:
        valid_values = quantized.ravel()

    lower_bound = int(valid_values.min())
    upper_bound = int(valid_values.max())

    if upper_bound <= lower_bound:
        reconstructed = pywt.idwtn(
            coefficients,
            wavelet=wavelet,
            axes=(0, 1, 2),
            mode="periodization",
        )
        reconstructed = crop_from_pad_3d(reconstructed, pad_width)
        reconstructed = np.clip(reconstructed, 0, levels - 1).astype(np.float32)
        return reconstructed, {
            "k": lower_bound,
            "K": upper_bound,
            "E": None,
            "T": None,
            "X": None,
            "approximation_shape": tuple(approximation.shape),
        }

    quantized_clipped = np.clip(quantized, lower_bound, upper_bound)
    histogram = np.bincount(
        (quantized_clipped[approximation_mask] - lower_bound).ravel(),
        minlength=upper_bound - lower_bound + 1,
    ).astype(np.float64)
    coefficient_values = np.arange(
        lower_bound,
        upper_bound + 1,
        dtype=np.float64,
    )

    histogram_mean = np.sum(coefficient_values * histogram) / (
        np.sum(histogram) + eps
    )
    exposure = (histogram_mean - lower_bound) / (
        upper_bound - lower_bound + eps
    )
    exposure = float(np.clip(exposure, 0.0, 1.0))

    clipping_threshold = int(
        np.rint(np.sum(histogram) / (upper_bound - lower_bound + 1))
    )
    clipping_threshold = max(clipping_threshold, 1)
    truncated_histogram = np.minimum(histogram, clipping_threshold)

    split_point = int(
        np.rint(
            lower_bound
            + (upper_bound - lower_bound) * (1.0 - exposure)
        )
    )
    split_point = max(lower_bound, min(split_point, upper_bound))
    split_index = split_point - lower_bound

    lookup_table = np.arange(
        lower_bound,
        upper_bound + 1,
        dtype=np.float32,
    )

    lower_histogram = truncated_histogram[: split_index + 1].copy()
    lower_mass = lower_histogram.sum()
    if lower_mass > 0:
        lower_histogram /= lower_mass
        lower_cdf = np.cumsum(lower_histogram)
        lookup_table[: split_index + 1] = (
            lower_bound
            + (split_point - lower_bound) * lower_cdf
        )

    upper_histogram = truncated_histogram[split_index + 1 :].copy()
    upper_mass = upper_histogram.sum()
    if upper_mass > 0 and len(upper_histogram) > 0:
        upper_histogram /= upper_mass
        upper_cdf = np.cumsum(upper_histogram)
        lookup_table[split_index + 1 :] = (
            split_point
            + 1
            + (upper_bound - split_point - 1) * upper_cdf
        )

    enhanced_approximation = approximation.copy()
    mapped_values = lookup_table[quantized_clipped - lower_bound]
    enhanced_approximation[approximation_mask] = mapped_values[
        approximation_mask
    ]
    enhanced_approximation[~approximation_mask] = approximation[
        ~approximation_mask
    ]

    enhanced_coefficients = coefficients.copy()
    enhanced_coefficients[approximation_key] = enhanced_approximation

    reconstructed = pywt.idwtn(
        enhanced_coefficients,
        wavelet=wavelet,
        axes=(0, 1, 2),
        mode="periodization",
    )
    reconstructed = crop_from_pad_3d(reconstructed, pad_width)
    reconstructed = np.clip(reconstructed, 0, levels - 1).astype(np.float32)

    return reconstructed, {
        "k": lower_bound,
        "K": upper_bound,
        "E": exposure,
        "T": clipping_threshold,
        "X": split_point,
        "approximation_shape": tuple(approximation.shape),
        "histogram": histogram,
        "truncated_histogram": truncated_histogram,
    }


def enhance_ct_volume_3d(
    volume_hu: np.ndarray,
    config: EnhancementConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the complete proposed enhancement pipeline.

    The implementation follows manuscript Eqs. (1)-(20):

    * clip to the configured HU window;
    * construct the 3D body mask;
    * normalize to discrete intensity levels;
    * suppress the outside-body background during enhancement;
    * remap only the 3D approximation band;
    * reconstruct and return to HU; and
    * restore original outside-body voxels.
    """

    if volume_hu.ndim != 3:
        raise ValueError(f"Expected a 3D volume, received {volume_hu.shape}.")
    parameters = config or EnhancementConfig()

    clipped = np.clip(
        volume_hu,
        parameters.hu_min,
        parameters.hu_max,
    ).astype(np.float32)
    body_mask = make_body_mask_ct(
        clipped,
        threshold_hu=parameters.body_threshold_hu,
    )
    normalized = normalize_hu_to_levels(
        clipped,
        hu_min=parameters.hu_min,
        hu_max=parameters.hu_max,
        levels=parameters.levels,
    )

    normalized_masked = normalized.copy()
    normalized_masked[~body_mask] = 0

    enhanced_normalized, debug = entropy_enhance_3d_wavelet(
        normalized_masked,
        body_mask=body_mask,
        wavelet=parameters.wavelet,
        levels=parameters.levels,
        eps=parameters.eps,
    )
    enhanced_hu = denormalize_levels_to_hu(
        enhanced_normalized,
        hu_min=parameters.hu_min,
        hu_max=parameters.hu_max,
        levels=parameters.levels,
    )

    if parameters.preserve_outside_mask:
        output_hu = volume_hu.copy().astype(np.float32)
        output_hu[body_mask] = enhanced_hu[body_mask]
    else:
        output_hu = enhanced_hu

    debug["entropy_before"] = shannon_entropy(
        normalized,
        mask=body_mask,
        num_bins=parameters.entropy_bins,
    )
    debug["entropy_after"] = shannon_entropy(
        enhanced_normalized,
        mask=body_mask,
        num_bins=parameters.entropy_bins,
    )
    debug["body_mask"] = body_mask
    return output_hu, debug
