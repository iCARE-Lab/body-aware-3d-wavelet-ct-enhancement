"""Configuration for the body-aware CT enhancement pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class EnhancementConfig:
    """Parameters used by the proposed enhancement method."""

    hu_min: float = -1000.0
    hu_max: float = 1000.0
    levels: int = 2048
    wavelet: str = "haar"
    body_threshold_hu: float = -600.0
    preserve_outside_mask: bool = True
    entropy_bins: int = 256
    eps: float = 1.0e-8
    save_hu_min: float = -1024.0
    save_hu_max: float = 3071.0

    def __post_init__(self) -> None:
        if self.hu_max <= self.hu_min:
            raise ValueError("hu_max must be greater than hu_min.")
        if self.levels < 2:
            raise ValueError("levels must be at least 2.")
        if self.entropy_bins < 2:
            raise ValueError("entropy_bins must be at least 2.")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")
        if self.save_hu_max <= self.save_hu_min:
            raise ValueError("save_hu_max must be greater than save_hu_min.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)
