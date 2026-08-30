from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from mne.transforms import _cart_to_sph
from torch import Tensor, nn

from .montage import MontageSource, load_electrode_positions


def _cartesian_to_angles(xyz):
    """Use MNE and return azimuth and polar angle in radians."""
    spherical = _cart_to_sph(xyz)  # columns: radius, azimuth, polar angle
    return spherical[:, 1], spherical[:, 2]


def _sinusoidal(
    values: Tensor,
    width: int,
    base: float,
    frequency_scale_dim: int,
) -> Tensor:
    if width % 2:
        raise ValueError("Each angular encoding width must be even.")
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(base) / frequency_scale_dim)
    )
    encoded = torch.empty(values.numel(), width, dtype=torch.float32)
    encoded[:, 0::2] = torch.sin(values[:, None] * frequencies[None, :])
    encoded[:, 1::2] = torch.cos(values[:, None] * frequencies[None, :])
    return encoded


class SphericalPositionalEncoding(nn.Module):
    """Add electrode-location encodings to an EEG tensor shaped ``(B, C, W, D)``.

    ``C`` follows ``ch_names``, ``W`` is the window axis, and ``D`` is ``embedding_dim``. The channel encoding is shared across windows.
    """

    def __init__(
        self,
        embedding_dim: int,
        montage: MontageSource,
        *,
        ch_names: Sequence[str] | None = None,
        aliases: Mapping[str, str] | None = None,
        dropout: float = 0.0,
        base: float = 10_000.0,
        spatial_scale: float = 180.0 / math.pi
    ) -> None:
        super().__init__()
        if embedding_dim % 4:
            raise ValueError("embedding_dim must be divisible by 4 (theta/phi, sin/cos).")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1.")

        ordered_names, xyz = load_electrode_positions(
            montage, ch_names=ch_names, aliases=aliases
        )
        azimuth, inclination = _cartesian_to_angles(xyz)
        azimuth = azimuth * spatial_scale
        inclination = inclination * spatial_scale
        half = embedding_dim // 2
        theta = _sinusoidal(
            torch.from_numpy(azimuth).float(), half, base, embedding_dim
        )
        phi = _sinusoidal(
            torch.from_numpy(inclination).float(), half, base, embedding_dim
        )
        encoding = torch.cat((theta, phi), dim=-1)[None, :, None, :]

        self.embedding_dim = embedding_dim
        self.ch_names = tuple(ordered_names)
        self.spatial_scale = float(spatial_scale)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("encoding", encoding)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shaped (B, C, W, D), got {tuple(x.shape)}.")
        if x.shape[1] != len(self.ch_names):
            raise ValueError(
                f"Input has {x.shape[1]} channels, but the montage has {len(self.ch_names)}."
            )
        if x.shape[3] != self.embedding_dim:
            raise ValueError(
                f"Input D is {x.shape[3]}, but embedding_dim is {self.embedding_dim}."
            )
        encoding = self.encoding.to(device=x.device, dtype=x.dtype)
        return self.dropout(x + encoding)

    def get_encoding(self) -> Tensor:
        return self.encoding
