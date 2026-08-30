from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TemporalPositionalEncoding(nn.Module):
    """Standard sinusoidal temporal encoding for tensors shaped ``(B, C, W, D)``.

    ``W`` is the window axis and ``D`` is the embedding dimension. The
    encoding varies along ``W``, occupies the embedding dimension ``D``,
    and is shared across batches and EEG channels.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        max_len: int = 5_000,
        dropout: float = 0.0,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if max_len <= 0:
            raise ValueError("max_len must be positive.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1.")

        positions = torch.arange(max_len, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(base) / embedding_dim)
        )
        encoding = torch.zeros(max_len, embedding_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )

        # (W, T) becomes (1, 1, W, T), ready for B,C,W,T broadcasting.
        self.register_buffer("encoding", encoding[None, None, :, :])
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shaped (B, C, W, T), got {tuple(x.shape)}.")
        if x.shape[3] != self.embedding_dim:
            raise ValueError(
                f"Input D is {x.shape[3]}, but embedding_dim is {self.embedding_dim}."
            )
        if x.shape[2] > self.max_len:
            raise ValueError(
                f"Input W is {x.shape[2]}, but max_len is {self.max_len}."
            )

        encoding = self.encoding[:, :, : x.shape[2], :].to(
            device=x.device, dtype=x.dtype
        )
        return self.dropout(x + encoding)

    def get_encoding(self, length: int | None = None) -> Tensor:
        """Return ``(1, 1, length, D)``; default to the configured maximum."""
        if length is None:
            return self.encoding
        if not 0 < length <= self.max_len:
            raise ValueError(f"length must be between 1 and {self.max_len}.")
        return self.encoding[:, :, :length, :]
