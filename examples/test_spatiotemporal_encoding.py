"""Compute and visualize spherical, temporal, and combined encoding.

Run from the repository root:

    python examples/show_spatiotemporal_encoding.py

Optionally save the figure instead of opening a window:

    python examples/show_spatiotemporal_encoding.py --save encoding.png
"""

import argparse
import torch

from src.spherical_positional_encoding import SphericalPositionalEncoding
from src.temporal_positional_encoding import TemporalPositionalEncoding


def main(save_path=None):
    torch.manual_seed(42)

    batch_size = 2
    channel_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]
    n_windows = 20
    embedding_dim = 64

    # Random input: (B, C, W, T)
    x = torch.randn(
        batch_size,
        len(channel_names),
        n_windows,
        embedding_dim,
    )

    spherical_pe = SphericalPositionalEncoding(
        embedding_dim=embedding_dim,
        montage="standard_1020",
        ch_names=channel_names,
        dropout=0.0,
    )
    temporal_pe = TemporalPositionalEncoding(
        embedding_dim=embedding_dim,
        max_len=n_windows,
        dropout=0.0,
    )

    # Raw encoding components. Broadcasting produces (1, C, W, T).
    spherical_encoding = spherical_pe.get_encoding()  # (1, C, 1, T)
    temporal_encoding = temporal_pe.get_encoding(n_windows)  # (1, 1, W, T)
    final_encoding = spherical_encoding + temporal_encoding  # (1, C, W, T)

    # Add the combined encoding to every sample in the batch.
    final_output = x + final_encoding

    # Applying the two modules sequentially must give the same result when
    # dropout is zero.
    sequential_output = temporal_pe(spherical_pe(x))
    assert torch.allclose(final_output, sequential_output, atol=1e-6)
    assert final_output.shape == x.shape

    print("Input shape:             ", tuple(x.shape))
    print("Spherical encoding shape:", tuple(spherical_encoding.shape))
    print("Temporal encoding shape: ", tuple(temporal_encoding.shape))
    print("Final encoding shape:    ", tuple(final_encoding.shape))
    print("Final output shape:      ", tuple(final_output.shape))
    print("Sequential check:         PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Optional path for saving the figure.")
    args = parser.parse_args()
    main(save_path=args.save)

