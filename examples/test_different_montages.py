"""Try SPE with different montage input types.

Run from the repository root:

    python examples/test_different_montages.py
"""

from pathlib import Path

import numpy as np
import torch

from src.spherical_positional_encoding import SphericalPositionalEncoding


EMBEDDING_DIM = 64
BATCH_SIZE = 2
N_WINDOWS = 10


def run_example(name, montage, ch_names, aliases=None):
    """Create SPE, apply it to B,C,W,T input, and print the shapes."""
    spe = SphericalPositionalEncoding(
        embedding_dim=EMBEDDING_DIM,
        montage=montage,
        ch_names=ch_names,
        aliases=aliases,
    )

    x = torch.randn(
        BATCH_SIZE,
        len(ch_names),
        N_WINDOWS,
        EMBEDDING_DIM,
    )
    y = spe(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert tuple(spe.ch_names) == tuple(ch_names)

    print(f"\n{name}")
    print(f"  channels: {len(ch_names)}")
    print(f"  input:    {tuple(x.shape)}")
    print(f"  encoding: {tuple(spe.get_encoding().shape)}")
    print(f"  output:   {tuple(y.shape)}")
    print("  status:   PASS")


def main():
    # 1. Select dataset channels from an MNE built-in montage.
    standard_1020_channels = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"
    ]
    run_example(
        name="MNE built-in: standard_1020",
        montage="standard_1020",
        ch_names=standard_1020_channels,
    )

    # 2. Another MNE built-in montage. Supplying the dataset channel list keeps
    #    its exact channel count and ordering.
    hydrocel_channels = [f"E{i}" for i in range(1, 33)]
    run_example(
        name="MNE built-in: GSN-HydroCel-32",
        montage="GSN-HydroCel-32",
        ch_names=hydrocel_channels,
    )

    # 3. Manual Cartesian coordinates: {channel_name: [x, y, z]}.
    manual_positions = {
        "EEG1": np.array([0.000, 0.080, 0.050]),
        "EEG2": np.array([-0.060, 0.000, 0.050]),
        "EEG3": np.array([0.060, 0.000, 0.050]),
        "EEG4": np.array([0.000, -0.080, 0.050]),
    }
    run_example(
        name="Manual Cartesian coordinates",
        montage=manual_positions,
        ch_names=["EEG3", "EEG1", "EEG4", "EEG2"],
    )

    # 4. Custom montage file supported by MNE.
    montage_file = (
        Path(__file__).resolve().parents[1] / "examples" / "example.sfp"
    )
    run_example(
        name="Custom MNE montage file (.sfp)",
        montage=montage_file,
        ch_names=["EEG1", "EEG2", "EEG3", "EEG4"],
    )

    # 5. FACED: standard_1020 positions with dataset-specific ocular aliases.
    faced_channels = [
        "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
        "FC1", "FC2", "FC5", "FC6",
        "Cz", "C3", "C4", "T7", "T8",
        "CP1", "CP2", "CP5", "CP6",
        "Pz", "P3", "P4", "P7", "P8",
        "PO3", "PO4", "Oz", "O1", "O2",
        "HEOR", "HEOL",
    ]
    run_example(
        name="FACED aliases",
        montage="standard_1020",
        ch_names=faced_channels,
        aliases={"HEOL": "A1", "HEOR": "A2"},
    )

    print("\nAll montage examples passed.")


if __name__ == "__main__":
    main()

