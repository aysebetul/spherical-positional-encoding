# Spherical Positional Encoding (SPE)

A concise, reusable PyTorch implementation of **Spherical Positional Encoding
(SPE)** for EEG. SPE derives channel positional representations from the
spherical geometry of electrode placement instead of treating EEG channels as
an arbitrary one-dimensional sequence.

> This repository provides the proposed positional encoding module. It does
> not reproduce the complete pretraining and downstream evaluation pipeline of
> the paper.

## Motivation

Transformers are permutation-invariant and therefore require explicit
positional information. Text tokens follow a linear order, while EEG
electrodes form a non-linear spatial configuration defined by the recording
montage. SPE incorporates this montage geometry using the azimuth and
inclination of each electrode.

For electrode `c`, SPE constructs a fixed positional representation:

```text
pos_c = [sin(ω_i θ_c), cos(ω_i θ_c),
         sin(ω_i φ_c), cos(ω_i φ_c)]
```

where `θ_c` is the azimuth, `φ_c` is the inclination, and `ω_i` denotes the
sinusoidal frequencies. The azimuth and inclination encodings are concatenated
to form the channel positional embedding.

SPE is:

- derived from physical electrode locations;
- parameter-free and precomputed;
- applicable to different EEG montages;
- invariant to global coordinate scaling and head-size differences.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/aysebetul/spherical-positional-encoding.git
cd spherical-positional-encoding
pip install -e .
```

## Quick start

```python
import torch

from spherical_positional_encoding import SphericalPositionalEncoding

channel_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]

spe = SphericalPositionalEncoding(
    embedding_dim=64,
    montage="standard_1020",
    ch_names=channel_names,
)

# B: batch, C: channels, W: windows/patches, D: embedding dimension
x = torch.randn(8, len(channel_names), 20, 64)
y = spe(x)

assert y.shape == x.shape
```

Tensor shapes:

```text
Input:          (B, C, W, D)
Spherical PE:   (1, C, 1, D)
Output:         (B, C, W, D)
```

The channel order in `ch_names` must match the channel axis `C` of the input
tensor. Montage positions are looked up by channel name and returned in the
provided dataset order.

## Montage inputs

### Built-in MNE montage

Any montage listed by `mne.channels.get_builtin_montages()` can be used:

```python
spe = SphericalPositionalEncoding(
    embedding_dim=64,
    montage="GSN-HydroCel-128",
    ch_names=dataset_channel_names,
)
```

### Custom montage file

Files supported by `mne.channels.read_custom_montage()` can be passed directly:

```python
spe = SphericalPositionalEncoding(
    embedding_dim=64,
    montage="path/to/montage.sfp",
    ch_names=dataset_channel_names,
)
```

### Manual Cartesian coordinates

```python
manual_positions = {
    "EEG1": [0.0, 1.0, 0.5],
    "EEG2": [1.0, 0.0, 0.5],
}

spe = SphericalPositionalEncoding(
    embedding_dim=64,
    montage=manual_positions,
    ch_names=["EEG1", "EEG2"],
)
```

### Dataset-specific aliases

Dataset channel names can be mapped to their corresponding montage names. For
FACED, the ocular labels correspond to the left and right mastoid positions:

```python
spe = SphericalPositionalEncoding(
    embedding_dim=64,
    montage="standard_1020",
    ch_names=faced_channel_names,
    aliases={
        "HEOL": "A1",
        "HEOR": "A2",
    },
)
```

Missing positions raise an error instead of silently producing a misaligned
encoding.

## Spatial scale

MNE spherical coordinates are kept in radians. Before applying the sinusoidal
mapping, SPE multiplies both angular coordinates by one global
`spatial_scale`.

The default value reproduces the configuration reported in the paper:

```python
PAPER_SPATIAL_SCALE = 180.0 / math.pi
```

```python
from spherical_positional_encoding import (
    PAPER_SPATIAL_SCALE,
    RADIAN_SPATIAL_SCALE,
    SphericalPositionalEncoding,
)

# Paper configuration; this is also the default.
paper_spe = SphericalPositionalEncoding(
    64,
    "standard_1020",
    ch_names=channel_names,
    spatial_scale=PAPER_SPATIAL_SCALE,
)

# Unscaled radian coordinates.
radian_spe = SphericalPositionalEncoding(
    64,
    "standard_1020",
    ch_names=channel_names,
    spatial_scale=RADIAN_SPATIAL_SCALE,
)
```

The scale controls the angular frequency range of the resulting positional
features. `PAPER_SPATIAL_SCALE` is mathematically equivalent to applying the
sinusoidal mapping to degree-valued spherical coordinates.

## Temporal positional encoding

The standard sinusoidal temporal encoding is provided as a separate module. It
encodes the ordered window/patch axis `W` and is shared across channels:

```python
from spherical_positional_encoding import TemporalPositionalEncoding

temporal_pe = TemporalPositionalEncoding(
    embedding_dim=64,
    max_len=1_000,
    dropout=0.1,
)

y = temporal_pe(x)
assert y.shape == x.shape
```

```text
Temporal PE:    (1, 1, W, D)
Spherical PE:   (1, C, 1, D)
Final PE:       (1, C, W, D)
```

The combined positional encoding is obtained by summation:

```python
y = temporal_pe(spe(x))
```

## Examples

Run the montage-input example:

```bash
python examples/test_different_montages.py
```

Run the combined spherical and temporal encoding example:

```bash
python examples/show_spatiotemporal_encoding.py
```

## Paper

SPE was introduced in:

> Ayşe Betül Yüce and Sebastian Stober. **Benchmarking Positional Encoding
> Strategies for Transformer-Based EEG Foundation Models: A Systematic
> Comparison.** 10th Graz Brain-Computer Interface Conference, 2026.

The paper evaluates SPE within the CBraMod backbone under linear-probing and
fine-tuning protocols on motor-imagery and emotion-recognition tasks. The
results indicate that positional-encoding performance is task-dependent; SPE
provides strong representations for motor imagery while requiring no learned
parameters and remaining applicable across montages.

## Citation

```bibtex
@inproceedings{yuce2026benchmarking,
  title     = {Benchmarking Positional Encoding Strategies for
               Transformer-Based EEG Foundation Models: A Systematic Comparison},
  author    = {Yüce, Ayşe Betül and Stober, Sebastian},
  booktitle = {Proceedings of the 10th Graz Brain-Computer Interface Conference},
  year      = {2026}
}
```

## License

See `LICENSE` for licensing information.
