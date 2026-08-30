from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import mne
import numpy as np

PositionMapping = Mapping[str, Sequence[float]]
MontageSource = str | Path | PositionMapping | mne.channels.DigMontage


def _as_position_dict(source: MontageSource) -> dict[str, np.ndarray]:
    """Resolve an MNE montage name, montage file, DigMontage, or position mapping."""
    if isinstance(source, mne.channels.DigMontage):
        positions = source.get_positions()["ch_pos"]
    elif isinstance(source, Mapping):
        positions = source
    else:
        source_str = str(source)
        if source_str in mne.channels.get_builtin_montages():
            montage = mne.channels.make_standard_montage(source_str)
        else:
            path = Path(source).expanduser()
            if not path.is_file():
                available = ", ".join(mne.channels.get_builtin_montages())
                raise ValueError(
                    f"{source!r} is neither a montage file nor a built-in MNE montage. "
                    f"Available built-ins include: {available}"
                )
            montage = mne.channels.read_custom_montage(path)
        positions = montage.get_positions()["ch_pos"]

    result: dict[str, np.ndarray] = {}
    for name, xyz in positions.items():
        value = np.asarray(xyz, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Position for channel {name!r} must contain three finite values.")
        result[str(name)] = value
    return result


def load_electrode_positions(
    source: MontageSource,
    *,
    ch_names: Sequence[str] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Load Cartesian electrode positions in the requested dataset channel order.

    ``aliases`` maps a dataset channel name to the corresponding montage channel,
    for example ``{"HEOL": "A1", "HEOR": "A2"}`` for FACED.
    """
    positions = _as_position_dict(source)
    aliases = dict(aliases or {})
    requested = list(ch_names) if ch_names is not None else list(positions)
    if len(requested) != len(set(requested)):
        raise ValueError("ch_names must not contain duplicates.")

    missing: list[str] = []
    ordered: list[np.ndarray] = []
    for dataset_name in requested:
        montage_name = aliases.get(dataset_name, dataset_name)
        if montage_name not in positions:
            missing.append(f"{dataset_name} (looked for {montage_name})")
        else:
            ordered.append(positions[montage_name])
    if missing:
        raise ValueError("Missing electrode positions: " + ", ".join(missing))
    if not ordered:
        raise ValueError("The montage contains no electrode positions.")

    return requested, np.stack(ordered)
