"""Geometry of EOG electrodes around the eye."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class EyeGeometry:
    """Positions of EOG electrodes relative to the center of the eye.

    Parameters
    ----------
    names : tuple of str
        Electrode names. The order must match the rows of ``positions``.
    positions : numpy.ndarray, shape (n_electrodes, 3)
        Cartesian electrode coordinates in meters. The eye center is the
        origin ``[0, 0, 0]``.

    Notes
    -----
    The coordinate axes are intentionally not assigned anatomical labels by
    the class. The user only needs to use one consistent coordinate system for
    electrode positions and dipole vectors.
    """

    names: tuple[str, ...]
    positions: np.ndarray

    def __post_init__(self):
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n_electrodes, 3)")
        if len(self.names) != len(positions):
            raise ValueError("names and positions must have the same length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("electrode names must be unique")
        object.__setattr__(self, "positions", positions)

    @classmethod
    def from_dict(cls, electrodes: Mapping[str, Sequence[float]]):
        """Create geometry from electrode coordinates given in meters."""
        names = tuple(electrodes.keys())
        positions = np.asarray([electrodes[name] for name in names], dtype=float)
        return cls(names, positions)

    @classmethod
    def from_mm(cls, electrodes: Mapping[str, Sequence[float]]):
        """Create geometry from electrode coordinates given in millimeters."""
        names = tuple(electrodes.keys())
        positions_mm = np.asarray(
            [electrodes[name] for name in names],
            dtype=float,
        )
        return cls(names, positions_mm * 1e-3)

    @property
    def n_electrodes(self):
        """Number of physical electrodes in the geometry."""
        return len(self.names)

    def index(self, name):
        """Return the row index associated with an electrode name."""
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown electrode {name!r}") from exc
