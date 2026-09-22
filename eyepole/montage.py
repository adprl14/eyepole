"""Maps physical electrode potentials into recorded EOG channels."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Montage:
    """Linear map from electrode potentials to recorded EOG channels.

    If ``v`` contains the voltage at each physical electrode, the recorded EOG
    channels are

        y = D @ v

    where ``D`` is ``matrix``.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (n_channels, n_electrodes)
        Channel construction matrix.
    channel_names : tuple of str
        Human-readable name for each row of ``matrix``.
    """

    matrix: np.ndarray
    channel_names: tuple[str, ...]

    def __post_init__(self):
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("matrix must be 2-D")
        if len(self.channel_names) != matrix.shape[0]:
            raise ValueError("channel_names must match montage rows")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def from_pairs(cls, geometry, pairs, channel_names=None):
        """Create bipolar EOG channels such as right-minus-left.

        Parameters
        ----------
        geometry : EyeGeometry
            Geometry containing the named physical electrodes.
        pairs : sequence of tuple[str, str]
            Each ``(positive, negative)`` pair creates one channel equal to
            ``positive - negative``.
        channel_names : sequence of str or None
            Optional names for the resulting channels. If omitted, names such
            as ``"R-L"`` are generated automatically.
        """
        matrix = np.zeros((len(pairs), geometry.n_electrodes), dtype=float)

        for row, (positive, negative) in enumerate(pairs):
            matrix[row, geometry.index(positive)] = 1.0
            matrix[row, geometry.index(negative)] = -1.0

        if channel_names is None:
            channel_names = [f"{positive}-{negative}" for positive, negative in pairs]

        return cls(matrix, tuple(channel_names))

    @classmethod
    def from_weights(cls, geometry, channels):
        """Create channels from arbitrary named electrode weights.

        Example
        -------
        ``channels`` can be written as::

            {
                "horizontal": {"R": 1, "L": -1},
                "upper_avg": {"U1": 0.5, "U2": 0.5, "REF": -1},
            }
        """
        names = tuple(channels.keys())
        matrix = np.zeros((len(names), geometry.n_electrodes), dtype=float)

        for row, channel_name in enumerate(names):
            for electrode, weight in channels[channel_name].items():
                matrix[row, geometry.index(electrode)] = float(weight)

        return cls(matrix, names)

    @property
    def n_channels(self):
        """Number of recorded EOG channels."""
        return self.matrix.shape[0]

    @property
    def n_electrodes(self):
        """Number of physical electrodes expected by the montage."""
        return self.matrix.shape[1]
