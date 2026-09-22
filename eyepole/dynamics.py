"""Simple temporal models for the eye dipole."""

import numpy as np


class RandomWalkDipole:
    """Random-walk model for a three-dimensional eye dipole.

    The state model is

        p[t] = A @ p[t-1] + w[t]
        w[t] ~ Normal(0, Q)

    By default ``A`` is the identity matrix, so the previous dipole is simply
    the best prediction of the next dipole before seeing a new EOG sample.

    Parameters
    ----------
    Q : float or numpy.ndarray, default=1e-4
        Process-noise covariance. A scalar creates ``Q * I``. Larger values let
        the estimated dipole move more freely from sample to sample; smaller
        values impose stronger temporal smoothness.
    A : numpy.ndarray or None, shape (3, 3), default=None
        State transition matrix. ``None`` uses the identity matrix.
    """

    def __init__(self, Q=1e-4, A=None):
        self.A = np.eye(3) if A is None else np.asarray(A, dtype=float)
        if self.A.shape != (3, 3):
            raise ValueError("A must have shape (3, 3)")

        self.Q = np.eye(3) * float(Q) if np.isscalar(Q) else np.asarray(Q, dtype=float)
        if self.Q.shape != (3, 3):
            raise ValueError("Q must be a scalar or have shape (3, 3)")
