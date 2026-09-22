"""Result containers returned by EyePole filters."""

from dataclasses import dataclass
import numpy as np


@dataclass
class DipoleFilterResult:
    """Dipole estimates and their uncertainty over time.

    Parameters
    ----------
    filtered : numpy.ndarray, shape (n_samples, 3)
        Causal estimates. Each sample uses EOG up to and including that time.
    filtered_covariance : numpy.ndarray, shape (n_samples, 3, 3)
        Posterior covariance for each filtered dipole estimate.
    predicted : numpy.ndarray, shape (n_samples, 3)
        One-step predictions before incorporating each EOG sample.
    predicted_covariance : numpy.ndarray, shape (n_samples, 3, 3)
        Covariance of each one-step prediction.
    smoothed : numpy.ndarray or None, shape (n_samples, 3)
        RTS-smoothed estimates. These use the whole recording, including future
        samples, and are populated only when smoothing is requested.
    smoothed_covariance : numpy.ndarray or None, shape (n_samples, 3, 3)
        Covariance associated with the smoothed estimates.
    """

    filtered: np.ndarray
    filtered_covariance: np.ndarray
    predicted: np.ndarray
    predicted_covariance: np.ndarray
    smoothed: np.ndarray | None = None
    smoothed_covariance: np.ndarray | None = None

    @property
    def dipole(self):
        """Return smoothed dipoles when available, otherwise filtered dipoles."""
        return self.smoothed if self.smoothed is not None else self.filtered
