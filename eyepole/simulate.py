"""Small simulation helper for checking EyePole models."""

import numpy as np


def simulate_eog(model, dipoles, R=None, random_state=None):
    """Generate EOG samples from a sequence of dipole vectors.

    Parameters
    ----------
    model : FarFieldDipole or NearFieldDipole
        Physics-based forward model.
    dipoles : numpy.ndarray, shape (n_samples, 3)
        Dipole trajectory to simulate.
    R : numpy.ndarray or None, shape (n_channels, n_channels)
        Optional measurement-noise covariance. ``None`` returns noise-free EOG.
    random_state : int, numpy.random.Generator, or None
        Random seed or generator used when adding measurement noise.

    Returns
    -------
    eog : numpy.ndarray, shape (n_samples, n_channels)
        Simulated EOG recording.
    """
    dipoles = np.asarray(dipoles, dtype=float)
    eog = np.asarray([model.observe(p) for p in dipoles])

    if R is None:
        return eog

    R = np.asarray(R, dtype=float)
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    noise = rng.multivariate_normal(np.zeros(eog.shape[1]), R, size=len(eog))
    return eog + noise
