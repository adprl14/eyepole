"""Kalman filtering and smoothing for eye-dipole estimation."""

import numpy as np

from .results import DipoleFilterResult


def _rts_smoother(filtered, filtered_covariance, predicted, predicted_covariance, A):
    """Run a Rauch-Tung-Striebel backward smoothing pass."""
    smoothed = filtered.copy()
    smoothed_covariance = filtered_covariance.copy()

    # Work backward so that each estimate can borrow information from the
    # already-smoothed estimate at the next time point.
    for k in range(len(filtered) - 2, -1, -1):
        smoothing_gain = np.linalg.solve(
            predicted_covariance[k + 1].T,
            (filtered_covariance[k] @ A.T).T,
        ).T

        smoothed[k] = filtered[k] + smoothing_gain @ (
            smoothed[k + 1] - predicted[k + 1]
        )

        smoothed_covariance[k] = filtered_covariance[k] + smoothing_gain @ (
            smoothed_covariance[k + 1] - predicted_covariance[k + 1]
        ) @ smoothing_gain.T

    return smoothed, smoothed_covariance


class KalmanDipoleFilter:
    """Kalman filter for a linear, physics-derived EOG observation model."""

    def __init__(self, model, dynamics, R, x0=None, P0=None):
        if not hasattr(model, "C"):
            raise TypeError("linear Kalman filter requires a model with matrix C")

        self.model = model
        self.dynamics = dynamics
        self.A = dynamics.A
        self.Q = dynamics.Q
        self.C = model.C
        self.R = np.asarray(R, dtype=float)

        n_channels = self.C.shape[0]
        if self.R.shape != (n_channels, n_channels):
            raise ValueError(
                f"R must have shape ({n_channels}, {n_channels})"
            )

        self.x0 = np.zeros(3) if x0 is None else np.asarray(x0, dtype=float)
        self.P0 = np.eye(3) if P0 is None else np.asarray(P0, dtype=float)

    def fit(self, Y, smooth=True):
        """Estimate the dipole trajectory from an EOG recording."""
        Y = np.asarray(Y, dtype=float)
        n_samples = len(Y)
        identity = np.eye(3)

        filtered = np.zeros((n_samples, 3))
        filtered_covariance = np.zeros((n_samples, 3, 3))
        predicted = np.zeros((n_samples, 3))
        predicted_covariance = np.zeros((n_samples, 3, 3))

        x = self.x0.copy()
        P = self.P0.copy()

        for k in range(n_samples):
            # First predict where the dipole should be before looking at the
            # current EOG sample.
            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.Q

            # The innovation is the part of the measured EOG that our current
            # dipole prediction did not explain.
            innovation = Y[k] - self.C @ x_pred
            innovation_covariance = self.C @ P_pred @ self.C.T + self.R

            kalman_gain = np.linalg.solve(
                innovation_covariance.T,
                (P_pred @ self.C.T).T,
            ).T

            x = x_pred + kalman_gain @ innovation

            # Joseph form is a little more verbose than P=(I-KC)P, but it is
            # numerically safer and makes the role of measurement noise clear.
            residual_map = identity - kalman_gain @ self.C
            P = (
                residual_map @ P_pred @ residual_map.T
                + kalman_gain @ self.R @ kalman_gain.T
            )

            predicted[k] = x_pred
            predicted_covariance[k] = P_pred
            filtered[k] = x
            filtered_covariance[k] = P

        result = DipoleFilterResult(
            filtered=filtered,
            filtered_covariance=filtered_covariance,
            predicted=predicted,
            predicted_covariance=predicted_covariance,
        )

        if smooth:
            result.smoothed, result.smoothed_covariance = _rts_smoother(
                filtered,
                filtered_covariance,
                predicted,
                predicted_covariance,
                self.A,
            )

        return result


class ExtendedKalmanDipoleFilter:
    """Extended Kalman filter for the nonlinear finite two-pole model."""

    def __init__(self, model, dynamics, R, x0=None, P0=None):
        self.model = model
        self.dynamics = dynamics
        self.A = dynamics.A
        self.Q = dynamics.Q
        self.R = np.asarray(R, dtype=float)

        n_channels = model.montage.n_channels
        if self.R.shape != (n_channels, n_channels):
            raise ValueError(
                f"R must have shape ({n_channels}, {n_channels})"
            )

        self.x0 = np.zeros(3) if x0 is None else np.asarray(x0, dtype=float)
        self.P0 = np.eye(3) if P0 is None else np.asarray(P0, dtype=float)

    def fit(self, Y, smooth=True):
        """Estimate the dipole trajectory from EOG using an EKF."""
        Y = np.asarray(Y, dtype=float)
        n_samples = len(Y)
        identity = np.eye(3)

        filtered = np.zeros((n_samples, 3))
        filtered_covariance = np.zeros((n_samples, 3, 3))
        predicted = np.zeros((n_samples, 3))
        predicted_covariance = np.zeros((n_samples, 3, 3))

        x = self.x0.copy()
        P = self.P0.copy()

        for k in range(n_samples):
            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.Q

            # Near-field geometry is nonlinear. Around the current predicted
            # dipole, the Jacobian is the local linear observation matrix.
            C_local = self.model.jacobian(x_pred)
            predicted_eog = self.model.observe(x_pred)
            innovation = Y[k] - predicted_eog

            innovation_covariance = (
                C_local @ P_pred @ C_local.T + self.R
            )
            kalman_gain = np.linalg.solve(
                innovation_covariance.T,
                (P_pred @ C_local.T).T,
            ).T

            x = x_pred + kalman_gain @ innovation
            residual_map = identity - kalman_gain @ C_local
            P = (
                residual_map @ P_pred @ residual_map.T
                + kalman_gain @ self.R @ kalman_gain.T
            )

            predicted[k] = x_pred
            predicted_covariance[k] = P_pred
            filtered[k] = x
            filtered_covariance[k] = P

        result = DipoleFilterResult(
            filtered=filtered,
            filtered_covariance=filtered_covariance,
            predicted=predicted,
            predicted_covariance=predicted_covariance,
        )

        if smooth:
            result.smoothed, result.smoothed_covariance = _rts_smoother(
                filtered,
                filtered_covariance,
                predicted,
                predicted_covariance,
                self.A,
            )

        return result
