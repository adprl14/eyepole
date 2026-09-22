"""Simple public interface for physics-based eye-dipole estimation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .dynamics import RandomWalkDipole
from .filters import ExtendedKalmanDipoleFilter, KalmanDipoleFilter
from .geometry import EyeGeometry
from .models import FarFieldDipole, NearFieldDipole
from .montage import Montage


class EyePole:
    """Estimate a 3-D eye dipole from multichannel EOG using known geometry.

    ``EyePole`` is the simple interface intended for normal use. It builds a
    physics-derived forward model from electrode locations and then estimates a
    temporally smooth dipole trajectory with a Kalman filter/smoother.

    By default, EyePole uses the **finite two-pole near-field model**. The
    positive and negative poles are kept at separate positions, so each EOG
    electrode can be a different distance from the two poles. This is more
    faithful to nearby electrodes than treating the eye as an ideal point
    dipole.

    Parameters
    ----------
    electrode_positions_mm : mapping[str, sequence[float]]
        Mapping from electrode name to ``[x, y, z]`` position in millimeters,
        measured relative to the center of the eye.

        Example::

            {
                "R": [30, 0, 0],
                "L": [-30, 0, 0],
                "U": [0, 30, 0],
                "D": [0, -30, 0],
            }

    channels : sequence[tuple[str, str]]
        Bipolar EOG channels. Each tuple ``(positive, negative)`` means that
        the recorded channel is ``positive electrode - negative electrode``.

        Example::

            [("R", "L"), ("U", "D")]

    pole_separation_mm : float, default=20.0
        Distance between the positive and negative poles of the finite eye
        dipole, in millimeters. This parameter only affects the near-field
        model. It is intentionally exposed so the user can study how assumed
        eye-dipole geometry changes the inferred trajectory.

    field_model : {"near", "far"}, default="near"
        Physics model used to map eye dipole to EOG.

        ``"near"`` uses the finite two-pole model and an extended Kalman filter.
        It is the default.

        ``"far"`` uses the ideal point-dipole approximation and a standard
        linear Kalman filter. In this case the forward model has one fixed
        physics-derived observation matrix ``C``.

    gain : float, default=1.0
        Overall scale factor multiplying the predicted electrode potentials.
        This can absorb effective conductivity and unit conversions.

    Notes
    -----
    The state being estimated is a three-dimensional dipole vector ``p[t]``.
    The temporal model is a random walk by default:

        p[t] = p[t-1] + process noise

    For ``field_model="near"`` the EOG observation is nonlinear, so there is no
    single global matrix ``C``. Instead the forward model is ``y = h(p)`` and
    the extended Kalman filter evaluates the local Jacobian ``dh/dp`` at each
    time point.
    """

    def __init__(
        self,
        electrode_positions_mm: Mapping[str, Sequence[float]],
        channels: Sequence[tuple[str, str]],
        pole_separation_mm: float = 20.0,
        field_model: str = "near",
        gain: float = 1.0,
    ):
        if field_model not in {"near", "far"}:
            raise ValueError("field_model must be either 'near' or 'far'")

        self.geometry = EyeGeometry.from_mm(electrode_positions_mm)
        self.montage = Montage.from_pairs(self.geometry, channels)
        self.field_model = field_model
        self.pole_separation_mm = float(pole_separation_mm)
        self.gain = float(gain)

        if field_model == "near":
            self.model = NearFieldDipole(
                geometry=self.geometry,
                montage=self.montage,
                source_separation=self.pole_separation_mm * 1e-3,
                gain=self.gain,
            )
        else:
            self.model = FarFieldDipole(
                geometry=self.geometry,
                montage=self.montage,
                gain=self.gain,
            )

    @property
    def channel_names(self):
        """Tuple containing the recorded channel names in expected data order."""
        return self.montage.channel_names

    @property
    def C(self):
        """Return the fixed observation matrix for the far-field model.

        Raises
        ------
        AttributeError
            If the near-field model is being used. The finite two-pole model is
            nonlinear and therefore does not have one fixed global matrix C.
        """
        if self.field_model != "far":
            raise AttributeError(
                "the near-field model has no single fixed C; "
                "use local_jacobian(p) instead"
            )
        return self.model.C

    def local_jacobian(self, p):
        """Return the local EOG sensitivity matrix at dipole vector ``p``."""
        return self.model.jacobian(p)

    def forward(self, dipole):
        """Predict EOG channels from one or more eye dipole vectors.

        Parameters
        ----------
        dipole : numpy.ndarray, shape (3,) or (n_samples, 3)
            One dipole vector or a time series of dipole vectors.

        Returns
        -------
        eog : numpy.ndarray
            Shape ``(n_channels,)`` for one dipole or
            ``(n_samples, n_channels)`` for a dipole time series.
        """
        dipole = np.asarray(dipole, dtype=float)

        if dipole.ndim == 1:
            if dipole.shape != (3,):
                raise ValueError("a single dipole must have shape (3,)")
            return self.model.observe(dipole)

        if dipole.ndim == 2 and dipole.shape[1] == 3:
            return np.asarray([self.model.observe(p) for p in dipole])

        raise ValueError("dipole must have shape (3,) or (n_samples, 3)")

    def estimate(
        self,
        eog,
        process_noise=1e-4,
        measurement_noise=1e-3,
        smooth=True,
        initial_dipole=None,
        initial_covariance=None,
    ):
        """Estimate the eye-dipole trajectory from multichannel EOG.

        Parameters
        ----------
        eog : numpy.ndarray, shape (n_samples, n_channels)
            EOG time series. Columns must follow ``self.channel_names``.

        process_noise : float or numpy.ndarray, default=1e-4
            Process-noise covariance ``Q`` for the random-walk dipole model.
            A scalar means ``Q = process_noise * I``. Larger values allow the
            estimated dipole to change more rapidly between adjacent samples;
            smaller values enforce a smoother trajectory.

        measurement_noise : float or numpy.ndarray, default=1e-3
            EOG measurement-noise covariance ``R``. A scalar means independent
            channels with equal variance: ``R = measurement_noise * I``.
            If channel noise is correlated or differs by channel, pass the full
            ``(n_channels, n_channels)`` covariance matrix.

        smooth : bool, default=True
            If ``True``, run a Rauch-Tung-Striebel backward pass after filtering.
            The returned ``result.dipole`` then uses both past and future EOG
            samples. If ``False``, ``result.dipole`` contains causal filtered
            estimates only.

        initial_dipole : numpy.ndarray or None, shape (3,), default=None
            Initial dipole estimate. If omitted, EyePole obtains a simple
            physics-based initial guess by inverting the far-field model for the
            first EOG sample. This avoids starting the nonlinear near-field EKF
            exactly at the zero vector, where dipole direction is undefined.

        initial_covariance : float or numpy.ndarray or None, default=None
            Initial state covariance ``P0``. ``None`` uses the 3x3 identity
            matrix. A scalar creates ``initial_covariance * I``.

        Returns
        -------
        result : DipoleFilterResult
            Contains filtered, predicted, and optionally smoothed dipole
            trajectories together with their covariance matrices.
        """
        eog = np.asarray(eog, dtype=float)
        if eog.ndim != 2:
            raise ValueError("eog must have shape (n_samples, n_channels)")
        if eog.shape[1] != self.montage.n_channels:
            raise ValueError(
                f"eog has {eog.shape[1]} channels, but this EyePole model "
                f"expects {self.montage.n_channels}: {self.channel_names}"
            )
        if len(eog) == 0:
            raise ValueError("eog must contain at least one sample")

        dynamics = RandomWalkDipole(Q=process_noise)
        R = self._measurement_covariance(measurement_noise)
        P0 = self._initial_covariance(initial_covariance)

        if initial_dipole is None:
            x0 = self._physics_based_initial_guess(eog[0])
        else:
            x0 = np.asarray(initial_dipole, dtype=float)
            if x0.shape != (3,):
                raise ValueError("initial_dipole must have shape (3,)")

        if self.field_model == "near":
            filter_ = ExtendedKalmanDipoleFilter(
                model=self.model,
                dynamics=dynamics,
                R=R,
                x0=x0,
                P0=P0,
            )
        else:
            filter_ = KalmanDipoleFilter(
                model=self.model,
                dynamics=dynamics,
                R=R,
                x0=x0,
                P0=P0,
            )

        return filter_.fit(eog, smooth=smooth)

    def _physics_based_initial_guess(self, first_eog_sample):
        """Use the linear far-field model to get a reasonable starting state."""
        far_field = FarFieldDipole(
            geometry=self.geometry,
            montage=self.montage,
            gain=self.gain,
        )

        # Least squares is intentionally used here instead of a hand-written
        # inverse because EOG montages are often rectangular or rank deficient.
        initial, *_ = np.linalg.lstsq(
            far_field.C,
            np.asarray(first_eog_sample, dtype=float),
            rcond=None,
        )
        return initial

    def _measurement_covariance(self, measurement_noise):
        """Convert a scalar or matrix into an R covariance matrix."""
        if np.isscalar(measurement_noise):
            value = float(measurement_noise)
            if value < 0:
                raise ValueError("measurement_noise must be non-negative")
            return np.eye(self.montage.n_channels) * value

        R = np.asarray(measurement_noise, dtype=float)
        expected = (self.montage.n_channels, self.montage.n_channels)
        if R.shape != expected:
            raise ValueError(f"measurement_noise must have shape {expected}")
        return R

    @staticmethod
    def _initial_covariance(initial_covariance):
        """Convert the user-friendly P0 input into a 3x3 covariance matrix."""
        if initial_covariance is None:
            return np.eye(3)
        if np.isscalar(initial_covariance):
            value = float(initial_covariance)
            if value < 0:
                raise ValueError("initial_covariance must be non-negative")
            return np.eye(3) * value

        P0 = np.asarray(initial_covariance, dtype=float)
        if P0.shape != (3, 3):
            raise ValueError("initial_covariance must be scalar or shape (3, 3)")
        return P0
