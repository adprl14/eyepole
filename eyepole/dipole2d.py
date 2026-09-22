"""Readable two-dimensional eye-dipole estimators.

This module provides two related ways to estimate eye-dipole motion when the
recording montage mainly observes horizontal and vertical EOG (for example,
R-L and U-D) and does not provide an independent front-back channel.

The main user-facing method is ``estimate``. By default,
``estimate(..., constrained=False)`` estimates ``[p_x, p_y]`` directly with no
assumption about the total three-dimensional dipole magnitude. Setting
``constrained=True`` still estimates only ``[p_x, p_y]``, but reconstructs the
forward component from a known total magnitude ``p0``:

       p_z = +sqrt(p0**2 - p_x**2 - p_y**2)

   The positive root encodes the prior knowledge that the corneo-retinal dipole
   points forward. The reconstructed three-dimensional dipole is then passed
   through the selected physics model (finite near field by default).

The implementation favors explicit equations and readable intermediate
variables over compact or highly optimized code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import EyeGeometry
from .models import FarFieldDipole, NearFieldDipole
from .montage import Montage


@dataclass
class Dipole2DResult:
    """Filtering and smoothing results for a two-dimensional dipole state.

    Parameters
    ----------
    filtered : numpy.ndarray, shape (n_samples, 2)
        Causal estimates of ``[p_x, p_y]``. Each row uses EOG data only up to
        and including that sample.

    filtered_covariance : numpy.ndarray, shape (n_samples, 2, 2)
        Posterior covariance of the filtered two-dimensional state.

    predicted : numpy.ndarray, shape (n_samples, 2)
        One-step state predictions before the corresponding EOG sample is used.

    predicted_covariance : numpy.ndarray, shape (n_samples, 2, 2)
        Covariance of the one-step state predictions.

    smoothed : numpy.ndarray or None, shape (n_samples, 2)
        Rauch-Tung-Striebel smoothed estimates. These use the entire recording,
        including future samples. ``None`` when smoothing is disabled.

    smoothed_covariance : numpy.ndarray or None, shape (n_samples, 2, 2)
        Approximate covariance of the smoothed state.

    dipole_magnitude : float or None
        Constant three-dimensional dipole magnitude used by the constrained
        estimator. ``None`` for the unconstrained estimator.
    """

    filtered: np.ndarray
    filtered_covariance: np.ndarray
    predicted: np.ndarray
    predicted_covariance: np.ndarray
    smoothed: np.ndarray | None = None
    smoothed_covariance: np.ndarray | None = None
    dipole_magnitude: float | None = None

    @property
    def dipole_xy(self) -> np.ndarray:
        """Return smoothed ``[p_x, p_y]`` when available, otherwise filtered."""
        return self.smoothed if self.smoothed is not None else self.filtered

    @property
    def dipole(self) -> np.ndarray:
        """Alias for :attr:`dipole_xy` for consistency with other results."""
        return self.dipole_xy

    def dipole_xyz(self) -> np.ndarray:
        """Reconstruct the three-dimensional dipole for a constrained result.

        Returns
        -------
        dipole_xyz : numpy.ndarray, shape (n_samples, 3)
            Three-dimensional dipole vectors with a positive forward component.

        Raises
        ------
        ValueError
            If this result came from the unconstrained estimator. An
            unconstrained two-dimensional state does not contain enough
            information to infer ``p_z`` without adding another assumption.
        """
        if self.dipole_magnitude is None:
            raise ValueError(
                "unconstrained 2-D estimates do not define p_z; "
                "use dipole_xy instead"
            )

        return xy_to_constant_magnitude_xyz(
            self.dipole_xy,
            magnitude=self.dipole_magnitude,
        )


class EyePole2D:
    """Estimate horizontal and vertical eye-dipole components from EOG.

    This class is intended for U/D/L/R-style EOG montages where horizontal and
    vertical eye motion are observable but there is no independent front-back
    measurement.

    Parameters
    ----------
    electrode_positions_mm : mapping[str, sequence[float]]
        Electrode positions relative to the eye center, in millimeters. Each
        value must contain ``[x, y, z]``.

    channels : sequence[tuple[str, str]]
        Bipolar EOG channels. A pair ``("R", "L")`` means right-electrode
        potential minus left-electrode potential.

    pole_separation_mm : float, default=20.0
        Distance between the positive and negative poles of the finite dipole,
        in millimeters. This affects the constrained near-field estimator.

    field_model : {"near", "far"}, default="near"
        Physics model used by :meth:`estimate_constant_magnitude`.

        ``"near"`` keeps the positive and negative poles at separate physical
        locations and is the default.

        ``"far"`` uses the linear point-dipole approximation.

        The unconstrained estimator is intentionally always based on the
        two-dimensional far-field model because, without a value for ``p_z``,
        the finite three-dimensional pole geometry is not fully specified.

    gain : float, default=1.0
        Overall multiplicative scale of the forward model. In a fully physical
        model this would contain conductivity/unit factors. In practice it can
        also absorb unknown measurement scaling.

    Notes
    -----
    The default temporal model is a two-dimensional random walk,

        x[t] = x[t-1] + w[t]

    where ``x[t] = [p_x[t], p_y[t]]``.

    The constrained estimator adds the geometric relation

        p_z[t] = +sqrt(p0**2 - p_x[t]**2 - p_y[t]**2)

    before evaluating the physical EOG forward model.
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
        if pole_separation_mm <= 0:
            raise ValueError("pole_separation_mm must be positive")

        self.geometry = EyeGeometry.from_mm(electrode_positions_mm)
        self.montage = Montage.from_pairs(self.geometry, channels)
        self.pole_separation_mm = float(pole_separation_mm)
        self.field_model = field_model
        self.gain = float(gain)

        # We always construct the far-field model because it is useful for the
        # simple unconstrained 2-D estimator and for initialization.
        self.far_model = FarFieldDipole(
            geometry=self.geometry,
            montage=self.montage,
            gain=self.gain,
        )

        if self.field_model == "near":
            self.forward_model = NearFieldDipole(
                geometry=self.geometry,
                montage=self.montage,
                source_separation=self.pole_separation_mm * 1e-3,
                gain=self.gain,
            )
        else:
            self.forward_model = self.far_model

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Recorded channel names in the order expected by estimation methods."""
        return self.montage.channel_names

    @property
    def C_xy(self) -> np.ndarray:
        """Physics-derived linear observation matrix for ``[p_x, p_y]``.

        Returns
        -------
        C_xy : numpy.ndarray, shape (n_channels, 2)
            First two columns of the three-dimensional far-field lead field.
            The unconstrained model is

                y = C_xy @ [p_x, p_y].
        """
        return self.far_model.C[:, :2]

    def estimate(
        self,
        eog: np.ndarray,
        constrained: bool = False,
        dipole_magnitude: float = 1.0,
        process_noise: float | np.ndarray = 1e-4,
        measurement_noise: float | np.ndarray = 1e-3,
        smooth: bool = True,
        initial_xy: np.ndarray | None = None,
        initial_covariance: float | np.ndarray | None = None,
        jacobian_step: float = 1e-5,
    ) -> Dipole2DResult:
        """Estimate the eye dipole from horizontal/vertical EOG.

        This is the main user-facing estimation method. By default it estimates
        an unconstrained two-dimensional dipole ``[p_x, p_y]``. Set
        ``constrained=True`` to enforce a constant three-dimensional dipole
        magnitude while still estimating only the two observable components.

        Parameters
        ----------
        eog : numpy.ndarray, shape (n_samples, n_channels)
            EOG time series. Columns must have the same order as
            :attr:`channel_names`.

        constrained : bool, default=False
            Select which state model to use.

            If ``False``, estimate ``[p_x, p_y]`` directly with the linear
            physics-derived far-field observation model. No value of ``p_z``
            or total 3-D dipole magnitude is assumed.

            If ``True``, estimate ``[p_x, p_y]`` while enforcing

                p_x**2 + p_y**2 + p_z**2 = dipole_magnitude**2

            with a positive forward component

                p_z = +sqrt(dipole_magnitude**2 - p_x**2 - p_y**2).

            The reconstructed 3-D dipole is then evaluated with the selected
            physics model (near field by default).

        dipole_magnitude : float, default=1.0
            Constant magnitude of the full 3-D dipole when
            ``constrained=True``. This parameter is ignored when
            ``constrained=False``.

            If the main goal is eye direction rather than an absolute dipole
            moment in physical units, ``1.0`` is a convenient normalization.

        process_noise : float or numpy.ndarray, default=1e-4
            Random-walk process covariance ``Q`` for ``[p_x, p_y]``. A scalar
            means ``Q = process_noise * I``. This is a variance.

        measurement_noise : float or numpy.ndarray, default=1e-3
            EOG measurement covariance ``R``. A scalar means equal independent
            channel variance.

        smooth : bool, default=True
            If ``True``, run an RTS backward smoother after the forward filter.

        initial_xy : numpy.ndarray or None, shape (2,), default=None
            Optional initial value of ``[p_x, p_y]``. If omitted, the first EOG
            sample is inverted with the linear far-field model.

        initial_covariance : float or numpy.ndarray or None, default=None
            Initial state covariance. ``None`` uses the 2 x 2 identity matrix.

        jacobian_step : float, default=1e-5
            Relative finite-difference step used by the constrained nonlinear
            estimator. It is ignored when ``constrained=False``.

        Returns
        -------
        result : Dipole2DResult
            Filtered and optionally smoothed dipole estimates.

            ``result.dipole_xy`` is always available. If
            ``constrained=True``, ``result.dipole_xyz()`` also returns the full
            constant-magnitude 3-D dipole with positive ``p_z``.

        Examples
        --------
        The default is the simple unconstrained 2-D model::

            result = model.estimate(eog)
            p_xy = result.dipole_xy

        To enforce constant magnitude::

            result = model.estimate(
                eog,
                constrained=True,
                dipole_magnitude=1.0,
            )
            p_xyz = result.dipole_xyz()
        """
        if constrained:
            return self.estimate_constant_magnitude(
                eog=eog,
                dipole_magnitude=dipole_magnitude,
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                smooth=smooth,
                initial_xy=initial_xy,
                initial_covariance=initial_covariance,
                jacobian_step=jacobian_step,
            )

        return self.estimate_unconstrained(
            eog=eog,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            smooth=smooth,
            initial_xy=initial_xy,
            initial_covariance=initial_covariance,
        )

    def estimate_unconstrained(
        self,
        eog: np.ndarray,
        process_noise: float | np.ndarray = 1e-4,
        measurement_noise: float | np.ndarray = 1e-3,
        smooth: bool = True,
        initial_xy: np.ndarray | None = None,
        initial_covariance: float | np.ndarray | None = None,
    ) -> Dipole2DResult:
        """Estimate an unconstrained two-dimensional dipole ``[p_x, p_y]``.

        This is the simplest model for a U/D/L/R montage. It does **not** assume
        a fixed three-dimensional dipole magnitude and it does not attempt to
        infer ``p_z``.

        The observation model is the physics-derived point-dipole relation

            y[t] = C_xy @ x[t] + measurement noise

        with

            x[t] = [p_x[t], p_y[t]].

        Parameters
        ----------
        eog : numpy.ndarray, shape (n_samples, n_channels)
            EOG time series. Column order must match :attr:`channel_names`.

        process_noise : float or numpy.ndarray, default=1e-4
            Random-walk process covariance ``Q``. A scalar means
            ``Q = process_noise * I``. This is a variance, not a standard
            deviation. Larger values allow faster sample-to-sample changes.

        measurement_noise : float or numpy.ndarray, default=1e-3
            EOG measurement covariance ``R``. A scalar means equal independent
            channel variance: ``R = measurement_noise * I``.

        smooth : bool, default=True
            If ``True``, run a Rauch-Tung-Striebel backward smoother after the
            forward Kalman filter.

        initial_xy : numpy.ndarray or None, shape (2,), default=None
            Initial ``[p_x, p_y]`` estimate. If omitted, the first EOG sample is
            inverted by least squares using ``C_xy``.

        initial_covariance : float or numpy.ndarray or None, default=None
            Initial state covariance ``P0``. ``None`` uses the 2x2 identity.

        Returns
        -------
        result : Dipole2DResult
            Filtered and optionally smoothed two-dimensional dipole estimates.
        """
        eog = self._validate_eog(eog)
        Q = _covariance_matrix(process_noise, size=2, name="process_noise")
        R = _covariance_matrix(
            measurement_noise,
            size=self.montage.n_channels,
            name="measurement_noise",
        )
        P0 = _initial_covariance(initial_covariance)

        if initial_xy is None:
            x0, *_ = np.linalg.lstsq(self.C_xy, eog[0], rcond=None)
        else:
            x0 = _validate_xy(initial_xy, name="initial_xy")

        return _linear_kalman_filter_2d(
            eog=eog,
            C=self.C_xy,
            Q=Q,
            R=R,
            x0=x0,
            P0=P0,
            smooth=smooth,
        )

    def estimate_constant_magnitude(
        self,
        eog: np.ndarray,
        dipole_magnitude: float = 1.0,
        process_noise: float | np.ndarray = 1e-4,
        measurement_noise: float | np.ndarray = 1e-3,
        smooth: bool = True,
        initial_xy: np.ndarray | None = None,
        initial_covariance: float | np.ndarray | None = None,
        jacobian_step: float = 1e-5,
    ) -> Dipole2DResult:
        """Estimate ``[p_x, p_y]`` with a constant 3-D dipole magnitude.

        The free state is only

            x[t] = [p_x[t], p_y[t]].

        Before predicting EOG, the forward component is reconstructed as

            p_z[t] = +sqrt(p0**2 - p_x[t]**2 - p_y[t]**2),

        where ``p0`` is ``dipole_magnitude``. The positive square root encodes
        the known fact that the eye dipole points forward.

        The resulting 3-D vector is passed through either the finite near-field
        model or the far-field model selected when this object was created.
        Because the mapping from ``[p_x, p_y]`` to EOG is nonlinear, estimation
        uses an extended Kalman filter followed optionally by an RTS smoother.

        Parameters
        ----------
        eog : numpy.ndarray, shape (n_samples, n_channels)
            EOG time series. Column order must match :attr:`channel_names`.

        dipole_magnitude : float, default=1.0
            Assumed constant magnitude ``p0 = ||p||`` of the three-dimensional
            dipole vector.

            If the primary goal is eye *direction*, ``1.0`` is a useful
            normalized choice and ``gain`` can absorb the unknown voltage
            scale. Absolute physical dipole magnitude is only meaningful when
            the conductivity/unit scaling in ``gain`` is independently known.

        process_noise : float or numpy.ndarray, default=1e-4
            Random-walk process covariance ``Q`` for ``[p_x, p_y]``. This is a
            variance. A scalar creates ``Q = process_noise * I``.

        measurement_noise : float or numpy.ndarray, default=1e-3
            EOG measurement covariance ``R``. A scalar creates equal independent
            channel variance.

        smooth : bool, default=True
            If ``True``, run an RTS backward smoother after the EKF.

        initial_xy : numpy.ndarray or None, shape (2,), default=None
            Initial horizontal/vertical dipole components. If omitted, the
            first EOG sample is inverted using the linear far-field model and
            projected inside the physically allowed disk.

        initial_covariance : float or numpy.ndarray or None, default=None
            Initial covariance ``P0`` for the two-dimensional state.

        jacobian_step : float, default=1e-5
            Relative finite-difference step used to calculate the EOG Jacobian
            with respect to ``p_x`` and ``p_y``. The actual step scales with
            ``dipole_magnitude`` so normalized and small physical magnitudes can
            both be represented.

        Returns
        -------
        result : Dipole2DResult
            Filtered and optionally smoothed estimates. Use
            ``result.dipole_xy`` for the two estimated components or
            ``result.dipole_xyz()`` to reconstruct the full constant-magnitude
            vector with positive ``p_z``.

        Notes
        -----
        The EKF update can occasionally propose a state just outside the disk

            p_x**2 + p_y**2 <= p0**2.

        Such states are projected back just inside the disk. Normal ocular
        rotations occupy only a modest fraction of this hemisphere, so this
        boundary handling should rarely be active in ordinary use.
        """
        eog = self._validate_eog(eog)

        dipole_magnitude = float(dipole_magnitude)
        if dipole_magnitude <= 0:
            raise ValueError("dipole_magnitude must be positive")
        if jacobian_step <= 0:
            raise ValueError("jacobian_step must be positive")

        Q = _covariance_matrix(process_noise, size=2, name="process_noise")
        R = _covariance_matrix(
            measurement_noise,
            size=self.montage.n_channels,
            name="measurement_noise",
        )
        P0 = _initial_covariance(initial_covariance)

        if initial_xy is None:
            initial_xy, *_ = np.linalg.lstsq(self.C_xy, eog[0], rcond=None)
        else:
            initial_xy = _validate_xy(initial_xy, name="initial_xy")

        initial_xy = _project_inside_magnitude_disk(
            initial_xy,
            magnitude=dipole_magnitude,
        )

        def observation(xy: np.ndarray) -> np.ndarray:
            """Map the 2-D free state to EOG through the 3-D physics model."""
            dipole_xyz = xy_to_constant_magnitude_xyz(
                xy,
                magnitude=dipole_magnitude,
            )
            return self.forward_model.observe(dipole_xyz)

        def observation_jacobian(xy: np.ndarray) -> np.ndarray:
            """Numerically differentiate EOG with respect to p_x and p_y."""
            return _finite_difference_jacobian_2d(
                function=observation,
                xy=xy,
                relative_step=jacobian_step,
                state_scale=dipole_magnitude,
            )

        result = _extended_kalman_filter_2d(
            eog=eog,
            observation=observation,
            observation_jacobian=observation_jacobian,
            Q=Q,
            R=R,
            x0=initial_xy,
            P0=P0,
            smooth=smooth,
            maximum_magnitude=dipole_magnitude,
        )
        result.dipole_magnitude = dipole_magnitude
        return result

    def forward_constant_magnitude(
        self,
        dipole_xy: np.ndarray,
        dipole_magnitude: float = 1.0,
    ) -> np.ndarray:
        """Predict EOG from 2-D components under the constant-magnitude model.

        Parameters
        ----------
        dipole_xy : numpy.ndarray, shape (2,) or (n_samples, 2)
            Horizontal and vertical dipole components.

        dipole_magnitude : float, default=1.0
            Constant three-dimensional dipole magnitude.

        Returns
        -------
        eog : numpy.ndarray
            Predicted EOG channels. Shape is ``(n_channels,)`` for one state or
            ``(n_samples, n_channels)`` for a time series.
        """
        dipole_xyz = xy_to_constant_magnitude_xyz(
            dipole_xy,
            magnitude=dipole_magnitude,
        )

        if dipole_xyz.ndim == 1:
            return self.forward_model.observe(dipole_xyz)

        return np.asarray([self.forward_model.observe(p) for p in dipole_xyz])

    def _validate_eog(self, eog: np.ndarray) -> np.ndarray:
        """Check EOG shape and convert it to a floating-point array."""
        eog = np.asarray(eog, dtype=float)
        if eog.ndim != 2:
            raise ValueError("eog must have shape (n_samples, n_channels)")
        if eog.shape[1] != self.montage.n_channels:
            raise ValueError(
                f"eog has {eog.shape[1]} channels, but this model expects "
                f"{self.montage.n_channels}: {self.channel_names}"
            )
        if len(eog) == 0:
            raise ValueError("eog must contain at least one sample")
        return eog


def xy_to_constant_magnitude_xyz(
    dipole_xy: np.ndarray,
    magnitude: float = 1.0,
) -> np.ndarray:
    """Lift 2-D dipole components onto the forward constant-magnitude hemisphere.

    Parameters
    ----------
    dipole_xy : numpy.ndarray, shape (2,) or (n_samples, 2)
        Horizontal and vertical dipole components ``[p_x, p_y]``.

    magnitude : float, default=1.0
        Desired three-dimensional magnitude ``||p||``.

    Returns
    -------
    dipole_xyz : numpy.ndarray, shape (3,) or (n_samples, 3)
        Three-dimensional dipole vector(s) satisfying ``||p|| = magnitude`` and
        ``p_z >= 0``.

    Raises
    ------
    ValueError
        If any supplied ``[p_x, p_y]`` has magnitude larger than ``magnitude``.
    """
    magnitude = float(magnitude)
    if magnitude <= 0:
        raise ValueError("magnitude must be positive")

    dipole_xy = np.asarray(dipole_xy, dtype=float)
    single_vector = dipole_xy.ndim == 1

    if single_vector:
        if dipole_xy.shape != (2,):
            raise ValueError("a single dipole_xy vector must have shape (2,)")
        xy = dipole_xy[None, :]
    elif dipole_xy.ndim == 2 and dipole_xy.shape[1] == 2:
        xy = dipole_xy
    else:
        raise ValueError("dipole_xy must have shape (2,) or (n_samples, 2)")

    squared_xy_magnitude = np.sum(xy**2, axis=1)
    maximum_squared = magnitude**2

    # A tiny numerical tolerance prevents round-off at the exact boundary from
    # producing a negative value inside the square root.
    tolerance = 1e-12 * max(1.0, maximum_squared)
    if np.any(squared_xy_magnitude > maximum_squared + tolerance):
        raise ValueError(
            "p_x**2 + p_y**2 cannot exceed the squared dipole magnitude"
        )

    pz_squared = np.maximum(maximum_squared - squared_xy_magnitude, 0.0)
    pz = np.sqrt(pz_squared)
    xyz = np.column_stack([xy, pz])

    return xyz[0] if single_vector else xyz


def _linear_kalman_filter_2d(
    eog: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    smooth: bool,
) -> Dipole2DResult:
    """Run a readable two-dimensional random-walk Kalman filter."""
    n_samples = len(eog)
    identity = np.eye(2)

    filtered = np.zeros((n_samples, 2))
    filtered_covariance = np.zeros((n_samples, 2, 2))
    predicted = np.zeros((n_samples, 2))
    predicted_covariance = np.zeros((n_samples, 2, 2))

    x = np.asarray(x0, dtype=float).copy()
    P = np.asarray(P0, dtype=float).copy()

    for sample in range(n_samples):
        # Random-walk prediction: the best prediction of the next state is the
        # current state, but uncertainty grows because the eye can move.
        x_predicted = x.copy()
        P_predicted = P + Q

        # The innovation is the part of the measured EOG not explained by the
        # EOG predicted from our current dipole estimate.
        innovation = eog[sample] - C @ x_predicted
        innovation_covariance = C @ P_predicted @ C.T + R

        kalman_gain = np.linalg.solve(
            innovation_covariance.T,
            (P_predicted @ C.T).T,
        ).T

        x = x_predicted + kalman_gain @ innovation

        # Joseph form is slightly longer than P=(I-KC)P, but it is numerically
        # safer and keeps the covariance symmetric/positive more reliably.
        correction = identity - kalman_gain @ C
        P = (
            correction @ P_predicted @ correction.T
            + kalman_gain @ R @ kalman_gain.T
        )

        predicted[sample] = x_predicted
        predicted_covariance[sample] = P_predicted
        filtered[sample] = x
        filtered_covariance[sample] = P

    result = Dipole2DResult(
        filtered=filtered,
        filtered_covariance=filtered_covariance,
        predicted=predicted,
        predicted_covariance=predicted_covariance,
    )

    if smooth:
        result.smoothed, result.smoothed_covariance = _rts_smoother_2d(
            filtered=filtered,
            filtered_covariance=filtered_covariance,
            predicted=predicted,
            predicted_covariance=predicted_covariance,
            Q=Q,
        )

    return result


def _extended_kalman_filter_2d(
    eog: np.ndarray,
    observation,
    observation_jacobian,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    smooth: bool,
    maximum_magnitude: float,
) -> Dipole2DResult:
    """Run a two-dimensional EKF for the constant-magnitude observation model."""
    n_samples = len(eog)
    identity = np.eye(2)

    filtered = np.zeros((n_samples, 2))
    filtered_covariance = np.zeros((n_samples, 2, 2))
    predicted = np.zeros((n_samples, 2))
    predicted_covariance = np.zeros((n_samples, 2, 2))

    x = np.asarray(x0, dtype=float).copy()
    P = np.asarray(P0, dtype=float).copy()

    for sample in range(n_samples):
        # The two free components follow a random walk. Projection is only a
        # safety measure: ordinary eye rotations should remain well inside the
        # constant-magnitude disk rather than repeatedly hitting its boundary.
        x_predicted = _project_inside_magnitude_disk(
            x,
            magnitude=maximum_magnitude,
        )
        P_predicted = P + Q

        predicted_eog = observation(x_predicted)
        H = observation_jacobian(x_predicted)

        innovation = eog[sample] - predicted_eog
        innovation_covariance = H @ P_predicted @ H.T + R

        kalman_gain = np.linalg.solve(
            innovation_covariance.T,
            (P_predicted @ H.T).T,
        ).T

        x = x_predicted + kalman_gain @ innovation
        x = _project_inside_magnitude_disk(
            x,
            magnitude=maximum_magnitude,
        )

        correction = identity - kalman_gain @ H
        P = (
            correction @ P_predicted @ correction.T
            + kalman_gain @ R @ kalman_gain.T
        )

        predicted[sample] = x_predicted
        predicted_covariance[sample] = P_predicted
        filtered[sample] = x
        filtered_covariance[sample] = P

    result = Dipole2DResult(
        filtered=filtered,
        filtered_covariance=filtered_covariance,
        predicted=predicted,
        predicted_covariance=predicted_covariance,
        dipole_magnitude=maximum_magnitude,
    )

    if smooth:
        smoothed, smoothed_covariance = _rts_smoother_2d(
            filtered=filtered,
            filtered_covariance=filtered_covariance,
            predicted=predicted,
            predicted_covariance=predicted_covariance,
            Q=Q,
        )

        # RTS smoothing is unconstrained algebraically, so a very small amount
        # of projection is retained here as a final guard against numerical
        # excursions beyond the physical hemisphere.
        smoothed = np.asarray(
            [
                _project_inside_magnitude_disk(xy, maximum_magnitude)
                for xy in smoothed
            ]
        )
        result.smoothed = smoothed
        result.smoothed_covariance = smoothed_covariance

    return result


def _rts_smoother_2d(
    filtered: np.ndarray,
    filtered_covariance: np.ndarray,
    predicted: np.ndarray,
    predicted_covariance: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an RTS backward smoother for a two-dimensional random walk."""
    del Q  # Q is already represented in predicted_covariance; kept for clarity.

    smoothed = filtered.copy()
    smoothed_covariance = filtered_covariance.copy()

    for sample in range(len(filtered) - 2, -1, -1):
        # For the random walk A=I, so the usual smoother gain
        # P_filt A^T P_pred^{-1} simplifies to P_filt P_pred^{-1}.
        smoother_gain = np.linalg.solve(
            predicted_covariance[sample + 1].T,
            filtered_covariance[sample].T,
        ).T

        smoothed[sample] = filtered[sample] + smoother_gain @ (
            smoothed[sample + 1] - predicted[sample + 1]
        )

        smoothed_covariance[sample] = filtered_covariance[sample] + (
            smoother_gain
            @ (
                smoothed_covariance[sample + 1]
                - predicted_covariance[sample + 1]
            )
            @ smoother_gain.T
        )

    return smoothed, smoothed_covariance


def _finite_difference_jacobian_2d(
    function,
    xy: np.ndarray,
    relative_step: float,
    state_scale: float,
) -> np.ndarray:
    """Calculate d function(xy) / d xy with a centered finite difference."""
    xy = _validate_xy(xy, name="xy")
    output_size = np.asarray(function(xy)).size
    jacobian = np.zeros((output_size, 2))

    for dimension in range(2):
        step = relative_step * max(
            abs(xy[dimension]),
            abs(state_scale),
            np.finfo(float).eps,
        )

        delta = np.zeros(2)
        delta[dimension] = step

        # Keep the two evaluation points inside the allowed hemisphere. This is
        # mostly relevant only if a state gets extremely close to the boundary.
        plus = _project_inside_magnitude_disk(xy + delta, state_scale)
        minus = _project_inside_magnitude_disk(xy - delta, state_scale)

        actual_step = plus[dimension] - minus[dimension]
        if abs(actual_step) < np.finfo(float).eps:
            raise RuntimeError("could not form a finite-difference Jacobian")

        jacobian[:, dimension] = (
            np.asarray(function(plus)) - np.asarray(function(minus))
        ) / actual_step

    return jacobian


def _project_inside_magnitude_disk(
    xy: np.ndarray,
    magnitude: float,
    margin: float = 1e-9,
) -> np.ndarray:
    """Project ``[p_x, p_y]`` just inside the disk allowed by fixed magnitude."""
    xy = _validate_xy(xy, name="xy").copy()
    radius = np.linalg.norm(xy)
    maximum_radius = float(magnitude) * (1.0 - margin)

    if radius > maximum_radius:
        xy *= maximum_radius / radius

    return xy


def _covariance_matrix(value, size: int, name: str) -> np.ndarray:
    """Convert a scalar variance or a full matrix into a covariance matrix."""
    if np.isscalar(value):
        variance = float(value)
        if variance < 0:
            raise ValueError(f"{name} must be non-negative")
        return variance * np.eye(size)

    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (size, size):
        raise ValueError(f"{name} must be scalar or shape ({size}, {size})")
    return covariance


def _initial_covariance(value) -> np.ndarray:
    """Construct the 2x2 initial state covariance P0."""
    if value is None:
        return np.eye(2)
    return _covariance_matrix(value, size=2, name="initial_covariance")


def _validate_xy(xy: np.ndarray, name: str) -> np.ndarray:
    """Return a floating-point 2-vector or raise a readable shape error."""
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    return xy
