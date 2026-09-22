"""Minimal example of the two EyePole 2-D estimators."""

import numpy as np

from eyepole import EyePole2D


# Positions are deliberately simple here: four electrodes surrounding the eye.
electrodes_mm = {
    "R": [30.0, 0.0, 0.0],
    "L": [-30.0, 0.0, 0.0],
    "U": [0.0, 30.0, 0.0],
    "D": [0.0, -30.0, 0.0],
}

model = EyePole2D(
    electrode_positions_mm=electrodes_mm,
    channels=[("R", "L"), ("U", "D")],
    field_model="near",
    pole_separation_mm=20.0,
    gain=1.0,
)

# In real use, eog would be your measured array with shape
# (n_samples, 2), ordered as [R-L, U-D].
eog = np.zeros((100, 2))

# ---------------------------------------------------------------------------
# 1) Unconstrained 2-D estimate.
# ---------------------------------------------------------------------------
# This estimates p_x and p_y directly. There is no assumption about p_z or the
# total three-dimensional magnitude.
free_result = model.estimate_unconstrained(
    eog,
    process_noise=1e-4,
    measurement_noise=1e-3,
    smooth=True,
)
free_xy = free_result.dipole_xy

# ---------------------------------------------------------------------------
# 2) Constant-magnitude estimate.
# ---------------------------------------------------------------------------
# The state is still only [p_x, p_y], but the forward component is reconstructed
# from ||p|| = 1 and p_z > 0 before the near-field physics are evaluated.
constrained_result = model.estimate_constant_magnitude(
    eog,
    dipole_magnitude=1.0,
    process_noise=1e-4,
    measurement_noise=1e-3,
    smooth=True,
)
constrained_xy = constrained_result.dipole_xy
constrained_xyz = constrained_result.dipole_xyz()

print("free XY shape:", free_xy.shape)
print("constrained XY shape:", constrained_xy.shape)
print("reconstructed XYZ shape:", constrained_xyz.shape)
