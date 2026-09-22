"""Minimal EyePole example using the default near-field model."""

import numpy as np

from eyepole import EyePole


# Coordinates are relative to the eye center. These values are only a simple
# demonstration; a real analysis should use the geometry appropriate to the
# actual EOG electrode placement.
electrodes_mm = {
    "R": [30, 0, 0],
    "L": [-30, 0, 0],
    "U": [0, 30, 0],
    "D": [0, -30, 0],
    "F": [0, 0, 35],
    "B": [0, 0, -35],
}

model = EyePole(
    electrode_positions_mm=electrodes_mm,
    channels=[("R", "L"), ("U", "D"), ("F", "B")],
    pole_separation_mm=20.0,
)

# Make a small synthetic dipole trajectory and generate the EOG that the
# physical model predicts for it.
t = np.linspace(0, 2 * np.pi, 200)
true_dipole = np.column_stack(
    [
        0.5 * np.cos(t),
        0.5 * np.sin(t),
        0.1 * np.sin(2 * t),
    ]
)
eog = model.forward(true_dipole)

result = model.estimate(
    eog,
    process_noise=1e-3,
    measurement_noise=1e-4,
    smooth=True,
)

print("Expected channel order:", model.channel_names)
print("Estimated dipole shape:", result.dipole.shape)
