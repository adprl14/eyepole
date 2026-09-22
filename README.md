# EyePole

EyePole estimates a three-dimensional eye dipole from multichannel EOG using a
physics-based forward model plus Kalman filtering and optional RTS smoothing.
The implementation favors readable equations and readable code over compact or
highly optimized code.

## Simplest use

```python
import numpy as np
from eyepole import EyePole

model = EyePole(
    electrode_positions_mm={
        "R": [30, 0, 0],
        "L": [-30, 0, 0],
        "U": [0, 30, 0],
        "D": [0, -30, 0],
        "F": [0, 0, 35],
        "B": [0, 0, -35],
    },
    channels=[("R", "L"), ("U", "D"), ("F", "B")],
    pole_separation_mm=20.0,
)

# eog has shape (n_samples, 3) in the order ('R-L', 'U-D', 'F-B').
result = model.estimate(
    eog,
    process_noise=1e-4,
    measurement_noise=1e-3,
)

p = result.dipole  # shape (n_samples, 3)
```

The default is the finite **near-field two-pole model**. You do not need to
choose a model unless you explicitly want the far-field approximation.

## What the near-field model means

Let `p` be the 3-D eye dipole moment and `s` the distance between the two poles.
The model places the poles at

```text
r_plus  = +(s/2) * p / ||p||
r_minus = -(s/2) * p / ||p||
```

and gives each pole an effective strength

```text
q = ||p|| / s.
```

At electrode `i`, the potential is proportional to

```text
q * [1 / ||r_i - r_plus|| - 1 / ||r_i - r_minus||].
```

The two inverse-distance terms are important in the near field because an EOG
electrode can be meaningfully closer to one pole than the other.

### `pole_separation_mm`

`pole_separation_mm` is the distance between those two poles. It defaults to
20 mm and is a normal public parameter:

```python
model = EyePole(..., pole_separation_mm=16.0)
```

Changing this value changes the geometry of the finite dipole while preserving
its dipole moment. It is therefore useful for sensitivity analyses.

## Near field versus far field

The default near-field model is nonlinear. It should be written as

```text
y = h(p)
```

rather than `y = C @ p`. The extended Kalman filter uses the local Jacobian

```text
C(p) = dh(p) / dp
```

at each time point.

The far-field approximation is linear:

```text
y = C @ p
```

where `C` is derived entirely from electrode geometry and the montage. To use
it explicitly:

```python
model = EyePole(..., field_model="far")
print(model.C)
```

No empirical regression is performed in this version of EyePole.

For a fully observable 3-D dipole, the chosen electrode geometry and montage must
provide three independent spatial sensitivities. With only horizontal and
vertical EOG channels, the third dipole component is not independently
identifiable without additional assumptions.

## Temporal model

The default dipole dynamics are deliberately simple:

```text
p[t] = p[t-1] + w[t]
w[t] ~ Normal(0, Q)
```

`process_noise` controls `Q`. A larger value lets the dipole move more rapidly;
a smaller value makes the inferred trajectory smoother.

The EOG measurement covariance is `R`, supplied through `measurement_noise`.
A scalar creates the same independent variance for every EOG channel, while a
full covariance matrix can represent channel-specific and correlated noise.

## Filtering and smoothing

`model.estimate(..., smooth=True)` first runs a Kalman filter (far field) or
extended Kalman filter (near field), then runs a Rauch-Tung-Striebel backward
smoother.

- `result.filtered` uses only present and past EOG.
- `result.smoothed` uses the whole recording.
- `result.dipole` returns `smoothed` when available, otherwise `filtered`.

## Forward simulation

The same physical model can be used in the intuitive forward direction:

```python
p = np.array([1.0, 0.0, 0.0])
y = model.forward(p)
```

For a trajectory:

```python
y = model.forward(p_trajectory)
```

This is useful for understanding what the assumed eye geometry predicts before
using the inverse estimator.

## Advanced building blocks

The simple `EyePole` class is only a wrapper. The individual components remain
available for users who want to inspect or modify the model:

- `EyeGeometry`
- `Montage`
- `FarFieldDipole`
- `NearFieldDipole`
- `RandomWalkDipole`
- `KalmanDipoleFilter`
- `ExtendedKalmanDipoleFilter`
- `DipoleFilterResult`

The package is intentionally organized so that the equations can be followed
through the code without having to decode a highly abstract framework.

## Two-dimensional estimation with U/D/L/R electrodes

When only horizontal and vertical EOG are independently observed, EyePole can
estimate a two-dimensional state `[p_x, p_y]` directly:

```python
from eyepole import EyePole2D

model = EyePole2D(
    electrode_positions_mm={
        "R": [30, 0, 0],
        "L": [-30, 0, 0],
        "U": [0, 30, 0],
        "D": [0, -30, 0],
    },
    channels=[("R", "L"), ("U", "D")],
    field_model="near",
    pole_separation_mm=20.0,
)
```

### Unconstrained 2-D estimate

```python
result = model.estimate_unconstrained(eog)
p_xy = result.dipole_xy
```

This uses the two horizontal/vertical columns of the physics-derived far-field
lead field. It does not infer `p_z` and does not assume a constant 3-D dipole
magnitude.

### Constant-magnitude estimate

```python
result = model.estimate_constant_magnitude(
    eog,
    dipole_magnitude=1.0,
)

p_xy = result.dipole_xy
p_xyz = result.dipole_xyz()
```

The free state is still only `[p_x, p_y]`, but the forward component is
reconstructed using

```text
p_z = +sqrt(p0^2 - p_x^2 - p_y^2)
```

before evaluating the chosen physics model. The positive root represents the
known forward direction of the eye dipole.

For directional estimation, `dipole_magnitude=1.0` is usually the cleanest
normalization. Absolute dipole magnitude and the forward-model gain are
multiplicatively confounded unless conductivity and measurement scaling are
known independently.
