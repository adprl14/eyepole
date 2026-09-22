"""EyePole: readable physics-based eye-dipole estimation from EOG."""

from .core import EyePole
from .geometry import EyeGeometry
from .montage import Montage
from .models import FarFieldDipole, NearFieldDipole
from .dynamics import RandomWalkDipole
from .filters import KalmanDipoleFilter, ExtendedKalmanDipoleFilter
from .results import DipoleFilterResult
from .simulate import simulate_eog
from .dipole2d import EyePole2D, Dipole2DResult, xy_to_constant_magnitude_xyz

__all__ = [
    "EyePole",
    "EyeGeometry",
    "Montage",
    "FarFieldDipole",
    "NearFieldDipole",
    "RandomWalkDipole",
    "KalmanDipoleFilter",
    "ExtendedKalmanDipoleFilter",
    "DipoleFilterResult",
    "simulate_eog",
    "EyePole2D",
    "Dipole2DResult",
    "xy_to_constant_magnitude_xyz",
]

__version__ = "0.3.0"
