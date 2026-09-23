from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline


class SoundSpeedProfile(ABC):
    @abstractmethod
    def c(self, z: float | NDArray[np.floating]):
        pass

    @abstractmethod
    def gradient(self, z: float | NDArray[np.floating]):
        pass

    @abstractmethod
    def curvature(self, z: float | NDArray[np.floating]):
        pass


class SSPCubicSpline(SoundSpeedProfile):
    """Sound-speed profile interpolated with a cubic spline."""

    def __init__(self, depth_nodes: ArrayLike, speed_nodes: ArrayLike) -> None:
        self.ssp = CubicSpline(depth_nodes, speed_nodes)

    def c(self, z: float | NDArray[np.floating]):
        return self.ssp(z)

    def gradient(self, z: float | NDArray[np.floating]):
        return self.ssp(z, 1)

    def curvature(self, z: float | NDArray[np.floating]):
        return self.ssp(z, 2)


""" 
the three following examples are found in Porter & Bucker 
"Gaussian beam tracing for computing ocean acoustic fields"
doi: 10.1121/1.395269
"""


@dataclass(frozen=True)
class DownwardRefractingSSP(SoundSpeedProfile):
    # obviously i converted Porter cursed units into m/s
    c0: float = 1533.75228936  # m/s
    gamma: float = -1.2286762  # 1/s

    def _q(self, z: float | NDArray[np.floating]):
        q = 1.0 - 2.0 * self.gamma * np.asarray(z) / self.c0
        if np.any(q <= 0.0):
            raise ValueError("Profile requires 1 - 2*gamma*z/c0 > 0.")
        return q

    def c(self, z: float | NDArray[np.floating]):
        return self.c0 * self._q(z) ** (-0.5)

    def gradient(self, z: float | NDArray[np.floating]):
        return self.gamma * self._q(z) ** (-1.5)

    def curvature(self, z: float | NDArray[np.floating]):
        return 3.0 * self.gamma**2 / self.c0 * self._q(z) ** (-2.5)


@dataclass(frozen=True)
class MunkSSP(SoundSpeedProfile):
    # Depth in metres, speed in m/s.
    c0: float = 1500.0
    epsilon: float = 0.00737
    z_axis: float = 1300.0
    scale: float = 1300.0
    z_cutoff: float = 5000.0

    def _x(self, z: float | NDArray[np.floating]):
        z = np.asarray(z)
        return 2.0 * (np.minimum(z, self.z_cutoff) - self.z_axis) / self.scale

    def c(self, z: float | NDArray[np.floating]):
        x = self._x(z)
        return self.c0 * (1.0 + self.epsilon * (x + np.expm1(-x)))

    def gradient(self, z: float | NDArray[np.floating]):
        x = self._x(z)
        value = (
            self.c0 * self.epsilon * (2.0 / self.scale)
            * (-np.expm1(-x))
        )
        return np.where(np.asarray(z) < self.z_cutoff, value, 0.0)

    def curvature(self, z: float | NDArray[np.floating]):
        x = self._x(z)
        value = (
            self.c0 * self.epsilon * (2.0 / self.scale)**2
            * np.exp(-x)
        )
        return np.where(np.asarray(z) < self.z_cutoff, value, 0.0)