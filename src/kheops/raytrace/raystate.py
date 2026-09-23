from dataclasses import dataclass

import numpy as np


@dataclass
class RayState:
    r: float
    x: np.ndarray
    M: np.ndarray
    travel_time: np.ndarray

    def ray(self, index):
        return RayState(self.r, self.x[index:index+1].copy(),
                        self.M[index:index+1].copy(), self.travel_time[index:index+1].copy())

