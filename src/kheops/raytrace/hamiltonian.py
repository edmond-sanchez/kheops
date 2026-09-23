import numpy as np

from kheops.environment import FlatBottomEnvironment


class FlatBottomRaySystem:
    def __init__(self, environment: FlatBottomEnvironment):
        self.environment = environment

    def hamiltonian(self, x):
        z, p = x[..., 0], x[..., 1]
        n = 1 / self.environment.ssp.c(z)
        if np.any(np.abs(p) >= n):
            raise ValueError(
                "Range marching requires positive horizontal slowness."
            )
        H = -np.sqrt(n**2 - p**2)
        return H

    def vector_field(self, x):
        z, p = x[..., 0], x[..., 1]
        ssp = self.environment.ssp
        n = 1 / ssp.c(z)
        n_z = -ssp.gradient(z) * n**2
        H = self.hamiltonian(x)

        dz_dr = -p / H
        dp_dr = -n * n_z / H
        return np.stack((dz_dr, dp_dr), axis=-1)

    def hessian(self, x):
        z, p = x[..., 0], x[..., 1]
        ssp = self.environment.ssp
        n = 1 / ssp.c(z)
        c_z = ssp.gradient(z)
        c_zz = ssp.curvature(z)
        n_z = -c_z * n**2
        n_zz = 2 * c_z**2 * n**3 - c_zz * n**2
        H = self.hamiltonian(x)

        H_zz = (n_z**2 + n * n_zz) / H - n**2 * n_z**2 / H**3
        H_zp = n * n_z * p / H**3
        H_pp = -(n**2) / H**3
        return np.stack(
            (np.stack((H_zz, H_zp), axis=-1), np.stack((H_zp, H_pp), axis=-1)),
            axis=-2,
        )

    def linearized_field(self, x):
        hessian = self.hessian(x)
        return np.stack((hessian[..., 1, :], -hessian[..., 0, :]), axis=-2)

    def time_rate(self, x):
        z = x[..., 0]
        n = 1 / self.environment.ssp.c(z)
        H = self.hamiltonian(x)
        return -(n**2) / H
