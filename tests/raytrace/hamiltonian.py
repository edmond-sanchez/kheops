import numpy as np
from numpy.testing import assert_allclose

from kheops.environment import FlatBottomEnvironment
from kheops.environment.ssp import DownwardRefractingSSP, MunkSSP
from kheops.raytrace.hamiltonian import FlatBottomRaySystem
from kheops.utils.numerical import numerical_gradient_hessian

# Depth and slowness have very different scales: use separate increments.
STEPS = (0.5, 1e-7)  # metres, seconds/metre


def sample_states(ssp):
    z = np.array([100.0, 800.0, 1300.0, 3000.0, 4200.0])
    angles = np.array([-0.3, 0.2, 0.0, -0.2, 0.3])  # radians
    p = np.sin(angles) / ssp.c(z)
    return np.stack((z, p), axis=-1)




def test_hamilton_equations_against_numerical_gradient():
    for ssp in (DownwardRefractingSSP(), MunkSSP()):
        system = FlatBottomRaySystem(FlatBottomEnvironment(ssp, bottom_depth=4500))
        x = sample_states(ssp)
        gradient, _ = numerical_gradient_hessian(system.hamiltonian, x, STEPS)
        dz_dr, dp_dr = system.vector_field(x).T
        # Hamilton's equations: dz/dr = H_p and dp/dr = -H_z.
        assert_allclose(dz_dr, gradient[:, 1], rtol=1e-7, atol=1e-11)
        assert_allclose(dp_dr, -gradient[:, 0], rtol=1e-7, atol=1e-13)


def test_hessian_against_numerical_second_derivatives():
    for ssp in (DownwardRefractingSSP(), MunkSSP()):
        system = FlatBottomRaySystem(FlatBottomEnvironment(ssp, bottom_depth=4500))
        x = sample_states(ssp)
        analytic = system.hessian(x)
        # Check two step sizes, since finite differences also have numerical error.
        for steps in (STEPS, (0.25, 5e-8)):
            _, numeric = numerical_gradient_hessian(system.hamiltonian, x, steps)
            # Each entry has different units and magnitude; compare separately.
            assert_allclose(analytic[:, 0, 0], numeric[:, 0, 0], rtol=2e-5, atol=2e-15)
            assert_allclose(analytic[:, 0, 1], numeric[:, 0, 1], rtol=2e-5, atol=1e-10)
            assert_allclose(analytic[:, 1, 0], numeric[:, 1, 0], rtol=2e-5, atol=1e-10)
            assert_allclose(analytic[:, 1, 1], numeric[:, 1, 1], rtol=2e-5, atol=1e-5)


def test_linearized_field_against_numerical_jacobian():
    for ssp in (DownwardRefractingSSP(), MunkSSP()):
        system = FlatBottomRaySystem(FlatBottomEnvironment(ssp, bottom_depth=4500))
        x = sample_states(ssp)
        numeric = np.empty((len(x), 2, 2))
        for i, h in enumerate(STEPS):
            offset = np.eye(2)[i]*h
            numeric[:, :, i] = (system.vector_field(x + offset)
                                - system.vector_field(x - offset)) / (2*h)
        assert_allclose(system.linearized_field(x), numeric, rtol=2e-5, atol=1e-12)
