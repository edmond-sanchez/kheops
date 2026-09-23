import numpy as np
from numpy.testing import assert_allclose

from kheops.environment import FlatBottomEnvironment
from kheops.environment.ssp import MunkSSP
from kheops.raytrace.hamiltonian import FlatBottomRaySystem
from kheops.raytrace.raystate import RayState
from kheops.raytrace.stepper import (
    EulerStepper,
    ImplicitMidpointStepper,
    RK2Stepper,
    RK4Stepper,
    YoshidaStepper,
)

STEPPERS = (EulerStepper(), RK2Stepper(), RK4Stepper(),
            ImplicitMidpointStepper(), YoshidaStepper())
SYSTEM = FlatBottomRaySystem(FlatBottomEnvironment(MunkSSP(), bottom_depth=4500))


def initial_state():
    z = np.array([900.0, 1100.0, 1600.0])
    angles = np.array([-0.04, 0.1, -0.08])
    p = np.sin(angles) / SYSTEM.environment.ssp.c(z)
    return RayState(0.0, np.stack((z, p), axis=-1),
                    np.repeat(np.eye(2)[None], len(z), axis=0), np.zeros(len(z)))


def integrate(stepper, step, final_range=8000):
    state = initial_state()
    for _ in range(round(final_range / step)):
        state = stepper.advance(SYSTEM, state, step)
    return state


def scaled_state(state):
    # Compare depth, momentum and travel time using their typical scales.
    return np.column_stack((state.x[:, 0] / 1000,
                            state.x[:, 1] * 1500,
                            state.travel_time / (8000 / 1500)))


def test_convergence_order():
    reference = scaled_state(integrate(RK4Stepper(), 5))
    finer_reference = scaled_state(integrate(RK4Stepper(), 2.5))
    assert_allclose(reference, finer_reference, rtol=0, atol=1e-12)

    for stepper, order in zip(STEPPERS, (1, 2, 4, 2, 4)):
        errors = [np.linalg.norm(scaled_state(integrate(stepper, h)) - reference)
                  for h in (400, 200, 100)]
        # Halving h should divide global error by approximately 2**order.
        measured_orders = np.log2(np.array(errors[:-1]) / errors[1:])
        assert_allclose(measured_orders, order, atol=0.15, rtol=0,
                        err_msg=type(stepper).__name__)


def test_symplecticity():
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    for stepper in (ImplicitMidpointStepper(), YoshidaStepper()):
        state = initial_state()
        for _ in range(40):
            state = stepper.advance(SYSTEM, state, 200)
            # Rescale coordinates to avoid large entries caused just by units.
            scale = np.array([1 / 1000, 1500])
            M = state.M * scale[:, None] / scale[None, :]
            product = np.swapaxes(M, -1, -2) @ J @ M
            assert_allclose(product, np.broadcast_to(J, product.shape), rtol=0, atol=1e-11)
            assert_allclose(np.linalg.det(M), 1, rtol=0, atol=1e-11)


def test_explicit_methods_are_not_symplectic():
    # A nonconstant SSP exposes the defect; straight rays would hide it.
    for stepper in (EulerStepper(), RK2Stepper(), RK4Stepper()):
        end = stepper.advance(SYSTEM, initial_state(), 1000)
        defect = np.max(np.abs(np.linalg.det(end.M) - 1))
        assert defect > 1e-10, type(stepper).__name__


def test_time_reversibility():
    start = initial_state()
    for stepper in (ImplicitMidpointStepper(), YoshidaStepper()):
        end = stepper.advance(SYSTEM, start, 200)
        back = stepper.advance(SYSTEM, end, -200)
        assert_allclose(back.x, start.x, rtol=1e-11, atol=1e-12)
        assert_allclose(back.M, start.M, rtol=0, atol=1e-9)
        assert_allclose(back.travel_time, start.travel_time, rtol=0, atol=1e-12)


def test_monodromy_matches_numerical_derivative_of_step():
    start = initial_state()
    for stepper in STEPPERS:
        end = stepper.advance(SYSTEM, start, 200)
        numeric = np.empty_like(end.M)
        for column, h in enumerate((0.001, 1e-9)):
            plus, minus = initial_state(), initial_state()
            plus.x[:, column] += h
            minus.x[:, column] -= h
            numeric[..., column] = (
                stepper.advance(SYSTEM, plus, 200).x
                - stepper.advance(SYSTEM, minus, 200).x
            ) / (2*h)
        assert_allclose(end.M, numeric, rtol=2e-6, atol=1e-7,
                        err_msg=type(stepper).__name__)


def test_step_does_not_modify_input():
    for stepper in STEPPERS:
        start = initial_state()
        expected = initial_state()
        stepper.advance(SYSTEM, start, 200)
        assert start.r == expected.r
        assert_allclose(start.x, expected.x, rtol=0, atol=0)
        assert_allclose(start.M, expected.M, rtol=0, atol=0)
        assert_allclose(start.travel_time, expected.travel_time, rtol=0, atol=0)
