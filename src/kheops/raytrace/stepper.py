from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from kheops.raytrace.raystate import RayState


class Stepper(ABC):
    """Advance all rays by dr metres, returning a new state."""

    @abstractmethod
    def advance(
        self,
        system: FlatBottomRaySystem,
        state: RayState,
        dr: float,
    ) -> RayState:
        pass


class RK4Stepper(Stepper):
    def advance(self, system, state, dr):
        def rhs(x, M):
            return (system.vector_field(x), system.linearized_field(x) @ M,
                    system.time_rate(x))

        x, M, t = state.x, state.M, state.travel_time
        a = rhs(x, M)
        b = rhs(x + dr/2*a[0], M + dr/2*a[1])
        c = rhs(x + dr/2*b[0], M + dr/2*b[1])
        d = rhs(x + dr*c[0], M + dr*c[1])
        increments = [dr/6*(a[i] + 2*b[i] + 2*c[i] + d[i]) for i in range(3)]
        return RayState(state.r + dr, x + increments[0], M + increments[1], t + increments[2])


@dataclass
class ImplicitMidpointStepper(Stepper):
    """Symmetric, second-order symplectic method for the nonseparable H."""
    tolerance: float = 1e-12
    max_iterations: int = 20

    def advance(self, system, state, dr):
        x = state.x
        y = x + dr*system.vector_field(x)
        # Scale momentum corrections by local slowness, not by one second/metre.
        scale = np.stack((np.maximum(1.0, np.abs(x[:, 0])),
                          1/system.environment.ssp.c(x[:, 0])), axis=-1)
        for _ in range(self.max_iterations):
            mid = (x + y)/2
            A = system.linearized_field(mid)
            lhs = np.eye(2) - dr/2*A
            residual = y - x - dr*system.vector_field(mid)
            correction = np.linalg.solve(lhs, residual[..., None])[..., 0]
            y = y - correction
            if np.max(np.abs(correction)/scale) < self.tolerance:
                break
        else:
            raise RuntimeError("Midpoint solve did not converge; reduce range_step.")
        mid = (x + y)/2
        A = system.linearized_field(mid)
        tangent = np.linalg.solve(np.eye(2) - dr/2*A, np.eye(2) + dr/2*A)
        return RayState(state.r + dr, y, tangent @ state.M,
                        state.travel_time + dr*system.time_rate(mid))


@dataclass
class YoshidaStepper(Stepper):
    """Fourth-order composition of a symmetric second-order base stepper.

    Its internal stages include negative steps. Events are handled around the
    complete composition, not at its internal stages.
    """
    base: Stepper = field(default_factory=ImplicitMidpointStepper)

    def advance(self, system, state, dr):
        w = 1/(2 - 2**(1/3))
        result = state
        for weight in (w, 1 - 2*w, w):
            result = self.base.advance(system, result, weight*dr)
        result.r = state.r + dr
        return result

