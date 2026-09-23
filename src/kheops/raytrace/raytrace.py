from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from kheops.environment import FlatBottomEnvironment
from kheops.raytrace.hamiltonian import FlatBottomRaySystem
from kheops.raytrace.raystate import RayState
from kheops.raytrace.stepper import RK4Stepper


@dataclass
class RayEvent:
    ray_id: int
    kind: str
    r: float
    x_before: np.ndarray
    x_after: np.ndarray
    M_before: np.ndarray
    M_after: np.ndarray
    travel_time: float


@dataclass
class TraceConfig:
    final_range: float
    range_step: float = 10.0
    save_every: int = 1
    event_tolerance: float = 1e-7  # metres of range
    maximum_events_per_step: int = 20


@dataclass
class RayTraceResult:
    r: np.ndarray
    x: np.ndarray
    M: np.ndarray
    travel_time: np.ndarray
    launch_depth: np.ndarray
    launch_angle: np.ndarray
    events: list[RayEvent]
    metadata: dict

    @property
    def z(self):
        return self.x[..., 0]

    @property
    def pz(self):
        return self.x[..., 1]

    @property
    def ray_id(self):
        return np.arange(self.x.shape[1])

    def events_for(self, ray_id, kind=None):
        return [e for e in self.events if e.ray_id == ray_id and (kind is None or e.kind == kind)]

    def path_for(self, ray_id, include_events=True):
        """Return (range, depth), inserting event positions for plotting."""
        r, z = self.r.copy(), self.z[:, ray_id].copy()
        if include_events:
            events = self.events_for(ray_id)
            r = np.concatenate((r, [e.r for e in events]))
            z = np.concatenate((z, [e.x_before[0] for e in events]))
            order = np.argsort(r, kind='stable')
            r, z = r[order], z[order]
        return r, z

    def spreading(self):
        """dz/d(theta0) at fixed initial depth; theta0 is in radians."""
        return self.M[..., 0, 1] * self.metadata['dpz_dangle']

    def to_xarray(self):
        import xarray as xr
        return xr.Dataset(
            {'z': (('range', 'ray'), self.z),
             'pz': (('range', 'ray'), self.pz),
             'travel_time': (('range', 'ray'), self.travel_time),
             'M': (('range', 'ray', 'component', 'initial_component'), self.M)},
            coords={'range': self.r, 'ray': self.ray_id,
                    'component': ['z', 'pz'], 'initial_component': ['z', 'pz']},
        )


class RayTracer:
    def __init__(self, environment, stepper=None):
        self.system = FlatBottomRaySystem(environment)
        self.stepper = stepper if stepper is not None else RK4Stepper()

    def _locate(self, start, end, value, tolerance):
        """Bisect a bracket by taking shorter complete steps from start."""
        low, high = 0.0, end.r - start.r
        initial = value(start)
        for _ in range(80):
            middle = (low + high)/2
            state = self.stepper.advance(self.system, start, middle)
            if high - low <= tolerance:
                return state
            if initial * value(state) > 0:
                low = middle
            else:
                high = middle
        raise RuntimeError('Event localization did not converge.')

    @staticmethod
    def _event(ray_id, kind, before, after):
        return RayEvent(ray_id, kind, before.r, before.x[0].copy(), after.x[0].copy(),
                        before.M[0].copy(), after.M[0].copy(), float(before.travel_time[0]))

    def _observe(self, ray_id, start, end, events, config):
        # M[0,1] is proportional to angular spreading for a fixed-depth launch.
        for kind, value in [('turning', lambda s: s.x[0, 1]),
                            ('caustic', lambda s: s.M[0, 0, 1])]:
            a, b = value(start), value(end)
            if a != 0 and (a*b < 0 or b == 0):
                state = self._locate(start, end, value, config.event_tolerance)
                if not any(e.ray_id == ray_id and e.kind == kind
                           and abs(e.r - state.r) <= 2*config.event_tolerance
                           for e in reversed(events)):
                    events.append(self._event(ray_id, kind, state, state))

    def _advance_ray(self, ray_id, start, end, events, config):
        env = self.system.environment
        target = end.r
        for _ in range(config.maximum_events_per_step + 1):
            boundary = None
            depth_tolerance = abs(self.system.vector_field(end.x)[0, 0])*config.event_tolerance
            if end.x[0, 0] <= env.surface_depth + depth_tolerance and end.x[0, 1] < 0:
                boundary = ('surface', env.surface_depth)
            elif end.x[0, 0] >= env.bottom_depth - depth_tolerance and end.x[0, 1] > 0:
                boundary = ('bottom', env.bottom_depth)
            if boundary is None:
                self._observe(ray_id, start, end, events, config)
                return end
            kind, depth = boundary
            hit = self._locate(start, end, lambda s: s.x[0, 0] - depth,
                               config.event_tolerance)
            hit.x[0, 0] = depth
            self._observe(ray_id, start, hit, events, config)
            reflected = hit.ray(0)
            reflected.x[0, 1] *= -1
            # Saltation matrix: includes the variation of reflection range.
            a, b = self.system.vector_field(hit.x)[0]
            saltation = np.array([[-1.0, 0.0], [2*b/a, -1.0]])
            reflected.M[0] = saltation @ hit.M[0]
            events.append(self._event(ray_id, kind, hit, reflected))
            remaining = target - hit.r
            if remaining <= 0:
                return reflected
            start = reflected
            end = self.stepper.advance(self.system, start, remaining)
        raise RuntimeError('Too many reflections in one step; reduce range_step.')

    def trace(self, depths, angles, config):
        """Launch at r=0. Angles are radians, positive downward from horizontal."""
        env = self.system.environment
        z, theta = [a.ravel().copy() for a in np.broadcast_arrays(
            np.atleast_1d(np.asarray(depths, dtype=float)),
            np.atleast_1d(np.asarray(angles, dtype=float)))]
        if config.range_step <= 0 or config.final_range < 0 or config.save_every < 1:
            raise ValueError('Use a positive step/save interval and nonnegative final range.')
        if np.any((z <= env.surface_depth) | (z >= env.bottom_depth)):
            raise ValueError('Launch depths must lie strictly inside the water column.')
        if np.any(np.abs(theta) >= np.pi/2):
            raise ValueError('Launch angles must lie strictly between -pi/2 and pi/2.')
        c = env.ssp.c(z)
        x = np.stack((z, np.sin(theta)/c), axis=-1)
        state = RayState(0.0, x, np.broadcast_to(np.eye(2), (len(z), 2, 2)).copy(), np.zeros(len(z)))
        samples, events = [state], []
        count = int(np.ceil(config.final_range/config.range_step))
        for step in range(1, count + 1):
            target = min(step*config.range_step, config.final_range)
            proposed = self.stepper.advance(self.system, state, target - state.r)
            for ray_id in range(len(z)):
                resolved = self._advance_ray(ray_id, state.ray(ray_id), proposed.ray(ray_id), events, config)
                proposed.x[ray_id] = resolved.x[0]
                proposed.M[ray_id] = resolved.M[0]
                proposed.travel_time[ray_id] = resolved.travel_time[0]
            state = proposed
            if step % config.save_every == 0 or step == count:
                samples.append(state)
        events.sort(key=lambda e: (e.r, e.ray_id))
        return RayTraceResult(
            np.array([s.r for s in samples]), np.stack([s.x for s in samples]),
            np.stack([s.M for s in samples]), np.stack([s.travel_time for s in samples]),
            z, theta, events,
            {'stepper': type(self.stepper).__name__, 'range_step': config.range_step,
             'save_every': config.save_every, 'dpz_dangle': np.cos(theta)/c,
             'surface_depth': env.surface_depth, 'bottom_depth': env.bottom_depth,
             'ssp': repr(env.ssp), 'units': 'metres, seconds, radians'},
        )
