from dataclasses import dataclass

from kheops.environment.ssp import SoundSpeedProfile


@dataclass
class FlatBottomEnvironment:
    ssp: SoundSpeedProfile
    bottom_depth: float
    surface_depth: float = 0.0

