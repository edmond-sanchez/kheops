import matplotlib.pyplot as plt
import numpy as np

from kheops.art.rays import plot_rays
from kheops.environment import FlatBottomEnvironment
from kheops.environment.ssp import MunkSSP
from kheops.raytrace.raytrace import RayTracer, TraceConfig

environment = FlatBottomEnvironment(MunkSSP(), bottom_depth=4500)
tracer = RayTracer(environment)

result = tracer.trace(
    depths=1000,
    angles=np.deg2rad(np.linspace(-12, 12, 17)),
    config=TraceConfig(final_range=50000, range_step=100),
)

plot_rays(result, show_reflections=True)
plt.show()