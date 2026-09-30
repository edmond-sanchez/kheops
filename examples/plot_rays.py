import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from kheops.art.rays import plot_rays
from kheops.environment import FlatBottomEnvironment
from kheops.environment.ssp import MunkSSP
from kheops.raytrace.raytrace import RayTracer, TraceConfig

environment = FlatBottomEnvironment(MunkSSP(), bottom_depth=4500)
tracer = RayTracer(environment)

result = tracer.trace(
    depths=1000,
    angles=np.deg2rad(np.linspace(-12, 12, 50)),
    config=TraceConfig(final_range=50000, range_step=100),
)

plot_rays(
    result,
    show_reflections=True,
    show_caustics=True,
    show_transmission_loss=True,
    show_turning_points=True,
    n_max_rays=50,
    cmap="rainbow_r",
    norm=Normalize(70, 90)
)
plt.show()