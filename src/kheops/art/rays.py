import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from kheops.raytrace.raytrace import RayTraceResult


def plot_rays(
    result: RayTraceResult,
    ax: Axes | None = None,
    *,
    show_transmission_loss: bool = False,
    show_reflections: bool = False,
    show_caustics: bool = False,
    show_turning_points: bool = False,
    cmap: str = 'viridis',
    norm: Normalize | None = None,
    colorbar: bool = True,
    linewidth: float = 0.8,
    n_max_rays: int = 50,
) -> Axes:

    ray_ids = result.ray_id
    if len(ray_ids) > n_max_rays:
        ray_ids = ray_ids[np.linspace(0, len(ray_ids) - 1, n_max_rays, dtype=int)]

    transmission_loss = result.transmission_loss() if show_transmission_loss else None
    if transmission_loss is not None:
        finite_tl = transmission_loss[:, ray_ids]
        finite_tl = finite_tl[np.isfinite(finite_tl)]
        if not finite_tl.size:
            raise ValueError('transmission loss has no finite values to plot.')
        if norm is None:
            norm = Normalize(vmin=finite_tl.min(), vmax=finite_tl.max())

    if ax is None:
        _, ax = plt.subplots(layout='constrained')

    segments, colours = [], []
    for ray_id in ray_ids:
        r, z = result.path_for(ray_id, include_events=True)
        if transmission_loss is None:
            ax.plot(r, z, color='black', linewidth=linewidth)
        else:
            points = np.column_stack((r, z))
            ray_segments = np.stack((points[:-1], points[1:]), axis=1)
            values = np.interp(r, result.r, transmission_loss[:, ray_id])
            segments.extend(ray_segments)
            colours.extend((values[:-1] + values[1:]) / 2)

    if transmission_loss is not None:
        rays = LineCollection(segments, cmap=cmap, norm=norm, linewidths=linewidth)
        rays.set_array(np.ma.masked_invalid(colours))
        ax.add_collection(rays)
        ax.autoscale_view()
        if colorbar:
            ax.figure.colorbar(rays, ax=ax, label='Transmission loss (dB re 1 m)')

    event_styles = [
        ('surface', show_reflections, 'tab:blue', 'v', 'Surface reflection'),
        ('bottom', show_reflections, 'tab:orange', '^', 'Bottom reflection'),
        ('caustic', show_caustics, 'tab:red', 'o', 'Caustic'),
        ('turning', show_turning_points, 'tab:green', 's', 'Turning point'),
    ]
    has_markers = False
    for kind, visible, colour, marker, label in event_styles:
        events = [e for e in result.events if e.kind == kind] if visible else []
        if events:
            ax.scatter([e.r for e in events], [e.x_before[0] for e in events],
                       color=colour, marker=marker, s=25, label=label, zorder=3)
            has_markers = True
    if has_markers:
        ax.legend()

    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Depth (m)')
    # Unlike invert_yaxis(), this also works when plotting onto inverted axes.
    ax.yaxis.set_inverted(True)
    return ax
