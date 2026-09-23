import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
from numpy.typing import NDArray

from kheops.raytrace.raytrace import RayTraceResult


def plot_rays(
    result: RayTraceResult,
    ax: Axes | None = None,
    *,
    amplitude: NDArray[np.floating] | None = None,
    show_reflections: bool = False,
    show_caustics: bool = False,
    show_turning_points: bool = False,
    cmap: str = 'viridis',
    norm: Normalize | None = None,
    colorbar: bool = True,
    amplitude_label: str = 'Amplitude',
    linewidth: float = 0.8,
) -> Axes:
    """Plot rays with depth increasing downward. Return ax; do not call show().

    amplitude, when supplied, has shape (saved ranges, rays), like result.z.
    Pass finite, real values from your amplitude model, or precomputed dB values
    with a suitable amplitude_label. One colour scale is shared by all rays.
    norm can be a Matplotlib Normalize or LogNorm instance.

    Event positions are always included in the ray paths, even when their
    markers are hidden. Amplitudes at those positions are linearly interpolated
    from saved samples; discontinuities at reflections are not reconstructed.
    """
    if amplitude is not None:
        amplitude = np.asarray(amplitude)
        if amplitude.shape != result.z.shape:
            raise ValueError('amplitude must have the same shape as result.z.')
        if not np.isrealobj(amplitude) or not np.all(np.isfinite(amplitude)):
            raise ValueError('amplitude must contain finite real values.')
        if norm is None:
            norm = Normalize(vmin=amplitude.min(), vmax=amplitude.max())

    if ax is None:
        _, ax = plt.subplots(layout='constrained')

    segments, colours = [], []
    for ray_id in result.ray_id:
        r, z = result.path_for(ray_id, include_events=True)
        if amplitude is None:
            ax.plot(r, z, color='black', linewidth=linewidth)
        else:
            points = np.column_stack((r, z))
            ray_segments = np.stack((points[:-1], points[1:]), axis=1)
            values = np.interp(r, result.r, amplitude[:, ray_id])
            # Colour each segment by the mean amplitude at its two ends.
            segments.extend(ray_segments)
            colours.extend((values[:-1] + values[1:]) / 2)

    if amplitude is not None:
        rays = LineCollection(segments, cmap=cmap, norm=norm, linewidths=linewidth)
        rays.set_array(np.asarray(colours))
        ax.add_collection(rays)
        ax.autoscale_view()
        if colorbar:
            ax.figure.colorbar(rays, ax=ax, label=amplitude_label)

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
