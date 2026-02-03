import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from numpy import float64
from scipy.constants import g
from pytroleum.sdyna.conductors import _compute_pressure_for


def visualize_pressure_profile():
    """
    Tests the _compute_pressure_for function's correctness.

    Verification involves comparing interpolated pressure profiles with
    control points derived from hydrostatic calculations.

    """
    # Geometry parameters [m]
    h0 = 2.0
    h1 = 1.4
    h2 = 0.6

    # Fluid properties [kg/m³]
    rho_water = 1000.0
    rho_oil = 650.0
    p0 = 101325.0

    # Calculate pressures at phase boundaries
    p1 = p0 + rho_oil * g * (h1 - h2)
    p2 = p0 + rho_oil * g * (h1 - h2) + rho_water * g * (h2 - 0)

    # Additional control points for verification
    h01 = 1.7
    h12 = 1.0
    h20 = 0.3

    # Calculate pressures at control points
    p00 = p0
    p01 = p0+rho_oil * g * (h1 - h12)
    p20 = p0+rho_oil * g * (h1 - h2) + rho_water * g * (h2 - h20)

    # Arrays for interpolation
    levels = np.array([h0, h1, h2])
    pressures = np.array([p0, p1, p2])

    # Create height array
    heights = np.linspace(0, h0, 500)

    # Calculate pressure profile using interpolation
    pressure_values = _compute_pressure_for(
        heights, levels, pressures)  # type: ignore

    # Boundary points for plotting
    interp_heights = np.array([0, h2, h1, h0])
    interp_pressures = np.array([p2, p1, p0, p0])

    # Control points for plotting
    control_heights = np.array([h20, h12, h01])
    control_pressures = np.array([p20, p01, p00])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhspan(h1*1000, h0*1000, alpha=0.2,
               color='lightblue', label='Gas zone')
    ax.axhspan(h2*1000, h1*1000, alpha=0.25, color='gold', label='Oil zone')
    ax.axhspan(0, h2*1000, alpha=0.2, color='lightgreen', label='Water zone')

    ax.plot(pressure_values / 1e5, heights * 1000, 'k-', linewidth=2,
            label='Pressure profile', alpha=0.8, zorder=3)

    ax.plot(interp_pressures / 1e5, interp_heights * 1000, 'ko', markersize=6,
            label='Boundary points', zorder=5, markerfacecolor='black')

    ax.plot(control_pressures / 1e5, control_heights * 1000, 'ro',
            markersize=8, markerfacecolor='none', markeredgewidth=2,
            label='Control points', zorder=6)

    ax.set_xlabel('Pressure, bar', fontsize=12)
    ax.set_ylabel('Height, mm', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, h0 * 1000)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_pressure_profile()
