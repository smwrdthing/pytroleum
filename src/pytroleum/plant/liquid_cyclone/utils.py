from typing import Literal, TYPE_CHECKING
from numpy.typing import NDArray

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pytroleum.plant.liquid_cyclone.velocities import VelocityField
from pytroleum.plant.liquid_cyclone.flowsheets import FlowSheet

from pytroleum.plant.liquid_cyclone.cyclone import Design

from pytroleum.plant.liquid_cyclone.cyclone import SouthamptonDiameters
from pytroleum.plant.liquid_cyclone.cyclone import SouthamptonLengths
from pytroleum.plant.liquid_cyclone.cyclone import SouthamptonAngles
from pytroleum.plant.liquid_cyclone.cyclone import SouthamptonDesign


# NOTE :
# maybe later rewright this for instance of LiquidLiquidHydrocyclone

# Because matplotlib's figsize works with inches only
_INCH_TO_CM = 2.54
_CM_TO_INCHES = 1/_INCH_TO_CM
_METER_TO_MM = 1000

FIG_WIDTH_CM = 35
FIG_HEIGHT_CM = 10

AX_ASPECT = 20
CBAR_ASPECT = 75

MESH_SIZE = (100, 100)
CONTOUR_LEVELS = 100
CBAR_PAD = 0.2

_DIVIDER_LENGTH = 80
_MINOR_DIVIDER = '-'*_DIVIDER_LENGTH
_MAJOR_DIVIDER = '='*_DIVIDER_LENGTH


def _minor_divider():
    print(_MINOR_DIVIDER)


def _major_divider():
    print(_MAJOR_DIVIDER)


def _generate_model_region_mesh(design: Design, velocity_field: VelocityField,
                                size: tuple[int, int]) -> tuple[NDArray, NDArray]:

    n, m = size

    ndim_r = np.linspace(velocity_field.ndim_reversal_radius, 1, n)
    ndim_z = np.linspace(0, 1, m)

    NDIM_R, NDIM_Z = np.meshgrid(ndim_r, ndim_z)

    Z = NDIM_Z*design.model_length
    R = NDIM_R*design.wall(Z)

    return R, Z


def plot_model_region(design: Design, velocity_field: VelocityField, half: bool = False):

    # NOTE :
    # Currently works only with Southampton desing because it is one of the most
    # fully-defined design options of liquid-liquid hydrocyclone

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_CM*_CM_TO_INCHES,
                                    FIG_HEIGHT_CM*_CM_TO_INCHES))
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r [mm]")

    # Main arrays to work with, axial & radial coordinates
    z = design._model_length_array
    r_wall = design.wall(z)

    # Max radius
    r_max = np.max(r_wall)

    # Reversal region
    r_reversal = r_wall*velocity_field.ndim_reversal_radius

    # unit conversion
    z = z*_METER_TO_MM

    r_wall = r_wall*_METER_TO_MM
    r_reversal = r_reversal*_METER_TO_MM

    r_max = r_max*_METER_TO_MM

    # Plotting walls
    ax.plot(z, r_wall, "k")
    ax.plot(z, -r_wall, "k")
    ax.plot(z, r_reversal, "--k")
    ax.plot(z, -r_reversal, "--k")

    if hasattr(design, "diameters") and hasattr(design, "lengths"):
        # ^^^^^^^^^^ hasattr is a temporary patch, because isinstance fails to
        # check if design is a SouthamptonDesign

        # NOTE :
        # For some reasons this check fails even when we pass SothamptonDesgn
        # object as design

        z_TU = design.lengths[SouthamptonLengths.TAPERED]  # type: ignore
        r_TU = design.diameters[  # type: ignore
            SouthamptonDiameters.UNDERFLOW]/2

        z_TU = z_TU*_METER_TO_MM
        r_TU = r_TU*_METER_TO_MM

        # Segmentation with dashed line & axis
        ax.vlines(z_TU, -r_TU, r_TU, 'k', '--')

    # Axis
    ax.hlines((0, 0), 0, z[-1], '0.8', '-.', linewidths=0.8)

    if half:
        ymin = 0
    else:
        ymin = -r_max

    # plot restriction
    ax.set_xlim((z[0], z[-1]))
    ax.set_ylim((ymin, r_max))

    # Decoratives
    ax.fill_between(
        z, r_max*np.ones_like(z), r_wall,
        color='none', edgecolor='0.8', hatch='///', hatch_linewidth=0.5)
    ax.fill_between(
        z, -r_max*np.ones_like(z), -r_wall,
        color='none', edgecolor='0.8', hatch='///', hatch_linewidth=0.5)

    ax.fill_between(z, -r_reversal, r_reversal, color='0.95')

    return fig, ax


def plot_velocity_field(
        flowsheet: FlowSheet, design: SouthamptonDesign, velocity_field: VelocityField,
        component: Literal["radial", "tangent", "axial"], with_slip: bool = False):

    fig, ax = plot_model_region(design, velocity_field)
    coordinates = _generate_model_region_mesh(
        design, velocity_field, MESH_SIZE)

    match component:
        case "radial":
            # TODO :
            # figure why type checker is not happy with Southampton desing,
            # remove all "type: ignore" flags after
            velocity = velocity_field.radial_component(
                coordinates, flowsheet, design)  # type: ignore
            if with_slip:
                # TODO :
                # implemet slip velocity
                # velocity = velocity + velocity_field.drop_slip_velocity()
                raise
            cbar_label = "u [mm/s]"
            unit_scale = 1e3
        case "tangent":
            velocity = velocity_field.tangent_component(
                coordinates, flowsheet, design)  # type: ignore
            cbar_label = "v [m/s]"
            unit_scale = 1
        case "axial":
            velocity = velocity_field.axial_component(
                coordinates, flowsheet, design)  # type: ignore
            cbar_label = "w [m/s]"
            unit_scale = 1

    R, Z = coordinates

    Z = Z*_METER_TO_MM
    R = R*_METER_TO_MM
    velocity = velocity*unit_scale

    cf = ax.contourf(Z, R, velocity, CONTOUR_LEVELS)
    ax.contourf(Z, -R, velocity, CONTOUR_LEVELS)
    fig.colorbar(cf, label=cbar_label, location="bottom",
                 pad=CBAR_PAD, aspect=CBAR_ASPECT)

    return fig, ax
