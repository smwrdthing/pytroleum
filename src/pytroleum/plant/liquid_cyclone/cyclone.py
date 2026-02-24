from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Literal

from numpy.typing import NDArray

import numpy as np
from scipy.integrate import cumulative_trapezoid

from pytroleum.plant.liquid_cyclone.flowsheets import FlowSheet
from pytroleum.plant.liquid_cyclone.velocities import VelocityField

from enum import IntEnum, auto

# TODO :
# fix type annotations, docstrings


START_OF_ARRAY = 0
NUMBER_OF_ARRAY_POINTS = 500

# region Abstract


@dataclass
class Design(ABC):

    inlet_diameter: float
    characteristic_diameter: float
    model_length: float

    inlet_area: float = field(init=False)

    _model_length_array: NDArray = field(init=False)
    _wall_area_array: NDArray = field(init=False)

    def __post_init__(self):

        self.inlet_area = np.pi/4*self.inlet_diameter**2

        # we use precomputed areas & interpolation instead of array creation to compute
        # integrals each time
        self._model_length_array = np.linspace(
            START_OF_ARRAY, self.model_length, NUMBER_OF_ARRAY_POINTS)
        self._wall_area_array = 2*np.pi*cumulative_trapezoid(
            self.wall(self._model_length_array), self._model_length_array, initial=0)

    @abstractmethod
    def wall(self, axial_coordinate: NDArray | float) -> NDArray:
        raise

    @abstractmethod
    def wall_slope(self, axial_coordinate: NDArray | float) -> NDArray:
        raise

    def wall_incline(self, axial_coordinate, total: Literal[0, 1] | bool = 0) -> NDArray:
        # convenience method
        return (1+total)*np.arctan(self.wall_slope(axial_coordinate))


# region Southampton
# Auxiliary classes for consistent storage of data pertaining to Southampton LLH design
# and convenient access to it

class SouthamptonDiameters(IntEnum):
    I, INLET = 0, 0
    O, OVERFLOW = 1, 1
    E, ENTRY = 2, 2
    C, CONE, CHARARCTER = 3, 3, 3
    U, UNDERFLOW = 4, 4

    SIZE = auto()


class SouthamptonLengths(IntEnum):
    E, ENTRY = 0, 0
    C, CONE, CHARACTER = 1, 1, 1
    T, TAPERED = 2, 2
    U, UNDERFLOW = 3, 3

    SIZE = auto()


class SouthamptonAngles(IntEnum):
    C, CONE, CHARACTER = 0, 0, 0
    T, TAPERED = 1, 1

    SIZE = auto()


@dataclass
class SouthamptonDesign(Design):

    entrance_diameter: float

    inlet_diameter: float = field(init=False)
    inlet_area: float = field(init=False)

    characteristic_diameter: float = field(init=False)

    model_length: float = field(init=False)

    # Initialize containers, fill in __post_init__ later for clarity, probably slower,
    # but much more readable
    diameters: NDArray = field(
        default_factory=lambda: np.zeros(SouthamptonDiameters.SIZE))
    lengths: NDArray = field(
        default_factory=lambda: np.zeros(SouthamptonLengths.SIZE))
    angles: NDArray = field(
        default_factory=lambda: np.zeros(SouthamptonAngles.SIZE))

    _length_proportions: NDArray = field(
        default_factory=lambda: np.zeros(SouthamptonLengths.SIZE))
    _diameter_proportions: NDArray = field(
        default_factory=lambda: np.zeros(SouthamptonDiameters.SIZE))

    def __post_init__(self):

        # Diameters specification
        self._diameter_proportions[SouthamptonDiameters.INLET] = 0.175
        self._diameter_proportions[SouthamptonDiameters.OVERFLOW] = 0.05
        self._diameter_proportions[SouthamptonDiameters.ENTRY] = 1.0
        self._diameter_proportions[SouthamptonDiameters.CHARARCTER] = 0.5
        self._diameter_proportions[SouthamptonDiameters.UNDERFLOW] = 0.25
        self.diameters = self.entrance_diameter*self._diameter_proportions

        self.inlet_diameter = self.diameters[SouthamptonDiameters.I]
        self.characteristic_diameter = self.diameters[SouthamptonDiameters.C]

        # Angles specification
        self.angles[SouthamptonAngles.CHARACTER] = np.deg2rad(20.0)
        self.angles[SouthamptonAngles.TAPERED] = np.deg2rad(1.5)

        # Length specification
        self._length_proportions[SouthamptonLengths.ENTRY] = 1.0

        self._length_proportions[SouthamptonLengths.CONE] = (
            1-self._diameter_proportions[SouthamptonDiameters.CONE])/(
                2*np.tan(self.angles[SouthamptonAngles.CONE]/2))

        self._length_proportions[SouthamptonLengths.TAPERED] = (
            self._diameter_proportions[SouthamptonDiameters.C] -
            self._diameter_proportions[SouthamptonDiameters.U])/(
                2*np.tan(self.angles[SouthamptonAngles.T]/2))

        self._length_proportions[SouthamptonLengths.UNDERFLOW] = 15.0
        self.lengths = self.entrance_diameter*self._length_proportions

        # This is used for interpolation in wall function
        self.model_length = np.sum(self.lengths[SouthamptonLengths.TAPERED:])

        # Coordinates and function values for wall-related computations
        self._axial_wall_coordinates = np.array([
            0.0,
            self.lengths[SouthamptonLengths.TAPERED],
            self.lengths[SouthamptonLengths.TAPERED] +
            self.lengths[SouthamptonLengths.UNDERFLOW]])

        self._radial_wall_coordinates = np.array([
            self.diameters[SouthamptonDiameters.CHARARCTER],
            self.diameters[SouthamptonDiameters.UNDERFLOW],
            self.diameters[SouthamptonDiameters.UNDERFLOW]])/2

        self._slopes = -np.tan(np.array([
            self.angles[SouthamptonAngles.TAPERED]/2,
            self.angles[SouthamptonAngles.TAPERED]/2,
            0]))

        super().__post_init__()

    def wall(self, axial_coordinate: NDArray) -> NDArray:
        radius = np.interp(
            axial_coordinate,
            self._axial_wall_coordinates,
            self._radial_wall_coordinates)
        return radius

    def wall_slope(self, axial_coordinate: NDArray) -> NDArray:
        slope = np.tan(-self.angles[SouthamptonAngles.T]/2*(
            axial_coordinate < self.lengths[SouthamptonLengths.T]))
        return slope

    def summary(self):

        from pytroleum.plant.liquid_cyclone.utils import _minor_divider, _major_divider

        to_mm = 1000
        to_squared_cm = 1e4

        _major_divider()
        print("SOUTHAMPTON LIQUID-LIQUID-HYDROCYCLONE SUMMARY")
        _major_divider()

        print("DIAMETERS :: ")
        print(
            f"Inlet : {self.diameters[SouthamptonDiameters.INLET]*to_mm:.2f} mm")
        print(
            f"Overflow : {self.diameters[SouthamptonDiameters.OVERFLOW]*to_mm:.2f} mm")
        print(
            f"Entry : {self.diameters[SouthamptonDiameters.ENTRY]*to_mm:.2f} mm")
        print(
            f"Cone : {self.diameters[SouthamptonDiameters.CONE]*to_mm:.2f} mm")
        print(
            f"Underflow : {self.diameters[SouthamptonDiameters.UNDERFLOW]*to_mm:.2f} mm")

        _minor_divider()

        print("LENGHTS :: ")
        print(f"Entry : {self.lengths[SouthamptonLengths.ENTRY]*to_mm:.2f} mm")
        print(f"Cone : {self.lengths[SouthamptonLengths.CONE]*to_mm:.2f} mm")
        print(
            f"Tapered : {self.lengths[SouthamptonLengths.TAPERED]*to_mm:.2f} mm")
        print(
            f"Underflow : {self.lengths[SouthamptonLengths.UNDERFLOW]*to_mm:.2f} mm")
        print(f"Total : {np.sum(self.lengths)*to_mm:.2f} mm")

        _minor_divider()

        print("ANGLES :: ")
        print(
            f"Cone : {np.rad2deg(self.angles[SouthamptonAngles.CONE]):.2f} deg")
        print(
            f"Tapered : {np.rad2deg(self.angles[SouthamptonAngles.TAPERED]):.2f} deg")

        _minor_divider()

        print(f"Inlet area : {self.inlet_area*to_squared_cm: .2f} cm^2")
        print(
            f"Characteristic diameter : {self.characteristic_diameter*to_mm: .2f} mm")

        _major_divider()
        print("END OF SUMMARY")
        _major_divider()
        print()

# region LLH


class LiquidLiquidHydrocyclone:

    def __init__(self, design: Design, flowsheet: FlowSheet) -> None:
        self.design = design
        self.flowsheet = flowsheet


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pytroleum.plant.liquid_cyclone.flowsheets import ResistanceSpec
    from pytroleum.plant.liquid_cyclone import utils
    from pytroleum.plant.liquid_cyclone import separation as sep

    def compute_resistance(density, area, discharge_coeff):
        # auxiliary function
        return discharge_coeff*area/np.sqrt(density)

    # Fluid properties
    oil_fraction = 10/100
    water_fraction = 1-oil_fraction

    oil_density = 830
    water_density = 1000
    inlet_density = water_density*water_fraction + oil_density*oil_fraction

    # Diameters
    overflow_diameter = 3e-3
    underflow_diameter = 12e-3
    inlet_diameter = 25e-3

    overflow_valve_diameter = 6e-3
    underflow_valve_diameter = 12e-3

    # Areas
    overflow_area = np.pi/4*overflow_diameter**2
    underflow_area = np.pi/4*underflow_diameter**2
    inlet_area = np.pi/4*inlet_diameter**2

    overflow_valve_area = np.pi/4*overflow_valve_diameter**2
    underflow_valve_area = np.pi/4*underflow_valve_diameter**2

    # Discharge coefficients
    overflow_discharge_coeff = 0.2
    underflow_discharge_coeff = 0.4
    inlet_discharge_coeff = 0.1

    overflow_valve_discharge_coeff = 0.61
    underflow_valve_discharge_coeff = 0.61

    # Openings
    overflow_valve_opening = 1.0
    underflow_valve_opening = 1.0

    # Collecting inputs
    overflow_inputs = (oil_density,
                       overflow_area,
                       overflow_discharge_coeff)

    underflow_inputs = (water_density,
                        underflow_area,
                        underflow_discharge_coeff)

    inlet_inputs = (inlet_density,
                    inlet_area,
                    inlet_discharge_coeff)

    overflow_valve_inputs = (oil_density,
                             overflow_valve_area*overflow_valve_opening,
                             overflow_valve_discharge_coeff)

    underflow_valve_inputs = (water_density,
                              underflow_valve_area*underflow_valve_opening,
                              underflow_valve_discharge_coeff)

    # region design & flowsheet
    design = SouthamptonDesign(80e-3)
    design.summary()
    flowsheet = FlowSheet()

    # Set resistances
    flowsheet.resistance[ResistanceSpec.O] = compute_resistance(
        *overflow_inputs)
    flowsheet.resistance[ResistanceSpec.U] = compute_resistance(
        *underflow_inputs)
    flowsheet.resistance[ResistanceSpec.IN] = compute_resistance(
        *inlet_inputs)
    flowsheet.resistance[ResistanceSpec.OV] = compute_resistance(
        *overflow_valve_inputs)
    flowsheet.resistance[ResistanceSpec.UV] = compute_resistance(
        *underflow_valve_inputs)

    Q_in = 0.5e-3
    backpressures = (1.5e5, 4.5e5)
    flowsheet.solve_from_backpressures(Q_in, backpressures)
    flowsheet.account_for_recirculation(recirculation_rate=0.02)
    flowsheet.summary()

    # region velocity field
    velocity_field = VelocityField()
    velocity_field._solve_ndim_reversal_radius(flowsheet)
    velocity_field._restore_ndim_coeffs(flowsheet)

    setup = flowsheet, design, velocity_field

    fig, ax = utils.plot_velocity_field(
        setup, "radial", drop_diameter=0.0e-6)  # type: ignore
    ax.set_title("Radial velocity")

    fig, ax = utils.plot_velocity_field(
        setup, "tangent")  # type: ignore
    ax.set_title("Tangent velocity")

    fig, ax = utils.plot_velocity_field(
        setup, "axial")  # type: ignore
    ax.set_title("Axial velocity")

    # region separation
    r_final, z_final = sep.drop_final_point(5e-6, setup)  # type: ignore
    largest_diameter = sep.solve_largest_drop(setup)  # type: ignore

    diameters = np.linspace(0, largest_diameter, 10)  # type: ignore
    fig, ax = utils.plot_model_region(design, velocity_field, half=True)
    for i, d in enumerate(diameters):
        r, z = sep.drop_trajectroy(d, setup)  # type: ignore
        linewidth = 3
        if i == 0:
            color = "b"
        elif i == len(diameters)-1:
            color = "r"
        else:
            linewidth = 1
            color = None
        ax.plot(z*1e3, r*1e3, color=color, linewidth=linewidth)

    d_efficiency, G_efficiencey = sep.build_grade_efficiency_curve(
        setup, 30)  # type: ignore

    d50 = sep.extract_d50(d_efficiency, G_efficiencey)

    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    ax.set_title("Grade efficiency")
    ax.set_xlabel(r"drop diameter [$\mu m$]")
    ax.set_ylabel(r"G [$\%$]")
    ax.plot(d_efficiency*1e6, G_efficiencey*100)
    ax.plot(d50*1e6, 50, 'go')
    ax.set_xlim((0, largest_diameter*1e6))  # type: ignore
    ax.set_ylim((0, 100))
    ax.vlines(d50*1e6, 0, 100, 'g', '--')
    ax.hlines(50, 0, largest_diameter*1e6, 'g', '--')  # type: ignore
    ax.grid(True)

    print(f'd50 is {d50*1e6:.2f} micrometers')
    # Looks believable, I'd believe

    plt.show()

    # Total efficiency computations
    percentiles = (15e-6, 20e-6, 50e-6)
    efficiency = sep.evaluate_total_efficiency(
        setup, percentiles)  # type: ignore
    # Value is seriously off

    print("For given size distribution and LLH setup" +
          f"total efficiency is : {efficiency*100:.2f} %")

    # Something to do with quad and drop sizes being in micrometers scale
