from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Literal

from numpy.typing import NDArray

import numpy as np
from scipy.integrate import cumulative_trapezoid

from pytroleum.plant.liquid_cyclone.flowsheets import (
    FlowSheet, FlowSpec, ResistanceSpec, PressureSpec)
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
    is_twin_inlet: bool = True

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

        self.inlet_area = self.inlet_area * (1+self.is_twin_inlet)

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
        print(f"Twin inlet? : {self.is_twin_inlet}")
        print(
            f"Characteristic diameter : {self.characteristic_diameter*to_mm: .2f} mm")

        _major_divider()
        print("END OF SUMMARY")
        _major_divider()
        print()

# region LLH
# WIP


class LiquidLiquidHydrocyclone:

    def __init__(self, design: Design, flowsheet: FlowSheet) -> None:
        self.design = design
        self.flowsheet = flowsheet


# NOTE : ON GRADE EFFICIENCY FOR DROPLET WITH "ZERO" DIAMETER
#
# There is some confusion in Bram's work:
# non-dimensional profile dot product with non-dimensional radius should only yield
# underfolw flow rate when integration is performed over whole range of non-dimensional
# raidus values (0 to 1). It is stated in axial velocity model sections multiple
# times. If we start integration from some non-zero non-dimensionsla radius -
# we will not get underflow flow rate. Thus G(0) cannot be equal to flow split without
# adjustment to axial velocity model
#
# I revisited velocity functions multiple times at this point, they should be fine.
# We capture qualitative picture well, so for now let's assume that G(0) should not
# be equalt to split ratio. (radius of droplet with "zero-th" diameter is not zero
# itself due to the bare motion of the continuous flow)
