import numpy as np

from numpy.typing import NDArray

from pytroleum.plant.liquid_cyclone.flowsheets import FlowSheet
from pytroleum.plant.liquid_cyclone.interfaces import Design, VelocityField


class DropletSeparation:

    def __init__(self) -> None:
        self.grade_efficiency_size_array: NDArray
        self.grade_efficiency_array: NDArray

        self.efficiency: float

    def droplet_motion_equations(
            self, t, Y, setup: tuple[FlowSheet, Design, VelocityField]):

        flowsheet, design, velocity_field = setup

        dY_dt = (velocity_field.radial_component(Y, flowsheet, design) +
                 velocity_field.drop_slip_velocity(),
                 velocity_field.axial_component(Y, flowsheet, design))

        return dY_dt

    def droplet_trajectroy(self, diameter):
        pass

    def build_grade_efficeincy(self):
        pass

    def evaluate_grade_efficiency(self, size):
        grade_efficiency = np.interp(
            size, self.grade_efficiency_size_array, self.grade_efficiency_array)
        return grade_efficiency

    def evaluate_efficieny(self, inlet_size_distribution):
        size, cumulative_distribution_function = inlet_size_distribution
