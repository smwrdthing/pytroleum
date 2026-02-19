import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import fsolve

from pytroleum.plant.liquid_cyclone.flowsheets import FlowSheet, FlowSpec
from pytroleum.plant.liquid_cyclone.interfaces import Design


SUM_RESIDUALS_TOLERANCE = 1e-5
NDIM_REVRADIUS_INITIAL_GUESSE = 0.25

FITTING_POLYNOM_DEGREE = 3
DOTPROD_POLYNOM_DEGREE = FITTING_POLYNOM_DEGREE + 2


class VelocityField:

    # NOTE :
    # We could define different velocity field models by making this ABC
    # and stipulating inheritance for specific models

    # TODO :
    # average draining velocity and axial velocity scale recomputations / WIP

    def __init__(self) -> None:

        self._ndim_profile_polynom = Polynomial(
            np.zeros(FITTING_POLYNOM_DEGREE+1))
        self._ndim_profile_dotprod_polynom = Polynomial(
            np.zeros(DOTPROD_POLYNOM_DEGREE+1))

        self.ndim_reversal_radius: float

        self._axial_velocity_scale: float

        self._average_draining_velocity: float

    def radial_component(self, coordinates, flowsheet: FlowSheet, design: Design):

        Q_for = flowsheet.flow_rate[FlowSpec.FORWARD]

        radial_coordinate, axial_coordinate = coordinates

        axial_velocity = self.axial_component(coordinates, flowsheet, design)

        wall_radius = design.wall(axial_coordinate)
        wall_slope = design.wall_slope(axial_coordinate)

        # NOTE :
        # watch sign of slope, it might be already negative
        wall_induced_term = radial_coordinate/wall_radius*axial_velocity*wall_slope

        self._compute_draining_velocity(flowsheet, design)

        draininig_induced_term = (
            -self._average_draining_velocity *
            self.flow_reversal_point(coordinates, design)/radial_coordinate *
            2*np.pi/Q_for *
            self._ndim_profile_dotprod_polynom(1-radial_coordinate/wall_radius))

        velocity = wall_induced_term + draininig_induced_term

        return velocity

    def tangent_component(self, coordinates, flowsheet: FlowSheet, design: Design,
                          power: float = 0.8, imperfections_coeff: float = 0.8):

        radial_coordinate, _ = coordinates

        inlet_velocity = (
            flowsheet.flow_rate[FlowSpec.INLET] / design.inlet_area)
        velocity = (imperfections_coeff * inlet_velocity *
                    (2*design.wall(0)/radial_coordinate)**power)

        return velocity

    def axial_component(self, coordinates, flowsheet: FlowSheet, design: Design):

        radial_coordinate, axial_coordinate = coordinates

        wall_radius = design.wall(axial_coordinate)

        self._compute_axial_velocity_scale(coordinates, flowsheet, design)

        velocity = (self.ndim_profile(radial_coordinate/wall_radius) /
                    wall_radius**2 *
                    self._axial_velocity_scale)

        return velocity

    def drop_slip_velocity(self, drop_diameter):
        raise NotImplementedError()

    def flow_reversal_point(self, coordinates, design: Design):
        _, axial_coordinate = coordinates
        return self.ndim_reversal_radius*design.wall(axial_coordinate)

    def ndim_profile(self, ndim_radial_coordinate):
        return self._ndim_profile_polynom(ndim_radial_coordinate)

    def ndim_profile_radius_dotprod(self, ndim_radial_coordinate):
        return self._ndim_profile_dotprod_polynom(ndim_radial_coordinate)

    def ndim_profile_coeffs_residuals(self, flowsheet: FlowSheet):
        Q_for = flowsheet.flow_rate[FlowSpec.FORWARD]
        Q_rev = flowsheet.flow_rate[FlowSpec.REVERSE]

        theta1, theta2, theta3, theta4 = self._ndim_profile_polynom.coef
        ndim_RL = self.ndim_reversal_radius

        residuals = (
            theta1+2*theta3+3*theta4,
            theta2,
            self._ndim_profile_dotprod_polynom(1-ndim_RL) - Q_for/2/np.pi,
            self._ndim_profile_dotprod_polynom(ndim_RL) + Q_rev/2/np.pi,
            self._ndim_profile_polynom(ndim_RL))

        return residuals

    def solve_ndim_profile_coeffs(self, flowsheet: FlowSheet,
                                  enable_convergence_check: bool = False):
        self._solve_ndim_reversal_radius(flowsheet)
        self._restore_ndim_coeffs(flowsheet)

        if enable_convergence_check:
            sum_of_residuals = np.sum(
                np.abs(self.ndim_profile_coeffs_residuals(flowsheet)))
            if sum_of_residuals > SUM_RESIDUALS_TOLERANCE:
                raise ValueError("System of equations for nondimensional profile " +
                                 "coefficients did not converge")

    def _solve_ndim_reversal_radius(self, flowsheet: FlowSheet):
        self.ndim_reversal_radius = fsolve(
            lambda ndim_reversal_radius:
            self._ndim_reversal_radius_objective(
                ndim_reversal_radius, flowsheet),
            NDIM_REVRADIUS_INITIAL_GUESSE)[0]

    def _ndim_reversal_radius_objective(self, ndim_reversal_radius, flowsheet: FlowSheet):

        Q_u = flowsheet.flow_rate[FlowSpec.UNDERFLOW]
        Q_r = flowsheet.flow_rate[FlowSpec.RECIRCULATION]

        theta = (Q_u/np.pi - 7/np.pi*Q_r / (15*ndim_reversal_radius**4 -
                                            12*ndim_reversal_radius**5))

        residual = (theta + 30/7*(Q_u/np.pi - theta)*ndim_reversal_radius**2 -
                    20/7*(Q_u/np.pi-theta)*ndim_reversal_radius**3)

        return residual

    def _restore_ndim_coeffs(self, flowsheet: FlowSheet):

        Q_u = flowsheet.flow_rate[FlowSpec.UNDERFLOW]
        Q_r = flowsheet.flow_rate[FlowSpec.RECIRCULATION]

        theta1 = (Q_u/np.pi - 7/np.pi*Q_r / (15*self.ndim_reversal_radius**4 -
                                             12*self.ndim_reversal_radius**5))
        theta2 = 0.0
        theta3 = 30/7*(Q_u/np.pi - theta1)
        theta4 = -2/3*theta3

        self._ndim_profile_polynom.coef = np.array([theta1, 0, theta3, theta4])
        self._ndim_profile_dotprod_polynom.coef = np.array(
            [0, 0, theta1/2, theta2/3, theta3/4, theta4/5])

    def _compute_draining_velocity(self, flowsheet: FlowSheet, design: Design):

        Q_rev = flowsheet.flow_rate[FlowSpec.REVERSE]
        total_draining_area = design._wall_area_array[-1]
        self._average_draining_velocity = Q_rev / total_draining_area

    def _compute_axial_velocity_scale(
            self, coordinates, flowsheet: FlowSheet, design: Design):

        Q_rev = flowsheet.flow_rate[FlowSpec.REVERSE]
        Q_for = flowsheet.flow_rate[FlowSpec.FORWARD]

        _, axial_coordinate = coordinates

        # NOTE : this should do, but check anyways!
        total_length_integral = design._wall_area_array[-1]
        axial_coordinate_integral = np.interp(axial_coordinate,
                                              design._model_length_array,
                                              design._wall_area_array)

        self._axial_velocity_scale = 1-np.abs(
            Q_rev/Q_for*axial_coordinate_integral/total_length_integral)
