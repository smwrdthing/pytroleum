from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field

from typing import Iterable
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState  # type: ignore
else:
    from CoolProp import AbstractState
from CoolProp.constants import PT_INPUTS, PSmass_INPUTS, iSmass, iP, iT

import numpy as np
from scipy.optimize import fsolve

# Indices with shortcuts to access data corresponding to specific location in ejector,
# if grows above 5-6 consider using Enum/IntEnum
I = INLET = 0
L = LOBBY = 1
PM = PREMIX = 2
AM = AFTERMIX = 3
D = DRAIN = 4

# Phase indices
J = JET = 0
C = CARRY = 1
M = MIX = 2
A = ANY = -1
# ANY is used for access when phase index is not relevant, when assigning a value
# slicing with ":" will result in imposing value for all phases

# Used for container initialization later on, nan is imposed, so when data is not
# relevent somewhere for some reason we will be explicit about it
_LAST_PHASE = M+1
_LAST_LOC = D+1
_SHAPE = (_LAST_PHASE, _LAST_LOC)
CONTAINER = np.full(_SHAPE, np.nan)

# Data is organize in matrix-like nature where phase indices correspond to rows
# and location indices correspond to columns
#
# So, for example, pressure of jet phase in entry would be accessed like follows:
# pressure[JET, ENTRY], for arbitrary container, phase and location in ejector it will be:
#                       <container>[<phase>, <location>]
#
# There are exceptions, like flow rate - in steady state flow rate will be same over
# all sections, so it is not relevant to keep flow rate in different locations

# For example purposes only, actual design procedure should not rely on imposed
# coefficients like that
JET_COEFF = 3

type Requirements = OperationConditions  # to separate semantically


@dataclass
class OperationConditions:

    phase: list[AbstractState]
    flow_rate: np.ndarray

    pressure: np.ndarray = field(default_factory=lambda: CONTAINER.copy())
    temperature: np.ndarray = field(default_factory=lambda: CONTAINER.copy())

    velocity: np.ndarray = field(default_factory=lambda: CONTAINER.copy())
    velocity_head: np.ndarray = field(default_factory=lambda: CONTAINER.copy())

    def __post_init__(self):
        self._setup_mixture()

    def _setup_mixture(self):
        """Auxiliary method to set up mixture EoS"""

        self._validate_eos()

        self.flow_rate = np.array([*self.flow_rate, np.sum(self.flow_rate)])

        # .fluid_names() always returns list, even for pure fluids, we take advantage of
        # that to build string for mixture
        self.phase.append(
            AbstractState(self.phase[J].backend_name(),
                          "&".join(self.phase[J].fluid_names() +
                                   self.phase[C].fluid_names())))
        self.phase[MIX].set_mass_fractions(
            self.flow_rate[:MIX]/self.flow_rate[MIX])

    def _validate_eos(self):
        if self.phase[J].backend_name() != self.phase[C].backend_name():
            raise AttributeError(
                "Inconsistent backend for phases, for jet phase" +
                self.phase[J].backend_name() + " was provided while carry phase uses " +
                self.phase[C].backend_name())


class Ejector:

    def __init__(self, diameter: np.ndarray, length: np.ndarray):

        if (diameter.shape != _SHAPE) or (length.shape != _SHAPE):
            raise ValueError(
                "Data containers with inconsistent shape were provided: " +
                f"shape {_SHAPE} is required, " +
                f"but diameter container with shape {diameter.shape} " +
                f"and length container with {length.shape} are given.")

        self.diameter = diameter
        self.length = length

        # Infer geometry from given data
        self._infer_angle()
        self._infer_area()

    def _infer_angle(self):
        """Auxiliary method to compute inclination angles of elements in ejector."""

        # here we take advantage of ejector 3-section structure,
        # more complex geometry would require more general approach
        self.angle = CONTAINER.copy()

        # Straight part, no incline
        self.angle[:, AFTERMIX] = self.angle[:, PREMIX] = 0

        # First & last section cones
        self.angle[:, LOBBY] = np.arctan(
            0.5*(self.diameter[ANY, LOBBY]-self.diameter[ANY, PREMIX]) /
            self.length[ANY, PREMIX])
        self.angle[:, DRAIN] = np.arctan(
            0.5*(self.diameter[ANY, DRAIN] - self.diameter[ANY, AFTERMIX]) /
            self.length[ANY, DRAIN])

    def _infer_area(self):
        """Auxiliary method to compute areas of elements in ejector."""
        self.area = np.pi*self.diameter**2/4


def equation(design: Ejector, conditions: OperationConditions):
    """Main ejector equation in implicit form derived from mass balance, suitable to set
    up optimization problem with appropriate wrapper"""

    # Equation is written in non-dimensional form

    m = design.area[ANY, AFTERMIX]/design.area[JET, INLET]
    n = design.area[ANY, AFTERMIX]/design.area[CARRY, INLET]
    q = conditions.flow_rate[CARRY]/conditions.flow_rate[JET]

    dp = conditions.pressure[MIX, AFTERMIX] - conditions.pressure[JET, INLET]
    hj = conditions.velocity_head[JET, INLET]

    # use equations of state to get density ratios

    # jet phase
    conditions.phase[JET].update(
        PT_INPUTS,
        conditions.pressure[JET, INLET],
        conditions.temperature[JET, INLET])
    jet_density = conditions.phase[JET].rhomass()

    # carried phase
    conditions.phase[CARRY].update(
        PT_INPUTS,
        conditions.pressure[CARRY, INLET],
        conditions.temperature[CARRY, INLET])
    carry_density = conditions.phase[CARRY].rhomass()

    # mixture

    # We need mixture density in the section after mixing. We jump here isentropically
    # from conditions in the lobby
    conditions.phase[MIX].update(
        PT_INPUTS,
        conditions.pressure[MIX, LOBBY],
        conditions.temperature[MIX, LOBBY])
    mix_entropy = conditions.phase[MIX].smass()
    _isentropic_mixture_jump(
        conditions.phase[MIX], conditions.pressure[MIX, AFTERMIX], mix_entropy)
    mix_density = conditions.phase[M].rhomass()

    residual = (
        dp/hj - (2/m - 2/m**2 * ((1+q)**2 * (jet_density / mix_density) -
                                 q**2*n * (jet_density / carry_density))
                 )
    )

    return residual


def report(design: Ejector, conditions: OperationConditions):
    print("         jet / carry / mix")
    print("diameters")
    print(
        f"Inlet : {design.diameter[J, I]*1e3} " +
        f"/ {design.diameter[C, I]*1e3} " +
        f"/ {design.diameter[M, I]*1e3} [mm]")
    print(
        f"Lobby : {design.diameter[J, L]*1e3} " +
        f"/ {design.diameter[C, L]*1e3} " +
        f"/ {design.diameter[M, L]*1e3} [mm]")
    print(
        f"Premix : {design.diameter[J, PM]*1e3} " +
        f"/ {design.diameter[C, PM]*1e3} " +
        f"/ {design.diameter[M, PM]*1e3} [mm]")
    print(
        f"Aftermix : {design.diameter[J, AM]*1e3} " +
        f"/ {design.diameter[C, AM]*1e3} " +
        f"/ {design.diameter[M, AM]*1e3} [mm]")
    print(
        f"Drain : {design.diameter[J, D]*1e3} " +
        f"/ {design.diameter[C, D]*1e3} " +
        f"/ {design.diameter[M, D]*1e3} [mm]")
    print()
    print("lengths")
    print(
        f"Inlet : {design.length[J, I]*1e3} " +
        f"/ {design.length[C, I]*1e3} " +
        f"/ {design.length[M, I]*1e3} [mm]")
    print(
        f"Lobby : {design.length[J, L]*1e3} " +
        f"/ {design.length[C, L]*1e3} " +
        f"/ {design.length[M, L]*1e3} [mm]")
    print(
        f"Premix : {design.length[J, PM]*1e3} " +
        f"/ {design.length[C, PM]*1e3} " +
        f"/ {design.length[M, PM]*1e3} [mm]")
    print(
        f"Aftermix : {design.length[J, AM]*1e3} " +
        f"/ {design.length[C, AM]*1e3} " +
        f"/ {design.length[M, AM]*1e3} [mm]")
    print(
        f"Drain : {design.length[J, D]*1e3} " +
        f"/ {design.length[C, D]*1e3} " +
        f"/ {design.length[M, D]*1e3} [mm]")
    print()
    print("flow rates")
    print(
        f"Inlet : {conditions.flow_rate[J]*1e3:.2f} " +
        f"/ {conditions.flow_rate[C]*1e3:.2f} " +
        f"/ {conditions.flow_rate[M]*1e3:.2f} [l/s]")
    print("CONTINUITY HOLDS")
    print()
    print("temperatures")
    print(
        f"Inlet : {conditions.temperature[J, I]-273.15:.2f} " +
        f"/ {conditions.temperature[C, I]-273.15} " +
        f"/ {conditions.temperature[M, I]-273.15} [C]")
    print(
        f"Lobby : {conditions.temperature[J, L]-273.15:.2f} " +
        f"/ {conditions.temperature[C, L]-273.15} " +
        f"/ {conditions.temperature[M, L]-273.15} [C]")
    print(
        f"Premix : {conditions.temperature[J, PM]-273.15:.2f} " +
        f"/ {conditions.temperature[C, PM]-273.15} " +
        f"/ {conditions.temperature[M, PM]-273.15} [C]")
    print(
        f"Aftermix : {conditions.temperature[J, AM]-273.15:.2f} " +
        f"/ {conditions.temperature[C, AM]-273.15} " +
        f"/ {conditions.temperature[M, AM]-273.15} [C]")
    print(
        f"Drain : {conditions.temperature[J, D]-273.15:.2f} " +
        f"/ {conditions.temperature[C, D]-273.15} " +
        f"/ {conditions.temperature[M, D]-273.15} [C]")
    print()
    print("pressure")
    print(
        f"Inlet : {conditions.pressure[J, I]/1e5} " +
        f"/ {conditions.pressure[C, I]/1e5} " +
        f"/ {conditions.pressure[M, I]/1e5} [bar]")
    print(
        f"Lobby : {conditions.pressure[J, L]/1e5} " +
        f"/ {conditions.pressure[C, L]/1e5} " +
        f"/ {conditions.pressure[M, L]/1e5} [bar]")
    print(
        f"Premix : {conditions.pressure[J, PM]/1e5} " +
        f"/ {conditions.pressure[C, PM]/1e5} " +
        f"/ {conditions.pressure[M, PM]/1e5} [bar]")
    print(
        f"Aftermix : {conditions.pressure[J, AM]/1e5} " +
        f"/ {conditions.pressure[C, AM]/1e5} " +
        f"/ {conditions.pressure[M, AM]/1e5} [bar]")
    print(
        f"Drain : {conditions.pressure[J, D]/1e5} " +
        f"/ {conditions.pressure[C, D]/1e5} " +
        f"/ {conditions.pressure[M, D]/1e5} [bar]")


def vary_jet_flow(
        G: np.ndarray, conditions: OperationConditions,
        design: Ejector) -> OperationConditions:
    """Auxiliary function to set up nonlinear solver"""
    G = np.atleast_1d(G)
    new_conditions = OperationConditions(
        phase=conditions.phase,
        flow_rate=np.array([G[0], conditions.flow_rate[C]]),
        pressure=conditions.pressure,
        temperature=conditions.temperature
    )

    new_conditions.phase[J].update(
        PT_INPUTS,
        new_conditions.pressure[J, I],
        new_conditions.temperature[J, I])
    rho_JI = new_conditions.phase[J].rhomass()

    new_conditions.velocity[J, I] = u_JI = G[0]/design.area[J, I]/rho_JI
    new_conditions.velocity_head[J, I] = rho_JI*u_JI**2/2

    return new_conditions


def solve_jet_flow_rate(
        design: Ejector, req: Requirements) -> OperationConditions:
    """Solve main ejector equation for flow rate with known requirements and design"""

    GJ = fsolve(
        lambda G: equation(design, vary_jet_flow(G, req, design)),
        x0=req.flow_rate[JET])[0]

    conditions = vary_jet_flow(GJ, req, design)

    return conditions


# same idea for dimensions
# def solve_dimensions(req: Requirements, ...):
    # """Solve main ejector equation for design parameters with known requirements"""
    # pass

# auxiliary functions ahead // should be moved to separate module


def _isentropic_mixure_jump_residual(
        mix: AbstractState, p: np.ndarray, T: np.ndarray, smass: float) -> float:
    p, T = np.atleast_1d(p), np.atleast_1d(T)  # type: ignore
    mix.update(PT_INPUTS, p[0], T[0])
    return mix.smass()-smass


def _isentropic_mixture_jump(mix: AbstractState, p: float, smass: float) -> AbstractState:

    T_original = mix.T()

    # Use linear approximation as an initial guesse
    p_original = mix.p()
    dSdP = mix.first_partial_deriv(iSmass, iP, iT)
    dSdT = mix.first_partial_deriv(iSmass, iT, iP)
    T_guesse = T_original + dSdP/dSdT * (p-p_original)

    T_goal = fsolve(
        lambda T: _isentropic_mixure_jump_residual(mix, p, T, smass), T_guesse)[0]

    mix.update(PT_INPUTS, p, T_goal)

    return mix


if __name__ == "__main__":
    # Dummy design parameters, just to run nonlinear solver
    # diameters
    ejector_diameters = CONTAINER.copy()

    ejector_diameters[JET, INLET] = 15e-3
    ejector_diameters[CARRY, INLET] = 40e-3

    ejector_diameters[:, LOBBY] = 42e-3
    ejector_diameters[:, PREMIX] = ejector_diameters[:, AFTERMIX] = 22e-3

    ejector_diameters[:, DRAIN] = 30e-3

    # lengths
    ejector_lengths = CONTAINER.copy()

    ejector_lengths[:, PREMIX] = 78e-3
    ejector_lengths[:, DRAIN] = 150e-3

    ejector_design = Ejector(ejector_diameters, ejector_lengths)

    # Phases and parameters sepcification
    jet_phase = AbstractState("HEOS", "N2")
    carry_phase = AbstractState("HEOS", "CH4")

    # To compute required flow rate of jet phase we must know dimensions and conditions
    # at inlets along with required carried phase flow rate
    carried_phase_flow_rate = 0.07
    requirements = OperationConditions(
        [jet_phase, carry_phase],
        np.array([carried_phase_flow_rate, carried_phase_flow_rate]))
    # we set jet phase flow rate to carried_phase_flow_rate initially

    # Specifying conditions at boundaries of ejector
    # Inlets
    requirements.pressure[:, INLET] = 3e5
    requirements.temperature[:, INLET] = 25+273.15

    # Head at the inlet must be specified too
    requirements.phase[J].update(
        PT_INPUTS, requirements.pressure[J, I], requirements.temperature[J, I])
    requirements.velocity[J, I] = (
        requirements.flow_rate[J] /
        requirements.phase[J].rhomass() /
        ejector_design.area[J, I])
    requirements.velocity_head[J, I] = (
        requirements.phase[J].rhomass()*requirements.velocity[J, I]**2/2)

    # Outlet
    requirements.pressure[:, AFTERMIX] = 1.1e5
    requirements.temperature[:, AFTERMIX] = 5+273.15

    # We also need to specify initial mixture state for isentropic jump, here we
    # assign JET phase parameters, generally we should apply some mixing rule based
    # on the thermodynamic states of components prior to mixing
    requirements.pressure[MIX, LOBBY] = requirements.pressure[JET, INLET]
    requirements.temperature[MIX, LOBBY] = requirements.temperature[JET, INLET]

    # Computing reuiered flow rate of jet phase
    conditions = solve_jet_flow_rate(ejector_design, requirements)

    report(ejector_design, conditions)
