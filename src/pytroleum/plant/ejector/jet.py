from __future__ import annotations
from dataclasses import dataclass, field

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState  # type: ignore
else:
    from CoolProp import AbstractState
from CoolProp.constants import PT_INPUTS, PSmass_INPUTS

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
# relevant somewhere for some reason we will be explicit about it
_LAST_PHASE = M + 1
_LAST_LOC = D + 1
_SHAPE = (_LAST_PHASE, _LAST_LOC)
CONTAINER = np.full(_SHAPE, np.nan)

# Data is organised in matrix-like nature where phase indices correspond to rows
# and location indices correspond to columns
#
# So, for example, pressure of jet phase in entry would be accessed like follows:
# pressure[JET, INLET], for arbitrary container, phase and location in ejector it will be:
#                       <container>[<phase>, <location>]
#
# There are exceptions, like flow rate - in steady state flow rate will be same over
# all sections, so it is not relevant to keep flow rate in different locations

# For example purposes only, actual design procedure should not rely on imposed
# coefficients like that
JET_COEFF = 3


@dataclass
class Requirements:
    # For now both jet and carry phase EoS interfaces are assumed to be
    # for pure fluids, extension to mixtures is possible, but requires
    # additional consideration
    phase: list[AbstractState]
    head: float
    carry_flow_rate: float

    # Inlet conditions
    p_jet_inlet: float        # jet phase pressure at inlet [Pa]
    p_mix_aftermix: float     # mixture pressure after mixing zone [Pa]
    t_jet_inlet: float        # jet phase temperature at inlet [K]
    t_carry_inlet: float      # carry phase temperature at inlet [K]
    diameter_jet_inlet: float  # jet nozzle diameter at inlet [m]
    n: float                  # area ratio parameter [-], in range (0, 1]

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.phase[J].backend_name() != self.phase[C].backend_name():
            raise AttributeError(
                "Inconsistent backend for phases, for jet phase " +
                self.phase[J].backend_name() + " was provided while carry phase uses " +
                self.phase[C].backend_name())


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
        self.flow_rate = np.array([*self.flow_rate, np.sum(self.flow_rate)])

        # .fluid_names() always returns list, even for pure fluids, we take advantage of
        # that to build string for mixture
        self.phase.append(
            AbstractState(self.phase[J].backend_name(),
                          "&".join(self.phase[J].fluid_names() +
                                   self.phase[C].fluid_names())))
        self.phase[MIX].set_mass_fractions(
            self.flow_rate[:MIX] / self.flow_rate[MIX])

    def isentropic_jump(self, to_loc, from_loc):

        # Jump to origin state with PT first
        entropy = []
        for idx, eos in enumerate(self.phase):
            eos.update(PT_INPUTS,
                       self.pressure[idx, from_loc],
                       self.temperature[idx, from_loc])
            entropy.append(eos.smass())

        # Jump to other location isentropically
        for idx, eos in enumerate(self.phase[:MIX]):
            eos.update(PSmass_INPUTS, self.pressure[idx, to_loc], entropy[idx])

        # Mixture requires special treatment, because CoolProp would not allow
        # PSmass-jump for mixture backend, we must set up optimisation problems using
        # PT-jumps
        _isentropic_mixture_jump(
            self.phase[MIX], self.pressure[MIX, to_loc], entropy[MIX])


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
            0.5 * (self.diameter[ANY, LOBBY] - self.diameter[ANY, PREMIX]) /
            self.length[ANY, LOBBY])
        self.angle[:, DRAIN] = np.arctan(
            0.5 * (self.diameter[ANY, DRAIN] - self.diameter[ANY, AFTERMIX]) /
            self.length[ANY, DRAIN])

    def _infer_area(self):
        """Auxiliary method to compute areas of elements in ejector."""
        self.area = np.pi * self.diameter ** 2 / 4


def equation(m: float, n: float, conditions: OperationConditions) -> float:
    """Main ejector equation in implicit form derived from mass balance, suitable to set
    up optimization problem with appropriate wrapper.

    m -- area ratio: aftermix area / jet inlet area [-]
    n -- area ratio parameter [-], in range (0, 1]
    """

    # Equation is written in non-dimensional form
    q = conditions.flow_rate[CARRY] / conditions.flow_rate[JET]

    dp = conditions.pressure[MIX, AFTERMIX] - conditions.pressure[JET, INLET]
    hj = conditions.velocity_head[JET, INLET]

    # We want to read densities in the cross-section after mixing region, so we jump there
    # isentropically
    conditions.isentropic_jump(AFTERMIX)

    # Use EoS to get density ratios
    jet_density = conditions.phase[J].rhomass()
    carry_density = conditions.phase[C].rhomass()
    mix_density = conditions.phase[M].rhomass()

    residual = (
        dp / hj - (2 / m - 2 / m ** 2 * ((1 + q) ** 2 * (jet_density / mix_density) -
                                         q ** 2 * n * (jet_density / carry_density)))
    )

    return residual


def design(req: Requirements) -> float:
    """Function to find area ratio m that satisfies the ejector equation
    for given requirements.

    Returns m -- area ratio: aftermix area / jet inlet area [-]
    """

    flow_rates = np.ones(_LAST_PHASE - 1)  # [jet, carry]
    flow_rates[JET] = req.carry_flow_rate / JET_COEFF
    flow_rates[CARRY] = req.carry_flow_rate

    conditions = OperationConditions(req.phase, flow_rates)

    # Fill pressures
    conditions.pressure[JET, INLET] = req.p_jet_inlet
    conditions.pressure[MIX, AFTERMIX] = req.p_mix_aftermix

    # Fill temperatures
    conditions.temperature[JET, INLET] = req.t_jet_inlet
    conditions.temperature[CARRY, INLET] = req.t_carry_inlet

    # Get jet density at inlet via EoS
    conditions.phase[JET].update(PT_INPUTS, req.p_jet_inlet, req.t_jet_inlet)
    jet_density = conditions.phase[JET].rhomass()

    # Compute jet velocity at inlet from flow rate and nozzle area
    jet_area_inlet = np.pi * req.diameter_jet_inlet ** 2 / 4
    conditions.velocity[JET, INLET] = (
        conditions.flow_rate[JET] / (jet_density * jet_area_inlet)
    )

    # Compute velocity head for jet phase at inlet
    conditions.velocity_head[JET, INLET] = (
        jet_density * conditions.velocity[JET, INLET] ** 2 / 2
    )

    # Find m by solving the ejector equation
    m_solution = fsolve(
        lambda m: equation(m, req.n, conditions), x0=2.0
    )[0]

    return m_solution


def _isentropic_mixture_jump_residual(mix: AbstractState, p, T, smass):
    mix.update(PT_INPUTS, p, T)
    return mix.smass() - smass


def _isentropic_mixture_jump(mix: AbstractState, p, smass):
    T_original = mix.T()
    T_goal = fsolve(
        lambda T: _isentropic_mixture_jump_residual(mix, p, T, smass), T_original)[0]
    mix.update(PT_INPUTS, p, T_goal)


if __name__ == "__main__":

    # Флюиды
    jet_phase = AbstractState("HEOS", "Methane")
    carry_phase = AbstractState("HEOS", "Ethane")

    # Перевод давлений из мм вод. ст. в Па
    MM_H2O_TO_PA = 9.80665
    P_ATM = 101325.0

    p_jet_inlet = P_ATM - 5 * MM_H2O_TO_PA  # 101275.97 Па
    p_mix_aftermix = P_ATM + 5 * MM_H2O_TO_PA  # 101374.03 Па
    # dp = 10 мм вод. ст. = 98.07 Па

    req = Requirements(
        phase=[jet_phase, carry_phase],
        head=1.0,
        carry_flow_rate=1.0,           # кг/с
        p_jet_inlet=p_jet_inlet,
        p_mix_aftermix=p_mix_aftermix,
        t_jet_inlet=280.0,             # К, метан
        t_carry_inlet=290.0,           # К, этан
        diameter_jet_inlet=0.05,       # м
        n=0.5,                         # безразмерный параметр площадей [-]
    )

    m = design(req)
    print(f"m = {m:.4f}")
