from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field

from typing import Iterable
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
# relevent somewhere for some reason we will be explicit about it
_LAST_PHASE = M
_LAST_LOC = D
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


@dataclass
class Requirements:
    # For now both jet and carry phase EoS interfaces are assumed to be
    # for pure fluids, extension to mixtures is possible, but requieres
    # additional consideration
    phase: list[AbstractState]
    head: float
    carry_flow_rate: np.ndarray

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.phase[J].backend_name() != self.phase[C].backend_name():
            raise AttributeError(
                "Inconsistent backend for phases, for jet phase" +
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
            self.flow_rate[:MIX]/self.flow_rate[MIX])

    def isentropic_jump(self, to_loc, from_loc=LOBBY):

        # Jump to origin state with PT first
        entropy = []
        for idx, eos in enumerate(self.phase):
            eos.update(PT_INPUTS,
                       self.pressure[idx, from_loc],
                       self.temperature[idx, from_loc])
            entropy.append(eos.smass())

        # Jump to other location isentropically
        for idx, eos in enumerate(self.phase[:MIX]):
            eos.update(PSmass_INPUTS, self.pressure[to_loc], entropy[idx])

        # Mixture requires special treatment, because CoolProp would not allow
        # PSmass-jump for mixture backend, we must set up optimisation problems using
        # PT-jumps
        _isentropic_mixture_jump(
            self.phase[MIX], self.pressure[MIX, LOBBY], entropy[MIX])


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
            self.length[ANY, LOBBY])
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

    dp = conditions.pressure[MIX, AFTERMIX] - conditions.pressure[MIX, INLET]
    hj = conditions.velocity_head[JET, INLET]

    # ------------------------------ !WARNING! -------------------------------
    # we must ensure proper state for each phase before we read density values,
    # code below is incomplete

    # use eos to get density ratios
    jet_density = conditions.phase[J].rhomass()
    carry_density = conditions.phase[C].rhomass()
    mix_density = conditions.phase[M].rhomass()

    residual = (
        dp/hj - (2/m - 2/m**2 * ((1+q)**2 * (jet_density / mix_density) -
                                 q**2*n * (jet_density / carry_density))
                 )
    )

    return residual


def design(req: Requirements) -> Ejector:
    raise NotImplementedError("WIP")


def _isentropic_mixure_jump_residual(mix: AbstractState, p, T, smass):
    mix.update(PT_INPUTS, p, T)
    return mix.smass()-smass


def _isentropic_mixture_jump(mix: AbstractState, p, smass):
    T_original = mix.T()
    T_goal = fsolve(
        lambda T: _isentropic_mixure_jump_residual(mix, p, T, smass), T_original)[0]
    mix.update(PT_INPUTS, p, T_goal)
