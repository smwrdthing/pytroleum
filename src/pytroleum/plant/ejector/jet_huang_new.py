from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState  # type: ignore
else:
    from CoolProp import AbstractState
from CoolProp.constants import PT_INPUTS

import numpy as np
from scipy.optimize import fsolve
from enum import IntEnum


_R_UNIV = 8.314462618

# Huang et al. (1999), Eqs. (1), (5), (7)
PRIMARY_NOZZLE_EFFICIENCY = 0.95
CARRY_NOZZLE_EFFICIENCY = 0.85
PRIMARY_CORE_AREA_FACTOR = 0.88

# fsolve initial guess: supersonic branch of Eq. (2)
MACH_GUESS = 2.0


class Phase(IntEnum):
    """Stream indices (jet, carry, mixed)."""

    JET = 0
    CARRY = 1
    MIX = 2


class Loc(IntEnum):
    """Ejector section indices along the flow path (Huang et al., Fig. 2)."""

    INLET = 0
    THROAT = 1
    EXIT_NOZZLE = 2
    PREMIX = 3
    CHOKE = 4
    PRE_SHOCK = 5
    SHOCK = 6
    AFTERMIX = 7
    DRAIN = 8


_LAST_PHASE = len(Phase)
_LAST_LOC = len(Loc)
_SHAPE = (_LAST_PHASE, _LAST_LOC)
CONTAINER = np.full(_SHAPE, np.nan)

type Requirements = OperationConditions


@dataclass
class OperationConditions:

    phase: AbstractState
    mass_flow_rate: np.ndarray

    pressure: np.ndarray = field(default_factory=lambda: CONTAINER.copy())
    temperature: np.ndarray = field(default_factory=lambda: CONTAINER.copy())
    velocity: np.ndarray = field(default_factory=lambda: CONTAINER.copy())
    mach: np.ndarray = field(default_factory=lambda: CONTAINER.copy())

    def __post_init__(self) -> None:
        self.mass_flow_rate = np.array(
            [*self.mass_flow_rate, np.sum(self.mass_flow_rate)])

    def report(self) -> None:
        from jet_huang_report import report_conditions
        report_conditions(self)


@dataclass
class Design:
    """Ejector geometry from Huang et al. critical-mode analysis."""

    diameter: np.ndarray
    area: np.ndarray
    length: np.ndarray

    def report(self) -> None:
        from jet_huang_report import report_geometry
        report_geometry(self)


def _perfect_gas(
        eos: AbstractState, pressure: float, temperature: float,
) -> tuple[float, float, float]:
    """Constant gamma, R, cp at inlet state (Huang et al.)."""

    eos.update(PT_INPUTS, pressure, temperature)
    cp = eos.cpmass()
    gamma = cp / eos.cvmass()
    R = _R_UNIV / eos.molar_mass()
    return gamma, R, cp


def solve_dimensions(req: Requirements, design: Design) -> float:
    """Huang et al. (1999) critical-mode analysis, Eqs. (1)–(18), Fig. 3."""
    gamma, R, cp = _perfect_gas(
        req.phase,
        req.pressure[Phase.JET, Loc.INLET],
        req.temperature[Phase.JET, Loc.INLET],
    )
    # Primary flow through nozzle
    req.mass_flow_rate[Phase.JET] = (
        req.pressure[Phase.JET, Loc.INLET] *
        design.area[Phase.JET, Loc.THROAT] /
        np.sqrt(req.temperature[Phase.JET, Loc.INLET]) *
        np.sqrt(gamma / R * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))) *
        np.sqrt(PRIMARY_NOZZLE_EFFICIENCY))

    # Eq. (2)
    req.mach[Phase.JET, Loc.EXIT_NOZZLE] = fsolve(
        lambda x: (
            1.0 / x[0] ** 2 *
            (2.0 / (gamma + 1.0) * (1.0 + (gamma - 1.0) / 2.0 * x[0] ** 2)) **
            ((gamma + 1.0) / (gamma - 1.0)) -
            (design.area[Phase.JET, Loc.EXIT_NOZZLE] /
                design.area[Phase.JET, Loc.THROAT]) ** 2),
        [MACH_GUESS],
    )[0]

    return
