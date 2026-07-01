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


_M_TO_MM = 1e3
_M2_TO_CM2 = 1e4
_PA_TO_BAR = 1e-5
_K_TO_C = 273.15
_R_UNIV = 8.314462618

# Huang et al. (1999), Eqs. (1), (5), (7)
PRIMARY_NOZZLE_DISCHARGE_COEFF = 0.95
CARRY_NOZZLE_DISCHARGE_COEFF = 0.85
PRIMARY_CORE_AREA_FACTOR = 0.88


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

    phase: list[AbstractState]
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
    """Ejector geometry"""

    diameter: np.ndarray
    area: np.ndarray
    length: np.ndarray
