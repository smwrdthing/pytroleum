from __future__ import annotations

from enum import IntEnum, auto
from typing import Any, Protocol

import numpy as np
from numpy import float64
from numpy.typing import NDArray


class Phase(IntEnum):
    """Stream indices (jet, carry, mixed)."""

    P, PRIMARY = 0, 0
    S, SECONDARY = 1, 1
    M, MIX = 2, 2

    SIZE = auto()


class Loc(IntEnum):
    """Ejector section indices along the flow path (Huang et al., Fig. 2)."""

    IN, INLET = 0, 0
    TH, THROAT = 1, 1
    EX, EXHAUST = 2, 2
    PM, PREMIX = 3, 3
    CH, CHOKE = 4, 4
    PS, PRE_SHOCK = 5, 5
    SH, SHOCK = 6, 6
    AM, AFTERMIX = 7, 7
    D, DRAIN = 8, 8

    SIZE = auto()


_CONTAINER = np.full((Phase.SIZE, Loc.SIZE), np.nan)


class Requirements(Protocol):
    phase: Any
    Pc_star: float
    pressure: NDArray[float64]
    temperature: NDArray[float64]


class OperationConditions(Protocol):
    mass_flow_rate: NDArray[float64]
    pressure: NDArray[float64]
    temperature: NDArray[float64]
    mach: NDArray[float64]
    velocity: NDArray[float64]


class Design(Protocol):
    diameter: NDArray[float64]
    area: NDArray[float64]
