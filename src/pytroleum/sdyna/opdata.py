from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterable, Optional
from numpy.typing import NDArray, DTypeLike
import numpy as np
from scipy.constants import R

# Listing of some standard temperature and pressure values.
# Maybe move to other place when library is more mature
STP_IUPAC82 = {"T": 0+273.15, "p": 100_000}
STP_NIST = {"T": 0+273.15, "p": 101_325}
STP_SPE = {"T": 15+273.15, "p": 100_000}
STP_ISO13443 = {"T": 15+273.15, "p": 101_325}
STP_GOST2936 = {"T": 20+273.15, "p": 101_330}


@dataclass
class OperationData(ABC):
    # Maybe will be extended to store more

    # EOS interfaces for phases
    eos: Iterable

    # Thermodynamic parameters
    p: Iterable[float]
    V: Iterable[float]
    T: Iterable[float]
    rho: Iterable[float]
    u: Iterable[float]

    # For potential VLE
    x_all: Iterable[float]
    x_vap: Iterable[float]
    x_liq: Iterable[float]
    Q_vap: Iterable[float]

    # Fields for transport properties
    mu: Iterable[float]
    lam: Iterable[float]


@dataclass
class StateData(OperationData):
    # This will be used for state description in ControlVolumes,
    # so mass and energy are of concerns
    m: Iterable[float]
    E: Iterable[float]
    # Levels might be presented
    h: Optional[Iterable[float]]


@dataclass
class FlowData(OperationData):
    # This will be used for stream description in Conductors,
    # so we need elevation, velocity and specific energy of stream.
    z: Iterable[float]
    w: Iterable[float]
    j: Iterable[float]
    G: Iterable[float]
    Q: Optional[Iterable[float]]
    J: Optional[Iterable[float]]  # technically j*G
