import numpy as np
import CoolProp.constants as CoolConst
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Iterable, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..tdyna.CoolStub import AbstractState  # type: ignore
else:
    from CoolProp.CoolProp import AbstractState


@dataclass
class OperationData(ABC):
    # Maybe will be extended to store more

    # EOS interfaces for phases
    eos: list[AbstractState]

    # Thermodynamic parameters
    p: NDArray
    V: NDArray
    T: NDArray
    rho: NDArray
    u: NDArray
    x: NDArray

    # Fields for transport properties
    mu: NDArray
    lam: NDArray


@dataclass
class StateData(OperationData):
    m: NDArray
    E: NDArray
    h: NDArray


@dataclass
class FlowData(OperationData):
    # This will be used for stream description in Conductors,
    # so we need elevation, velocity and specific energy of stream.
    w: NDArray
    j: NDArray
    G: NDArray
    Q: NDArray
    J: NDArray
    z: float = 0
