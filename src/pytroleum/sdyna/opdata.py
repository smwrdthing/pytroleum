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
    equation_of_state: list[AbstractState]

    # Thermodynamic parameters
    pressure: NDArray
    volume: NDArray
    temperature: NDArray
    density: NDArray
    energy_specific: NDArray
    molar_composition: NDArray

    # Fields for transport properties
    dynamic_vicosity: NDArray
    thermal_conductivity: NDArray


@dataclass
class StateData(OperationData):
    mass: NDArray
    energy: NDArray
    level: NDArray


@dataclass
class FlowData(OperationData):
    # This will be used for stream description in Conductors,
    # so we need elevation, velocity and specific energy of stream.
    velocity: NDArray
    energy_specific_flow: NDArray
    mass_flowrate: NDArray
    volume_flowrate: NDArray
    energy_flow: NDArray
    elevation: float = 0
