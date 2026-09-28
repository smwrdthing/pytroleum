from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import numpy as np
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState
else:
    from CoolProp import AbstractState

type MappedDimension = np.ndarray
type MappedField = np.ndarray
type MappedFluidList = list[AbstractState]

type FluidIndex = int
type DimensionIndex = int
type EquationOfState = AbstractState
type PressureAndTemperature = tuple[float, float] | np.ndarray
type FluidIndices = Iterable[int]


@runtime_checkable
class Design(Protocol):
    diameter: MappedDimension
    area: MappedDimension
    length: MappedDimension
    angle: MappedDimension

    def throat(self, diameter: float) -> None: ...
    def nozzle(self, diameter: float) -> None: ...
    def area_ratio(self, of: DimensionIndex, to: DimensionIndex) -> float: ...


@runtime_checkable
class Constraints(Protocol):

    """Base class for operation conditions and requirements storage and management"""

    fluid: MappedFluidList

    flow_rate: MappedField
    pressure: MappedField
    temperature: MappedField
    density: MappedField
    velocity: MappedField
    Mach: MappedField

    def adopt_state_from(self, place: int, whose: FluidIndices) -> None: ...
    def record_state_to(self, place: int, whose: FluidIndices) -> None: ...


@runtime_checkable
class Requirements(Constraints, Protocol):

    """Dataclass to hold and manage field requirements for the design procedure
    of the ejector"""

    fluid: list[EquationOfState]
    primary_inflow_state: PressureAndTemperature
    secondary_inflow_state: PressureAndTemperature
    backpressure: float  # starred P_c in the paper


@runtime_checkable
class Conditions(Constraints, Protocol):

    """Dataclass to hold and manage fields describing specific operation condition
    in the ejector"""

    def adopt(self, requirements: Requirements) -> None: ...
    def flow_through(self, area: float, whose: FluidIndex,
                     efficiency: float) -> None: ...

    def heat_capactiy_ratio(self, whose: FluidIndex) -> float: ...
    def nozzle_mach_for(self, design: Design) -> None: ...
    def nozzle_pressure(self) -> None: ...
    def shock_pressure(self) -> None: ...
    def mixing_mach_for(self, deisng: Design) -> None: ...
