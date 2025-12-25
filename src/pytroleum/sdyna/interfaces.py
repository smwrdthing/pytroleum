from __future__ import annotations
from typing import Iterable, Protocol, runtime_checkable, overload
from numpy.typing import NDArray
from numpy import float64
from pytroleum.sdyna.opdata import StateData, FlowData
from pytroleum.sdyna.controllers import PropIntDiff, StartStop


@runtime_checkable
class ControlVolume(Protocol):

    # Interface for control volume

    outlets: list[Conductor]
    inlets: list[Conductor]
    state: StateData
    net_mass_flow: NDArray[float64]
    net_energy_flow: NDArray[float64]

    # def __init__(self) -> None:
    #     ...

    def connect_inlet(self, conductor: Conductor):
        ...

    def connect_outlet(self, conductor: Conductor):
        ...

    def advance(self) -> None:
        ...


@runtime_checkable
class Section(Protocol):

    # Needed for type checking in conductors with embedded distribution logic
    length_left_semiaxis: float | float64
    length_cylinder: float | float64
    length_right_semiaxis: float | float64
    diameter: float | float64
    outlets: list[Conductor]
    inlets: list[Conductor]
    state: StateData
    net_mass_flow: NDArray[float64]
    net_energy_flow: NDArray[float64]
    volume: float | float64
    level_graduated: NDArray[float64]
    volume_graduated: NDArray[float64]

    def __init__(self) -> None:
        ...

    def connect_inlet(self, conductor: Conductor):
        ...

    def connect_outlet(self, conductor: Conductor):
        ...

    @overload
    def compute_volume_with_level(self, level: float | float64) -> float | float64:
        ...

    @overload
    def compute_volume_with_level(self, level: NDArray[float64]) -> NDArray[float64]:
        ...

    def advance(self) -> None:
        ...


@runtime_checkable
class Conductor(Protocol):

    # Interface for conductor

    of_phase: int | Iterable[int]
    source: ControlVolume
    sink: ControlVolume
    flow: FlowData
    controller: PropIntDiff | StartStop | None

    def __init__(self, of_phase: int | Iterable[int],
                 source: ControlVolume | None,
                 sink: ControlVolume | None) -> None:
        ...

    def connect_source(self, convolume: ControlVolume) -> None:
        ...

    def connect_sink(self, convolume: ControlVolume) -> None:
        ...

    def advance(self) -> None:
        ...
