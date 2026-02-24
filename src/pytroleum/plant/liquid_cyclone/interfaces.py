from typing import Protocol, runtime_checkable, overload
from numpy.typing import NDArray
from numpy.polynomial import Polynomial
from pytroleum.tdyna import eos

type OptionalPhase = eos.AbstractState | eos.AbstractStateImitator


@runtime_checkable
class FlowSheet(Protocol):

    light_phase_eos: OptionalPhase
    heavy_phase_eos: OptionalPhase

    resistance: list[float]
    pressure: list[float]
    flow_rate: list[float]


@runtime_checkable
class Design(Protocol):

    characteristic_diameter: float
    inlet_diameter: float
    inlet_area: float
    model_length: float

    _model_length_array: NDArray
    _wall_area_array: NDArray

    def wall(self, axial_coordinate: NDArray | float) -> NDArray:
        ...

    def wall_slope(self, axial_coordinate: NDArray | float) -> NDArray:
        ...

    def wall_incline(self, axial_coordinate: NDArray | float) -> NDArray:
        ...


@runtime_checkable
class VelocityField(Protocol):

    ndim_reversal_radius: float
    _ndim_profile_polynom: Polynomial
    _ndim_profile_dotprod_polynom: Polynomial

    def radial_component(self, coordinates: tuple, *args, **kwargs) -> NDArray:
        ...

    def tangent_component(self, coordinates: tuple, *args, **kwargs) -> NDArray:
        ...

    def axial_component(self, coordinates: tuple, *args, **kwargs) -> NDArray:
        ...

    def drop_slip_velocity(
            self, coordinates: tuple, drop_diameter: float,
            flowsheet: FlowSheet, design: Design) -> NDArray:
        ...
