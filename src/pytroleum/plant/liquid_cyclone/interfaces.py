from typing import Protocol, runtime_checkable, overload
from numpy.typing import NDArray

from pytroleum.plant.liquid_cyclone.flowsheets import FlowSheet


@runtime_checkable
class Design(Protocol):

    characteristic_diameter: float
    inlet_diametet: float
    inlet_area: float

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

    def radial_component(self, coordinates: tuple, *args, **kwargs) -> NDArray:
        ...

    def tangent_component(self, coordinates: tuple, *args, **kwargs) -> NDArray:
        ...

    def axial_component(self, coordinates: tuple, *args, **kwargs) -> NDArray:
        ...

    def drop_slip_velocity(self, coordinates: tuple, drop_diameter: float) -> NDArray:
        ...
