import numpy as np
import CoolProp.constants as CoolConst
from scipy.constants import g
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from ..tdyna.CoolStub import AbstractState  # type: ignore
else:
    from CoolProp.CoolProp import AbstractState

type Numeric = float | NDArray


@dataclass
class OperationData(ABC):
    # Maybe will be extended to store more

    # EOS interfaces for phases
    equation_of_state: list[AbstractState]

    # Thermodynamic parameters
    pressure: NDArray
    temperature: NDArray
    density: NDArray
    energy_specific: NDArray

    # Fields for transport properties
    dynamic_viscosity: NDArray
    thermal_conductivity: NDArray


@dataclass
class StateData(OperationData):
    mass: NDArray
    energy: NDArray
    level: NDArray
    volume: NDArray


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


def _extract_temperature_dependent(
        equation_of_state: list[AbstractState],
        pressure: NDArray,
        temperature: NDArray,
        use_ideal_gas_specific_energy: bool = True
) -> tuple[NDArray, NDArray, NDArray, NDArray]:

    # Maybe move this functionality to tdyna later

    density = []
    energy_specific = []
    dynamic_viscosity = []
    thermal_conductivity = []
    for eos, p, T in zip(equation_of_state, pressure, temperature):
        eos.update(CoolConst.PT_INPUTS, p, T)
        density.append(eos.rhomass())
        if use_ideal_gas_specific_energy:
            energy_specific.append(eos.umass_idealgas())
        else:
            energy_specific.append(eos.umass())
        dynamic_viscosity.append(eos.viscosity())
        thermal_conductivity.append(eos.conductivity())
    density = np.array(density)
    energy_specific = np.array(energy_specific)
    dynamic_viscosity = np.array(dynamic_viscosity)
    thermal_conductivity = np.array(thermal_conductivity)

    return density, energy_specific, dynamic_viscosity, thermal_conductivity


def fabric_state(
        equation_of_state: list[AbstractState],
        volume_fn: Callable[[Numeric], Numeric],
        pressure: NDArray,
        temperature: NDArray,
        level: NDArray,
        use_ideal_gas_specific_energy: bool = True) -> StateData:

    (density,
     energy_specific,
     dynamic_viscosity,
     thermal_conductivity) = _extract_temperature_dependent(equation_of_state,
                                                            pressure,
                                                            temperature,
                                                            use_ideal_gas_specific_energy)

    # Levels should be in descending oreder because data is stored in density-ascending
    # order (from light to heavy phase). So when we use provided volume function we get
    # volume values that correspond to this order
    volume_by_level = volume_fn(level)
    # To get volume of matter we use diff with negative sign and appended zero
    volume = -np.diff(volume_by_level, append=0)

    mass = volume*density
    energy = mass*energy_specific

    state = StateData(
        equation_of_state,
        pressure,
        volume,
        temperature,
        density,
        energy_specific,
        dynamic_viscosity,
        thermal_conductivity,
        mass,
        energy,
        level
    )

    return state


def fabric_flow(
        equation_of_state: list[AbstractState],
        pressure: NDArray,
        temperature: NDArray,
        flow_cross_area: float,
        elevation: float,
        mass_flowrate: NDArray,
        use_ideal_gas_specific_energy: bool = True) -> FlowData:

    (density,
     energy_specific,
     dynamic_viscosity,
     thermal_conductivity) = _extract_temperature_dependent(equation_of_state,
                                                            pressure,
                                                            temperature,
                                                            use_ideal_gas_specific_energy)

    volume_flowrate = mass_flowrate/density
    velocity = mass_flowrate/density/flow_cross_area
    energy_specific_flow = (energy_specific+g*elevation +
                            pressure/density+velocity**2/2)
    energy_flow = energy_specific_flow*mass_flowrate

    flow = FlowData(
        equation_of_state,
        pressure,
        temperature,
        density,
        energy_specific,
        dynamic_viscosity,
        thermal_conductivity,
        velocity,
        energy_specific_flow,
        mass_flowrate,
        volume_flowrate,
        energy_flow,
        elevation
    )

    return flow


stopd = StateData(
    equation_of_state=[AbstractState("HEOS", "air")],
    pressure=np.array([1e5, 1e5, 1e5]),
    volume=np.array([1, 1, 1]),
    temperature=np.array([300, 300, 300]),
    density=np.array([1, 800, 1000]),
    dynamic_viscosity=np.array([1.1e-5, 6e-3, 1e-3]),
    energy=np.array([2e6, 10e6, 12e6]),
    mass=np.array([122, 1000, 755]),
    energy_specific=np.array([1.66e5, 1.22e3, 14e3]),
    level=np.array([2, 1, 0.6]),
    thermal_conductivity=np.array([1, 11, 1]))
