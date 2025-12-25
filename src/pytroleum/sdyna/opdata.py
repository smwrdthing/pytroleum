import numpy as np
import CoolProp.constants as CoolConst
from scipy.constants import g
from dataclasses import dataclass
from abc import ABC
from numpy.typing import NDArray
from numpy import float64
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from ..tdyna.CoolStub import AbstractState  # type: ignore
    # TODO : figure why pyright complains about stub file being
    #        impossible to resolve from source
else:
    from CoolProp.CoolProp import AbstractState

# NOTE & TODO
# NDArray is imposed by default to handle possible multiphase situations
#
# Following convention is introduced for  such cases:
# Data is stored in the density-ascending order for multiphase situations,
# so vapor data comes first, then lightest liquid and so on, until we reach
# heaviest liquid
#
# This should be mentioned in documentations/docstrings


@dataclass
class OperationData(ABC):
    equation_of_state: list[AbstractState]

    pressure: NDArray[float64]
    temperature: NDArray[float64]
    density: NDArray[float64]
    energy_specific: NDArray[float64]

    dynamic_viscosity: NDArray[float64]
    thermal_conductivity: NDArray[float64]


@dataclass
class StateData(OperationData):
    mass: NDArray[float64]
    energy: NDArray[float64]
    level: NDArray[float64]
    volume: NDArray[float64]


@dataclass
class FlowData(OperationData):
    velocity: NDArray[float64]
    energy_specific_flow: NDArray[float64]
    mass_flow_rate: NDArray[float64]
    volume_flowrate: NDArray[float64]
    energy_flow: NDArray[float64]
    elevation: float | float64 = 0


def _extract_temperature_dependent(
    equation_of_state: list[AbstractState],
    pressure: NDArray,
    temperature: NDArray,
    use_ideal_gas_specific_energy: bool = True
) -> tuple[NDArray[float64],
           NDArray[float64],
           NDArray[float64],
           NDArray[float64]]:

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


def factory_state(
        equation_of_state: list[AbstractState],
        volume_fn: Callable[[NDArray[float64]], NDArray[float64]],
        pressure: NDArray[float64],
        temperature: NDArray[float64],
        level: NDArray[float64],
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
        temperature,
        density,
        energy_specific,
        dynamic_viscosity,
        thermal_conductivity,
        mass,
        energy,
        level,
        volume
    )

    return state


def factory_flow(
        equation_of_state: list[AbstractState],
        pressure: NDArray[float64],
        temperature: NDArray[float64],
        flow_cross_area: float | float64,
        elevation: float | float64,
        mass_flowrate: NDArray[float64],
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
