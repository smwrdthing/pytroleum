from scipy.constants import g
import numpy as np
from pytroleum.plant.ejector.inputs import (EjectingData,
                                            EjectedData,
                                            CommonParams)
UNIVERSAL_GAS_CONSTANT = 8.314  # Дж/(моль·К)
ATMOSPHERIC_PRESSURE = 1e5      # Па


def calculate_compression_ratio(ejected_data: EjectedData,
                                common_params: CommonParams) -> float:
    """Степень сжатия установки"""
    return common_params.outlet_pressure / ejected_data.inlet_pressure


def calculate_gas_constant(molecular_mass: float) -> float:
    """Газовая постоянная среды, Дж/(кг*K)"""
    return UNIVERSAL_GAS_CONSTANT / molecular_mass


def calculate_specific_heat_capacity(heat_capacity: float,
                                     molecular_mass: float) -> float:
    """Удельная теплоемкость среды, Дж/(кг·К)"""
    return heat_capacity / molecular_mass


def calculate_specific_weight(density: float) -> float:
    """Удельный вес, кг/(c2*м2) или H/м3 """
    return g * density


def calculate_critical_pressure_ratio(adiabatic_index: float) -> float:
    """Критическое отношение давлений"""
    return (2 / (adiabatic_index + 1)) ** (adiabatic_index / (adiabatic_index - 1))


def calculate_gas_outflow_velocity(mass_flow: float, temperature: float,
                                   pressure: float, diameter: float,
                                   molecular_mass: float) -> float:
    """Скорость истечения газа в газопроводе, м/c"""
    return (4 * mass_flow * calculate_gas_constant(molecular_mass) * temperature /
            ((pressure + ATMOSPHERIC_PRESSURE) * np.pi * diameter ** 2))
