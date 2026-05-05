from scipy.constants import g
import numpy as np
from dataclasses import dataclass
from pytroleum.plant.ejector.inputs import (ActiveMediumData,
                                            PassiveMediumData,
                                            CommonParams)
UNIVERSAL_GAS_CONSTANT = 8.314  # Дж/(моль·К)
ATMOSPHERIC_PRESSURE = 1e5      # Па
THERMAL_EQUIVALENT_OF_WORK = 0.982   # Дж/Дж


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


def calculate_gas_outflow_velocity(mass_flow: float, temperature: float,
                                   pressure: float, diameter: float,
                                   molecular_mass: float) -> float:
    """Скорость истечения газа в газопроводе, м/c"""
    return (4 * mass_flow * calculate_gas_constant(molecular_mass) * temperature /
            ((pressure + ATMOSPHERIC_PRESSURE) * np.pi * diameter ** 2))


@dataclass
class Ejector:
    compression_ratio: float
    entrainment_ratio: float
    critical_pressure: float
    critical_temperature: float


def operation_conditions(active: ActiveMediumData, passive: PassiveMediumData,
                         common_params: CommonParams) -> Ejector:
    # Степень сжатия установки
    compression_ratio = common_params.outlet_pressure / passive.inlet_pressure

    # Коэфициент эжекции
    entrainment_ratio = passive.mass_flow/active.mass_flow

    # Газовая постоянная R для активной и пассивной среды
    R_active = calculate_gas_constant(active.molecular_mass)
    R_passive = calculate_gas_constant(passive.molecular_mass)

    # Удельная теплоемкость Cp для активной и пассивной среды
    Cp_active = calculate_specific_heat_capacity(
        active.heat_capacity, active.molecular_mass)
    Cp_passive = calculate_specific_heat_capacity(
        passive.heat_capacity, passive.molecular_mass)

    # Показатель адиабаты
    adiabatic_index = 1/(1-THERMAL_EQUIVALENT_OF_WORK*(entrainment_ratio*R_passive+R_active) /
                         (Cp_passive*entrainment_ratio+Cp_active))

    # Критическое отношение давлений
    critical_pressure_ratio = (
        2/(adiabatic_index+1))**(adiabatic_index/(adiabatic_index-1))

    # Давление в критическом сечении сопла, Па
    critical_pressure = critical_pressure_ratio*active.inlet_pressure

    # Температура в критическом сечении сопла, К
    critical_temperature = active.temperature * \
        critical_pressure_ratio**((adiabatic_index-1)/adiabatic_index)

    return Ejector(compression_ratio=compression_ratio,
                   entrainment_ratio=entrainment_ratio,
                   critical_pressure=critical_pressure,
                   critical_temperature=critical_temperature)
