from scipy.constants import g
import numpy as np
from pytroleum.plant.ejectors.inputs import ActiveMediumData, PassiveMediumData

UNIVERSAL_GAS_CONSTANT = 8.314       # Дж/(моль·К)
ATMOSPHERIC_PRESSURE = 101325           # Па
THERMAL_EQUIVALENT_OF_WORK = 0.982   # Дж/Дж


def calculate_gas_constant(molecular_mass: float) -> float:
    """Газовая постоянная среды, Дж/(кг*K)"""
    return UNIVERSAL_GAS_CONSTANT / molecular_mass


def calculate_specific_heat_capacity(heat_capacity: float,
                                     molecular_mass: float) -> float:
    """Удельная теплоемкость среды, Дж/(кг·К)"""
    return heat_capacity / molecular_mass


def calculate_specific_weight(density: float) -> float:
    """Удельный вес, кг/(c2*м2)"""
    return g * density


def calculate_gas_outflow_velocity(mass_flow: float, temperature: float,
                                   pressure: float, diameter: float,
                                   molecular_mass: float) -> float:
    """Скорость истечения газа в газопроводе, м/c"""
    return (4 * mass_flow * calculate_gas_constant(molecular_mass) * temperature /
            ((pressure + ATMOSPHERIC_PRESSURE) * np.pi * diameter ** 2))


def calculate_adiabatic_index(active: ActiveMediumData,
                              passive: PassiveMediumData,
                              entrainment_ratio: float) -> float:
    """Показатель адиабаты смеси активной и пассивной сред"""
    R_active = calculate_gas_constant(active.molecular_mass)
    R_passive = calculate_gas_constant(passive.molecular_mass)
    Cp_active = calculate_specific_heat_capacity(
        active.heat_capacity, active.molecular_mass)
    Cp_passive = calculate_specific_heat_capacity(
        passive.heat_capacity, passive.molecular_mass)
    return 1 / (1 - THERMAL_EQUIVALENT_OF_WORK *
                (entrainment_ratio * R_passive + R_active) /
                (Cp_passive * entrainment_ratio + Cp_active))


def calculate_critical_pressure_ratio(adiabatic_index: float) -> float:
    """Критическое отношение давлений"""
    return (2 / (adiabatic_index + 1)) ** (adiabatic_index / (adiabatic_index - 1))


def calculate_critical_pressure(critical_pressure_ratio: float,
                                active_inlet_pressure: float) -> float:
    """Давление в критическом сечении сопла, Па"""
    return critical_pressure_ratio * active_inlet_pressure


def calculate_critical_temperature(active_temperature: float,
                                   critical_pressure_ratio: float,
                                   adiabatic_index: float) -> float:
    """Температура в критическом сечении сопла, К"""
    return active_temperature * critical_pressure_ratio ** ((adiabatic_index - 1) / adiabatic_index)


def calculate_section_temperature(inlet_temperature: float,
                                  outlet_pressure: float,
                                  inlet_pressure: float,
                                  adiabatic_index: float) -> float:
    """Температура в сечении через адиабатный процесс, К"""
    return (inlet_temperature *
            (outlet_pressure / inlet_pressure) ** ((adiabatic_index - 1) / adiabatic_index))


def calculate_circle_area(diameter: float) -> float:
    """Площадь круглого сечения, м²"""
    return np.pi * diameter ** 2 / 4


def calculate_circle_diameter(area: float) -> float:
    """Диаметр сечения, м"""
    return np.sqrt(4 * area / np.pi)


def calculate_reynolds_number(density: float,
                              velocity: float,
                              diameter: float,
                              dynamic_viscosity: float) -> float:
    """Число Рейнольдса Re = γ·w·D / (g·η)"""
    return (calculate_specific_weight(density) * velocity * diameter /
            (g * dynamic_viscosity))


def calculate_nozzle_throat_area(active: ActiveMediumData,
                                 psi: float) -> float:
    """Площадь сечения узкой части сопла, м²"""
    return active.mass_flow / (psi * np.sqrt(active.inlet_pressure / g / active.specific_volume))


def calculate_diffuser_length(diameter_exit: float, diameter_inlet: float,
                              opening_angle: float) -> float:
    """Длина диффузора, м"""
    return (diameter_exit - diameter_inlet) / (2 * np.tan(np.radians(opening_angle / 2)))


def calculate_section_area(area: float, ratio: float) -> float:
    """Площадь сечения через отношение площадей s=F4/F3, м²"""
    return area / ratio


def calculate_section_velocity(velocity: float, ratio: float) -> float:
    """Скорость потока через отношение площадей сечений, м/с"""
    return velocity / ratio
