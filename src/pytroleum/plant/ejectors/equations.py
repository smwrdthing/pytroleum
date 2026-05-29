import numpy as np
from pytroleum.plant.ejectors.inputs import (OperationConditions,
                                             EOSInterface,
                                             ACTIVE, PASSIVE)
from scipy.constants import R as UNIVERSAL_GAS_CONSTANT

THERMAL_EQUIVALENT_OF_WORK = 0.982   # Дж/Дж


def calculate_gas_outflow_velocity(mass_flow: float, diameter: float,
                                   eos: EOSInterface) -> float:
    """Скорость истечения газа в газопроводе, м/с."""
    return mass_flow / eos.rhomass() / (np.pi * diameter ** 2 / 4)


def calculate_adiabatic_index(conditions: OperationConditions,
                              entrainment_ratio: float) -> float:
    """Показатель адиабаты смеси активной и пассивной сред."""
    R_active = UNIVERSAL_GAS_CONSTANT / conditions.phase[ACTIVE].molar_mass()
    R_passive = UNIVERSAL_GAS_CONSTANT / conditions.phase[PASSIVE].molar_mass()
    Cp_active = conditions.phase[ACTIVE].cpmass()
    Cp_passive = conditions.phase[PASSIVE].cpmass()

    # NOTE Тоже можно через CoolProp посчитать, в целом, если есть состав смеси
    # NOTE и температура с давлением известны

    return 1 / (1 - THERMAL_EQUIVALENT_OF_WORK *
                (entrainment_ratio * R_passive + R_active) /
                (Cp_passive * entrainment_ratio + Cp_active))


def calculate_critical_pressure_ratio(eos: EOSInterface) -> float:
    """Критическое отношение давлений."""
    k = eos.cpmass() / eos.cvmass()
    return (2 / (k + 1)) ** (k / (k - 1))


def calculate_section_temperature(inlet_temperature: float,
                                  outlet_pressure: float,
                                  inlet_pressure: float,
                                  adiabatic_index: float) -> float:
    """Температура в сечении через адиабатный процесс, К."""
    return (inlet_temperature *
            (outlet_pressure / inlet_pressure) ** ((adiabatic_index - 1) /
                                                   adiabatic_index))


def calculate_circle_area(diameter: float) -> float:
    """Площадь круглого сечения, м²."""
    return np.pi * diameter ** 2 / 4


def calculate_circle_diameter(area: float) -> float:
    """Диаметр круглого сечения, м."""
    return np.sqrt(4 * area / np.pi)


def calculate_reynolds_number(density: float,
                              velocity: float,
                              diameter: float,
                              dynamic_viscosity: float) -> float:
    """Число Рейнольдса."""
    return (density * velocity * diameter / dynamic_viscosity)


def calculate_nozzle_throat_area(conditions: OperationConditions,
                                 psi: float) -> float:
    """Площадь сечения узкой части сопла, м²."""
    # psi = 2,14 для газов и psi = 2,03 для перегретого и насыщенного
    # водяного пара
    return (conditions.mass_flow_rate[ACTIVE] /
            (psi * np.sqrt(conditions.pressure[ACTIVE] *
                           conditions.phase[ACTIVE].rhomass())))


def calculate_diffuser_length(diameter_exit: float, diameter_inlet: float,
                              opening_angle: float) -> float:
    """Длина диффузора, м."""
    # opening_angle — угол раскрытия диффузора, ° (рекомендовано 6°,
    # допустимо 2°–13°; при α > 14° поток не заполняет сечения равномерно,
    # усиливается вихреобразование вдоль стенок, возникают обратные токи,
    # коэффициент φ резко падает)
    return (diameter_exit - diameter_inlet) / (2 * np.tan(np.radians(opening_angle / 2)))


def calculate_section_area(area: float, ratio: float) -> float:
    """Площадь сечения через отношение площадей, м²."""
    # NOTE эта функция просто делит два чсила, можно без неё
    return area / ratio


def calculate_section_velocity(velocity: float, ratio: float) -> float:
    """Скорость потока через отношение площадей сечений, м/с."""
    # NOTE эта функция просто делит два чсила, можно без неё
    return velocity / ratio
