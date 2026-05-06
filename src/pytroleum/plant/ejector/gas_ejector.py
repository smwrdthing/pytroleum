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


def calculate_adiabatic_index(entrainment_ratio: float,
                              R_active: float, R_passive: float,
                              Cp_active: float, Cp_passive: float) -> float:
    """Показатель адиабаты смеси активной и пассивной сред"""
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


def calculate_temperature_cyl_section_exit(critical_temperature: float,
                                           pressure_cyl_section_exit: float,
                                           critical_pressure: float,
                                           adiabatic_index: float) -> float:
    """Температура в конце цилиндрического участка, К"""
    return (critical_temperature *
            (pressure_cyl_section_exit / critical_pressure) ** (adiabatic_index - 1) / adiabatic_index)


@dataclass
class Ejector:
    compression_ratio: float
    entrainment_ratio: float
    adiabatic_index: float
    critical_pressure_ratio: float
    m1: float
    m: float
    n: float
    dynamic_head_nozzle_exit: float
    ejector_head_no_diff: float
    pressure_cyl_section_exit: float
    nozzle_exit_velocity: float


def operation_conditions(active: ActiveMediumData, passive: PassiveMediumData,
                         common_params: CommonParams) -> Ejector:
    # Степень сжатия установки
    compression_ratio = common_params.outlet_pressure / passive.inlet_pressure

    # Коэфициент эжекции
    entrainment_ratio = passive.mass_flow / active.mass_flow

    # Основное уравнение эжекции для участка струи от сопла до места
    # соприкосновения со стенкой
    m1 = 2 * (1 + entrainment_ratio) ** 2

    # Основной геометрический параметр эжектора m
    m = m1 / (1 + (2 * entrainment_ratio ** 2) / m1)

    n = m / m1

    # Газовая постоянная R для активной и пассивной среды
    R_active = calculate_gas_constant(active.molecular_mass)
    R_passive = calculate_gas_constant(passive.molecular_mass)

    # Удельная теплоемкость Cp для активной и пассивной среды
    Cp_active = calculate_specific_heat_capacity(
        active.heat_capacity, active.molecular_mass)
    Cp_passive = calculate_specific_heat_capacity(
        passive.heat_capacity, passive.molecular_mass)

    # Показатель адиабаты
    adiabatic_index = calculate_adiabatic_index(
        entrainment_ratio, R_active, R_passive, Cp_active, Cp_passive)

    # Критическое отношение давлений
    critical_pressure_ratio = calculate_critical_pressure_ratio(
        adiabatic_index)

    # Динамический напор эжектирующей струи на выходе из сопла (сечение I-I):
    dynamic_head_nozzle_exit = active.inlet_pressure / 1.1

    # Напор создаваемый эжектором без диффузора
    ejector_head_no_diff = dynamic_head_nozzle_exit / m

    # Давление в конце цилиндрического участка (сечение III-III):
    pressure_cyl_section_exit = ejector_head_no_diff + passive.inlet_pressure

    # Скорость истечения газа из сопла:
    nozzle_exit_velocity = np.sqrt(
        2 * g * dynamic_head_nozzle_exit / calculate_specific_weight(active.density))

    return Ejector(compression_ratio=compression_ratio,
                   entrainment_ratio=entrainment_ratio,
                   adiabatic_index=adiabatic_index,
                   critical_pressure_ratio=critical_pressure_ratio,
                   m1=m1,
                   m=m, n=n,
                   dynamic_head_nozzle_exit=dynamic_head_nozzle_exit,
                   ejector_head_no_diff=ejector_head_no_diff,
                   pressure_cyl_section_exit=pressure_cyl_section_exit,
                   nozzle_exit_velocity=nozzle_exit_velocity)

# ============================================================
# Пример использования
# ============================================================


if __name__ == '__main__':
    from pytroleum.plant.ejector.utils import (_major_header, _minor_header,
                                               print_row as p,
                                               PA_TO_MPA, KELVIN_TO_CELSIUS,
                                               KCAL_TO_J, KCAL_PER_KMOL_TO_J_PER_MOL,
                                               KGS_S_M2_TO_PA_S, KG_PER_KMOL_TO_KG_PER_MOL,
                                               KG_PER_MOL_TO_G_PER_MOL, J_TO_KJ, M_TO_MM)

# ============================================================
# Исходные данные
# ============================================================

    # Активная среда (эжектирующая)
    active = ActiveMediumData(
        mass_flow=36.25,  # кг/с
        temperature=248,  # К
        inlet_pressure=7e6,  # Па
        enthalpy=1045.91 * KCAL_TO_J,  # ккал/кг → Дж/кг
        entropy=1.73 * KCAL_TO_J,  # ккал/(кг·°С) → Дж/(кг·К)
        specific_volume=0.01,  # м³/кг
        density=98.89,  # кг/м³
        dynamic_viscosity=0.00000099 * KGS_S_M2_TO_PA_S,  # кгс·с/м² → Па·с
        inlet_diameter=0.33,  # м
        molecular_mass=18.70 * KG_PER_KMOL_TO_KG_PER_MOL,  # кг/кмоль → кг/моль
        # ккал/(кмоль·°С) → Дж/(моль·К)
        heat_capacity=16.23 * KCAL_PER_KMOL_TO_J_PER_MOL
    )

    # Пассивная среда (эжектируемая)
    passive = PassiveMediumData(
        mass_flow=6.52,  # кг/с
        temperature=289,  # К
        inlet_pressure=2.5e6,  # Па
        enthalpy=927.89 * KCAL_TO_J,  # ккал/кг → Дж/кг
        entropy=1.69 * KCAL_TO_J,  # ккал/(кг·°С) → Дж/(кг·К)
        specific_volume=0.04,  # м³/кг
        density=26.18,  # кг/м³
        dynamic_viscosity=0.00000116 * KGS_S_M2_TO_PA_S,  # кгс·с/м² → Па·с
        inlet_diameter=0.11,  # м
        molecular_mass=22.18 * KG_PER_KMOL_TO_KG_PER_MOL,  # кг/кмоль → кг/моль
        # ккал/(кмоль·°С) → Дж/(моль·К)
        heat_capacity=11.64 * KCAL_PER_KMOL_TO_J_PER_MOL
    )

    # Общие параметры
    common = CommonParams(
        num_stages=1,                      # количество ступеней
        outlet_pressure=4e6,               # Па
        outlet_diameter=0.325              # м
    )

    ejector = operation_conditions(active, passive, common)

# ============================================================
# Вывод результатов расчета эжектора
# ============================================================

    _major_header("ИСХОДНЫЕ ДАННЫЕ")

    _minor_header("АКТИВНАЯ СРЕДА (Эжектирующая)")
    p("Массовый расход:", f"{active.mass_flow:.2f}", "кг/с")
    p("Температура:",
      f"{active.temperature} К ({active.temperature - KELVIN_TO_CELSIUS:.0f} °C)")
    p("Давление на входе:", f"{active.inlet_pressure/PA_TO_MPA:.3f}", "МПа")
    p("Энтальпия:", f"{active.enthalpy/J_TO_KJ:.2f}", "кДж/кг")
    p("Энтропия:", f"{active.entropy:.2f}", "Дж/(кг·К)")
    p("Удельный объем:", f"{active.specific_volume:.3f}", "м³/кг")
    p("Плотность:", f"{active.density:.3f}", "кг/м³")
    p("Динамическая вязкость:", f"{active.dynamic_viscosity:.6f}", "Па·с")
    p("Диаметр трубопровода:", f"{active.inlet_diameter * M_TO_MM:.0f}", "мм")
    p("Молекулярная масса:",
      f"{active.molecular_mass * KG_PER_MOL_TO_G_PER_MOL:.1f}", "г/моль")
    p("Теплоемкость:", f"{active.heat_capacity/J_TO_KJ:.3f}", "кДж/(моль·К)")

    _minor_header("ПАССИВНАЯ СРЕДА (Эжектируемая)")
    p("Массовый расход:", f"{passive.mass_flow:.2f}", "кг/с")
    p("Температура:",
      f"{passive.temperature} К ({passive.temperature - KELVIN_TO_CELSIUS:.0f} °C)")
    p("Давление на входе:", f"{passive.inlet_pressure/PA_TO_MPA:.3f}", "МПа")
    p("Энтальпия:", f"{passive.enthalpy/J_TO_KJ:.2f}", "кДж/кг")
    p("Энтропия:", f"{passive.entropy:.2f}", "Дж/(кг·К)")
    p("Удельный объем:", f"{passive.specific_volume:.3f}", "м³/кг")
    p("Плотность:", f"{passive.density:.3f}", "кг/м³")
    p("Динамическая вязкость:", f"{passive.dynamic_viscosity:.6f}", "Па·с")
    p("Диаметр трубопровода:", f"{passive.inlet_diameter * M_TO_MM:.0f}", "мм")
    p("Молекулярная масса:",
      f"{passive.molecular_mass * KG_PER_MOL_TO_G_PER_MOL:.1f}", "г/моль")
    p("Теплоемкость:", f"{passive.heat_capacity/J_TO_KJ:.3f}", "кДж/(моль·К)")

    _minor_header("ОБЩИЕ ПАРАМЕТРЫ")
    p("Количество ступеней:", f"{common.num_stages}", "шт.")
    p("Давление на выходе:", f"{common.outlet_pressure/PA_TO_MPA:.3f}", "МПа")
    p("Диаметр выходного трубопровода:",
      f"{common.outlet_diameter * M_TO_MM:.0f}", "мм")

    _major_header("РЕЗУЛЬТАТЫ РАСЧЕТА ЭЖЕКТОРА")

    _minor_header("ОСНОВНЫЕ ПАРАМЕТРЫ")
    p("Степень сжатия:", f"{ejector.compression_ratio:.4f}")
    p("Коэффициент эжекции:", f"{ejector.entrainment_ratio:.4f}")

    R_active = calculate_gas_constant(active.molecular_mass)
    R_passive = calculate_gas_constant(passive.molecular_mass)
    Cp_active = calculate_specific_heat_capacity(
        active.heat_capacity, active.molecular_mass)
    Cp_passive = calculate_specific_heat_capacity(
        passive.heat_capacity, passive.molecular_mass)
    critical_pressure = calculate_critical_pressure(
        ejector.critical_pressure_ratio, active.inlet_pressure)
    critical_temperature = calculate_critical_temperature(
        active.temperature, ejector.critical_pressure_ratio, ejector.adiabatic_index)
    temperature_cyl_exit = calculate_temperature_cyl_section_exit(
        critical_temperature, ejector.pressure_cyl_section_exit, critical_pressure, ejector.adiabatic_index)

    _minor_header("ГАЗОДИНАМИЧЕСКИЕ ПАРАМЕТРЫ")
    p("Газовая постоянная активной среды R_a:", f"{R_active:.2f}", "Дж/(кг·К)")
    p("Газовая постоянная пассивной среды R_n:",
      f"{R_passive:.2f}", "Дж/(кг·К)")
    p("Теплоемкость активной среды Cp_a:", f"{Cp_active:.2f}", "Дж/(кг·К)")
    p("Теплоемкость пассивной среды Cp_n:", f"{Cp_passive:.2f}", "Дж/(кг·К)")
    p("Показатель адиабаты k:", f"{ejector.adiabatic_index:.4f}")
    p("Критическое отношение давлений β:",
      f"{ejector.critical_pressure_ratio:.6f}")

    _minor_header("КРИТИЧЕСКИЕ ПАРАМЕТРЫ СОПЛА")
    p("Критическое давление P_кр:",
      f"{critical_pressure/PA_TO_MPA:.3f}", "МПа")
    p("Критическая температура T_кр:",
      f"{critical_temperature:.2f} К ({critical_temperature - KELVIN_TO_CELSIUS:.0f} °C)")

    _minor_header("ГЕОМЕТРИЧЕСКИЕ ПАРАМЕТРЫ")
    p("m1 (участок струи до стенки):", f"{ejector.m1:.4f}")
    p("m (основной геометрический параметр):", f"{ejector.m:.4f}")
    p("n:", f"{ejector.n:.4f}")

    _minor_header("ГАЗОДИНАМИЧЕСКИЕ ПАРАМЕТРЫ ПО СЕЧЕНИЯМ")
    p("Динамический напор на выходе из сопла (I-I):",
      f"{ejector.dynamic_head_nozzle_exit/PA_TO_MPA:.3f}", "МПа")
    p("Напор эжектора без диффузора:",
      f"{ejector.ejector_head_no_diff/PA_TO_MPA:.3f}", "МПа")
    p("Давление в конце цилиндрического участка (III-III):",
      f"{ejector.pressure_cyl_section_exit/PA_TO_MPA:.3f}", "МПа")
    p("Температура в конце цилиндрического участка (III-III):",
      f"{temperature_cyl_exit:.2f} К ({temperature_cyl_exit - KELVIN_TO_CELSIUS:.0f} °C)")
    p("Скорость истечения газа из сопла:",
      f"{ejector.nozzle_exit_velocity:.2f}", "м/с")

    _minor_header("СКОРОСТИ ПОТОКОВ")
    p("Скорость активной среды в трубопроводе:",
      f"{calculate_gas_outflow_velocity(active.mass_flow,
                                        active.temperature,
                                        active.inlet_pressure,
                                        active.inlet_diameter,
                                        active.molecular_mass):.2f}", "м/с")
    p("Скорость пассивной среды в трубопроводе:",
      f"{calculate_gas_outflow_velocity(passive.mass_flow,
                                        passive.temperature,
                                        passive.inlet_pressure,
                                        passive.inlet_diameter,
                                        passive.molecular_mass):.2f}", "м/с")

    _major_header("РАСЧЕТ ЗАВЕРШЕН")
