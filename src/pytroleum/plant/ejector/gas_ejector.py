from scipy.constants import g
import numpy as np
from dataclasses import dataclass
from pytroleum.plant.ejector.inputs import (ActiveMediumData,
                                            PassiveMediumData,
                                            CommonParams)
from pytroleum.plant.ejector.equations import (
    calculate_gas_constant,
    calculate_specific_heat_capacity,
    calculate_specific_weight,
    calculate_gas_outflow_velocity,
    calculate_adiabatic_index,
    calculate_critical_pressure_ratio,
    calculate_critical_pressure,
    calculate_critical_temperature,
    calculate_section_temperature,
    calculate_circle_area,
    calculate_circle_diameter,
    calculate_reynolds_number,
    calculate_nozzle_throat_area,
    calculate_diffuser_length,
    calculate_section_area,
    calculate_section_velocity,
)


@dataclass
class EjectionParams:
    """Основные параметры эжекции"""
    compression_ratio: float
    entrainment_ratio: float
    critical_pressure_ratio: float
    m1: float
    m: float
    n: float


def calculate_ejection_params(active: ActiveMediumData,
                              passive: PassiveMediumData,
                              common_params: CommonParams) -> EjectionParams:
    """Основные параметры эжекции"""
    # Степень сжатия установки
    compression_ratio = common_params.outlet_pressure / passive.inlet_pressure

    # Коэфициент эжекции
    entrainment_ratio = passive.mass_flow / active.mass_flow

    # Показатель адиабаты
    adiabatic_index = calculate_adiabatic_index(
        active.molecular_mass, active.heat_capacity,
        passive.molecular_mass, passive.heat_capacity,
        entrainment_ratio)

    # Критическое отношение давлений
    critical_pressure_ratio = calculate_critical_pressure_ratio(
        adiabatic_index)

    # Основное уравнение эжекции для участка струи от сопла до места
    # соприкосновения со стенкой
    m1 = 2 * (1 + entrainment_ratio) ** 2

    # Основной геометрический параметр эжектора m
    m = m1 / (1 + (2 * entrainment_ratio ** 2) / m1)

    n = m / m1

    return EjectionParams(
        compression_ratio=compression_ratio,
        entrainment_ratio=entrainment_ratio,
        critical_pressure_ratio=critical_pressure_ratio,
        m1=m1, m=m, n=n)


@dataclass
class PressureParams:
    """Давления по сечениям эжектора"""
    dynamic_head_nozzle_exit: float
    ejector_head_no_diff: float
    critical: float
    cyl_section_exit: float
    ejector_outlet: float


def calculate_pressure_params(ejection: EjectionParams,
                              active: ActiveMediumData,
                              passive: PassiveMediumData,
                              mixer_exit_velocity: float,
                              mixture_density: float,
                              pressure_recovery_coefficient: float) -> PressureParams:
    """Давления по сечениям эжектора"""
    # Динамический напор эжектирующей струи на выходе из сопла (сечение I-I)
    dynamic_head_nozzle_exit = active.inlet_pressure / 1.1

    # Напор создаваемый эжектором без диффузора
    ejector_head_no_diff = dynamic_head_nozzle_exit / ejection.m

    # Давление в конце цилиндрического участка (сечение III-III)
    cyl_section_exit = ejector_head_no_diff - (-passive.inlet_pressure)

    # Давление в критическом сечении сопла
    critical = calculate_critical_pressure(
        ejection.critical_pressure_ratio, active.inlet_pressure)

    # Давление за диффузором
    ejector_outlet = (cyl_section_exit -
                      pressure_recovery_coefficient *
                      (calculate_specific_weight(mixture_density) +
                       mixer_exit_velocity ** 2) / (2 * g))

    return PressureParams(
        dynamic_head_nozzle_exit=dynamic_head_nozzle_exit,
        ejector_head_no_diff=ejector_head_no_diff,
        critical=critical,
        cyl_section_exit=cyl_section_exit,
        ejector_outlet=ejector_outlet)


@dataclass
class VelocityParams:
    """Скорости по сечениям эжектора"""
    active_inlet: float
    passive_inlet: float
    nozzle_exit: float
    cyl_section_exit: float
    ejector_outlet: float


def calculate_velocity_params(active: ActiveMediumData,
                              passive: PassiveMediumData,
                              nozzle_exit_pressure: float,
                              mixing_section_area: float,
                              mixture_density: float,
                              s: float) -> VelocityParams:
    """Скорости по сечениям эжектора"""
    # Скорость активной среды в трубопроводе w(a)
    active_inlet = calculate_gas_outflow_velocity(
        active.mass_flow, active.temperature,
        active.inlet_pressure, active.inlet_diameter,
        active.molecular_mass)

    # Скорость пассивной среды в трубопроводе w(n)
    passive_inlet = calculate_gas_outflow_velocity(
        passive.mass_flow, passive.temperature,
        passive.inlet_pressure, passive.inlet_diameter,
        passive.molecular_mass)

    # Скорость истечения газа из сопла (w1)
    nozzle_exit = np.sqrt(
        2 * g * nozzle_exit_pressure / calculate_specific_weight(active.density))

    # Скорость газа в конце смесительного участка (w3)
    cyl_section_exit = ((active.mass_flow + passive.mass_flow) /
                        (mixture_density * mixing_section_area))

    # Скорость на выходе из эжектора (w4)
    ejector_outlet = calculate_section_velocity(cyl_section_exit, s)

    return VelocityParams(
        active_inlet=active_inlet,
        passive_inlet=passive_inlet,
        nozzle_exit=nozzle_exit,
        cyl_section_exit=cyl_section_exit,
        ejector_outlet=ejector_outlet)


@dataclass
class TemperatureParams:
    """Температуры по сечениям эжектора"""
    nozzle_exit: float
    cyl_section_exit: float
    diffuser_exit: float


def calculate_temperature_params(active: ActiveMediumData,
                                 passive: PassiveMediumData,
                                 ejection: EjectionParams,
                                 pressure: PressureParams) -> TemperatureParams:
    """Температуры по сечениям эжектора"""
    # Показатель адиабаты
    adiabatic_index = calculate_adiabatic_index(
        active.molecular_mass, active.heat_capacity,
        passive.molecular_mass, passive.heat_capacity,
        ejection.entrainment_ratio)

    # Температура в критическом сечении сопла (T1)
    nozzle_exit = calculate_critical_temperature(
        active.temperature, ejection.critical_pressure_ratio, adiabatic_index)

    # Температура в конце цилиндрического участка (T3)
    cyl_section_exit = calculate_section_temperature(
        nozzle_exit, pressure.cyl_section_exit,
        pressure.critical, adiabatic_index)

    # Температура на выходе из диффузора (Т4)
    diffuser_exit = calculate_section_temperature(
        cyl_section_exit, pressure.ejector_outlet,
        pressure.cyl_section_exit, adiabatic_index)

    return TemperatureParams(
        nozzle_exit=nozzle_exit,
        cyl_section_exit=cyl_section_exit,
        diffuser_exit=diffuser_exit)


@dataclass
class GeometryParams:
    """Геометрические размеры эжектора"""
    # Площади сечений
    nozzle_exit_area: float
    nozzle_throat_area: float
    mixing_section_area: float
    diffuser_exit_area: float

    # Диаметры
    nozzle_exit_diameter: float
    nozzle_throat_diameter: float
    mixing_section_diameter: float

    # Осевые размеры
    jet_length: float
    mixing_section_length: float
    nozzle_to_inlet_distance: float
    cylinder_length: float
    diffuser_length: float


def calculate_geometry_params(ejection: EjectionParams,
                              active: ActiveMediumData,
                              common_params: CommonParams,
                              psi: float,
                              opening_angle: float,
                              s: float) -> GeometryParams:
    """Геометрические размеры эжектора"""
    # Площадь конечного сечения диффузора
    diffuser_exit_area = calculate_circle_area(common_params.outlet_diameter)

    # Площадь сечения цилиндрического смесительного участка
    mixing_section_area = calculate_section_area(diffuser_exit_area, s)

    # Диаметр сечения цилиндрического смесительного участка
    mixing_section_diameter = calculate_circle_diameter(mixing_section_area)

    # Площадь выходного сечения сопла
    nozzle_exit_area = calculate_section_area(mixing_section_area, ejection.m)

    # Диаметр выходного сечения сопла
    nozzle_exit_diameter = calculate_circle_diameter(nozzle_exit_area)

    # Площадь сечения узкой части сопла
    nozzle_throat_area = calculate_nozzle_throat_area(
        active.mass_flow, active.inlet_pressure, active.specific_volume, psi)

    # Диаметр сечения узкой части сопла
    nozzle_throat_diameter = calculate_circle_diameter(nozzle_throat_area)

    # Длина струи
    jet_length = nozzle_exit_diameter * \
        (4 * (1 + ejection.entrainment_ratio) - 1.8)

    # Длина смесительного участка
    mixing_section_length = 2.5 * mixing_section_diameter

    # Расстояние от сопла до начала цилиндрического участка L1
    nozzle_to_inlet_distance = jet_length - 0.5 * mixing_section_diameter

    # Длина цилиндрического участка L2
    cylinder_length = jet_length + mixing_section_length - nozzle_to_inlet_distance

    # Длина диффузора
    diffuser_length = calculate_diffuser_length(
        common_params.outlet_diameter, mixing_section_diameter, opening_angle)

    return GeometryParams(
        nozzle_exit_area=nozzle_exit_area,
        nozzle_throat_area=nozzle_throat_area,
        mixing_section_area=mixing_section_area,
        diffuser_exit_area=diffuser_exit_area,
        nozzle_exit_diameter=nozzle_exit_diameter,
        nozzle_throat_diameter=nozzle_throat_diameter,
        mixing_section_diameter=mixing_section_diameter,
        jet_length=jet_length,
        mixing_section_length=mixing_section_length,
        nozzle_to_inlet_distance=nozzle_to_inlet_distance,
        cylinder_length=cylinder_length,
        diffuser_length=diffuser_length)


@dataclass
class GasEjector:
    """Результаты расчёта газового эжектора"""
    ejection_params: EjectionParams
    pressure: PressureParams
    velocity: VelocityParams
    temperature: TemperatureParams
    geometry: GeometryParams
    reynolds_number: float


def calculate_gas_ejector(active: ActiveMediumData,
                          passive: PassiveMediumData,
                          common_params: CommonParams,
                          s: float,
                          mixture_density: float,
                          pressure_recovery_coefficient: float,
                          psi: float,
                          opening_angle: float,
                          mixture_dynamic_viscosity: float) -> GasEjector:
    """Полный расчёт газового эжектора"""
    ejection_params = calculate_ejection_params(active, passive, common_params)
    geometry = calculate_geometry_params(
        ejection_params, active, common_params, psi, opening_angle, s)
    velocity = calculate_velocity_params(
        active, passive,
        nozzle_exit_pressure=active.inlet_pressure / 1.1,
        mixing_section_area=geometry.mixing_section_area,
        mixture_density=mixture_density,
        s=s)
    pressure = calculate_pressure_params(
        ejection_params, active, passive,
        mixer_exit_velocity=velocity.cyl_section_exit,
        mixture_density=mixture_density,
        pressure_recovery_coefficient=pressure_recovery_coefficient)
    temperature = calculate_temperature_params(
        active, passive, ejection_params, pressure)

    # Расчет числа Re в нагнетательном трубопроводе, установленном за диффузором
    reynolds_number = calculate_reynolds_number(
        mixture_density, velocity.ejector_outlet,
        common_params.outlet_diameter, mixture_dynamic_viscosity)

    return GasEjector(
        ejection_params=ejection_params,
        pressure=pressure,
        velocity=velocity,
        temperature=temperature,
        geometry=geometry,
        reynolds_number=reynolds_number)


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
        mass_flow=36.25,                                    # кг/с
        temperature=248,                                    # К
        inlet_pressure=7e6,                                 # Па
        enthalpy=1045.91 * KCAL_TO_J,                      # ккал/кг → Дж/кг
        entropy=1.73 * KCAL_TO_J,                 # ккал/(кг·°С) → Дж/(кг·К)
        specific_volume=0.01,                               # м³/кг
        density=98.89,                                      # кг/м³
        dynamic_viscosity=0.00000099 * KGS_S_M2_TO_PA_S,  # кгс·с/м² → Па·с
        inlet_diameter=0.33,                                # м
        molecular_mass=18.70 * KG_PER_KMOL_TO_KG_PER_MOL,  # кг/кмоль → кг/моль
        # ккал/(кмоль·°С) → Дж/(моль·К)
        heat_capacity=16.23 * KCAL_PER_KMOL_TO_J_PER_MOL
    )

    # Пассивная среда (эжектируемая)
    passive = PassiveMediumData(
        mass_flow=6.52,                                      # кг/с
        temperature=289,                                     # К
        inlet_pressure=2.5e6,                                # Па
        enthalpy=927.89 * KCAL_TO_J,                        # ккал/кг → Дж/кг
        entropy=1.69 * KCAL_TO_J,                  # ккал/(кг·°С) → Дж/(кг·К)
        specific_volume=0.04,                                # м³/кг
        density=26.18,                                       # кг/м³
        dynamic_viscosity=0.00000116 * KGS_S_M2_TO_PA_S,   # кгс·с/м² → Па·с
        inlet_diameter=0.11,                                 # м
        molecular_mass=22.18 * KG_PER_KMOL_TO_KG_PER_MOL,  # кг/кмоль → кг/моль
        # ккал/(кмоль·°С) → Дж/(моль·К)
        heat_capacity=11.64 * KCAL_PER_KMOL_TO_J_PER_MOL
    )

    # Общие параметры
    common = CommonParams(
        num_stages=1,          # количество ступеней
        outlet_pressure=4e6,   # Па
        outlet_diameter=0.325  # м
    )

    ejector = calculate_gas_ejector(
        active, passive, common,
        s=2,
        mixture_density=56.05,
        pressure_recovery_coefficient=0.8,
        psi=2.14,
        opening_angle=8,
        mixture_dynamic_viscosity=0.0000012 * KGS_S_M2_TO_PA_S)

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

    R_active = calculate_gas_constant(active.molecular_mass)
    R_passive = calculate_gas_constant(passive.molecular_mass)
    Cp_active = calculate_specific_heat_capacity(
        active.heat_capacity, active.molecular_mass)
    Cp_passive = calculate_specific_heat_capacity(
        passive.heat_capacity, passive.molecular_mass)
    adiabatic_index = calculate_adiabatic_index(
        active.molecular_mass, active.heat_capacity,
        passive.molecular_mass, passive.heat_capacity,
        ejector.ejection_params.entrainment_ratio)

    _minor_header("ОСНОВНЫЕ ПАРАМЕТРЫ ЭЖЕКЦИИ")
    p("Степень сжатия:", f"{ejector.ejection_params.compression_ratio:.4f}")
    p("Коэффициент эжекции:",
      f"{ejector.ejection_params.entrainment_ratio:.4f}")
    p("m1 (участок струи до стенки):", f"{ejector.ejection_params.m1:.4f}")
    p("m (основной геометрический параметр):",
      f"{ejector.ejection_params.m:.4f}")
    p("n:", f"{ejector.ejection_params.n:.4f}")
    p("Напор эжектора без диффузора:",
      f"{ejector.pressure.ejector_head_no_diff/PA_TO_MPA:.3f}", "МПа")

    _minor_header("ГАЗОДИНАМИЧЕСКИЕ ПАРАМЕТРЫ")
    p("Газовая постоянная активной среды R_a:", f"{R_active:.2f}", "Дж/(кг·К)")
    p("Газовая постоянная пассивной среды R_n:",
      f"{R_passive:.2f}", "Дж/(кг·К)")
    p("Теплоемкость активной среды Cp_a:", f"{Cp_active:.2f}", "Дж/(кг·К)")
    p("Теплоемкость пассивной среды Cp_n:", f"{Cp_passive:.2f}", "Дж/(кг·К)")
    p("Показатель адиабаты k:", f"{adiabatic_index:.4f}")
    p("Критическое отношение давлений β:",
      f"{ejector.ejection_params.critical_pressure_ratio:.6f}")

    _minor_header("ДАВЛЕНИЯ")
    p("Динамический напор на выходе из сопла (I-I):",
      f"{ejector.pressure.dynamic_head_nozzle_exit/PA_TO_MPA:.3f}", "МПа")
    p("Напор эжектора без диффузора:",
      f"{ejector.pressure.ejector_head_no_diff/PA_TO_MPA:.3f}", "МПа")
    p("Критическое давление P_кр:",
      f"{ejector.pressure.critical/PA_TO_MPA:.3f}", "МПа")
    p("Давление в конце цилиндрического участка (III-III):",
      f"{ejector.pressure.cyl_section_exit/PA_TO_MPA:.3f}", "МПа")
    p("Давление за диффузором:",
      f"{ejector.pressure.ejector_outlet/PA_TO_MPA:.3f}", "МПа")

    _minor_header("СКОРОСТИ")
    p("Скорость активной среды в трубопроводе:",
      f"{ejector.velocity.active_inlet:.2f}", "м/с")
    p("Скорость пассивной среды в трубопроводе:",
      f"{ejector.velocity.passive_inlet:.2f}", "м/с")
    p("Скорость истечения газа из сопла (w1):",
      f"{ejector.velocity.nozzle_exit:.2f}", "м/с")
    p("Скорость газа в конце смесительного участка (w3):",
      f"{ejector.velocity.cyl_section_exit:.2f}", "м/с")
    p("Скорость на выходе из эжектора (w4):",
      f"{ejector.velocity.ejector_outlet:.2f}", "м/с")

    _minor_header("ТЕМПЕРАТУРЫ")
    p("Температура в критическом сечении сопла (t1):",
      f"{ejector.temperature.nozzle_exit:.2f} К "
      f"({ejector.temperature.nozzle_exit - KELVIN_TO_CELSIUS:.0f} °C)")
    p("Температура в конце цилиндрического участка (t3):",
      f"{ejector.temperature.cyl_section_exit:.2f} К "
      f"({ejector.temperature.cyl_section_exit - KELVIN_TO_CELSIUS:.0f} °C)")
    p("Температура на выходе из диффузора (t4):",
      f"{ejector.temperature.diffuser_exit:.2f} К "
      f"({ejector.temperature.diffuser_exit - KELVIN_TO_CELSIUS:.0f} °C)")

    _minor_header("ГЕОМЕТРИЧЕСКИЕ РАЗМЕРЫ")
    p("Площадь выходного сечения сопла F1:",
      f"{ejector.geometry.nozzle_exit_area:.4f}", "м²")
    p("Диаметр выходного сечения сопла D1:",
      f"{ejector.geometry.nozzle_exit_diameter * M_TO_MM:.2f}", "мм")
    p("Площадь узкой части сопла Fкр:",
      f"{ejector.geometry.nozzle_throat_area:.4f}", "м²")
    p("Диаметр узкой части сопла Dкр:",
      f"{ejector.geometry.nozzle_throat_diameter * M_TO_MM:.2f}", "мм")
    p("Площадь сечения смесительного участка F3:",
      f"{ejector.geometry.mixing_section_area:.4f}", "м²")
    p("Диаметр смесительного участка D3:",
      f"{ejector.geometry.mixing_section_diameter * M_TO_MM:.2f}", "мм")
    p("Площадь конечного сечения диффузора F4:",
      f"{ejector.geometry.diffuser_exit_area:.4f}", "м²")
    p("Длина струи Lx'':",
      f"{ejector.geometry.jet_length * M_TO_MM:.2f}", "мм")
    p("Длина смесительного участка Lсм:",
      f"{ejector.geometry.mixing_section_length * M_TO_MM:.2f}", "мм")
    p("Расстояние от сопла до цилиндрического участка L1:",
      f"{ejector.geometry.nozzle_to_inlet_distance * M_TO_MM:.2f}", "мм")
    p("Длина цилиндрического участка L2:",
      f"{ejector.geometry.cylinder_length * M_TO_MM:.2f}", "мм")
    p("Длина диффузора L3:",
      f"{ejector.geometry.diffuser_length * M_TO_MM:.2f}", "мм")

    _minor_header("НАГНЕТАТЕЛЬНЫЙ ТРУБОПРОВОД")
    p("Число Рейнольдса Re:", f"{ejector.reynolds_number:.0f}")
