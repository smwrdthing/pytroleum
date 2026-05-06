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
    calculate_temperature_cyl_section_exit,
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
    adiabatic_index: float
    critical_pressure_ratio: float
    m1: float
    m: float
    n: float
    dynamic_head_nozzle_exit: float
    ejector_head_no_diff: float
    pressure_cyl_section_exit: float
    nozzle_exit_velocity: float


@dataclass
class NozzleGeometry:
    """Геометрия сопла"""
    exit_area: float
    exit_diameter: float
    throat_area: float
    throat_diameter: float


@dataclass
class MixerGeometry:
    """Геометрия смесительного участка"""
    section_area: float
    section_diameter: float
    exit_velocity: float
    jet_length: float
    section_length: float
    nozzle_to_inlet_distance: float
    cylinder_length: float


@dataclass
class DiffuserGeometry:
    """Геометрия диффузора"""
    exit_area: float
    length: float
    pressure_after: float


@dataclass
class PipelineParams:
    """Параметры нагнетательного трубопровода"""
    velocity: float
    reynolds_number: float


@dataclass
class GasEjector:
    """Результаты расчёта газового эжектора"""
    ejection: EjectionParams
    nozzle: NozzleGeometry
    mixer: MixerGeometry
    diffuser: DiffuserGeometry
    pipeline: PipelineParams


def calculate_ejection_params(active: ActiveMediumData,
                              passive: PassiveMediumData,
                              common_params: CommonParams) -> EjectionParams:
    """Основные параметры эжекции: степень сжатия, адиабата, напоры, давления"""
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

    # Динамический напор эжектирующей струи на выходе из сопла (сечение I-I)
    dynamic_head_nozzle_exit = active.inlet_pressure / 1.1

    # Напор создаваемый эжектором без диффузора
    ejector_head_no_diff = dynamic_head_nozzle_exit / m

    # Давление в конце цилиндрического участка (сечение III-III)
    pressure_cyl_section_exit = ejector_head_no_diff + passive.inlet_pressure

    # Скорость истечения газа из сопла
    nozzle_exit_velocity = np.sqrt(
        2 * g * dynamic_head_nozzle_exit / calculate_specific_weight(active.density))

    return EjectionParams(
        compression_ratio=compression_ratio,
        entrainment_ratio=entrainment_ratio,
        adiabatic_index=adiabatic_index,
        critical_pressure_ratio=critical_pressure_ratio,
        m1=m1, m=m, n=n,
        dynamic_head_nozzle_exit=dynamic_head_nozzle_exit,
        ejector_head_no_diff=ejector_head_no_diff,
        pressure_cyl_section_exit=pressure_cyl_section_exit,
        nozzle_exit_velocity=nozzle_exit_velocity)


def calculate_nozzle(ejection: EjectionParams,
                     active: ActiveMediumData,
                     mixer: 'MixerGeometry',
                     psi: float) -> NozzleGeometry:
    """Геометрия сопла: площади и диаметры выходного и критического сечений"""
    # Площадь выходного сечения сопла
    exit_area = calculate_section_area(mixer.section_area, ejection.m)

    # Диаметр выходного сечения сопла
    exit_diameter = calculate_circle_diameter(exit_area)

    # Площадь сечения узкой части сопла
    throat_area = calculate_nozzle_throat_area(
        active.mass_flow, active.inlet_pressure, active.specific_volume, psi)

    # Диаметр сечения узкой части сопла
    throat_diameter = calculate_circle_diameter(throat_area)

    return NozzleGeometry(
        exit_area=exit_area,
        exit_diameter=exit_diameter,
        throat_area=throat_area,
        throat_diameter=throat_diameter)


def calculate_mixer(ejection: EjectionParams,
                    active: ActiveMediumData,
                    passive: PassiveMediumData,
                    diffuser_exit_area: float,
                    mixture_density: float,
                    s: float) -> MixerGeometry:
    """Геометрия смесительного участка: сечения, скорость, продольные размеры"""
    # Площадь сечения цилиндрического смесительного участка
    section_area = calculate_section_area(diffuser_exit_area, s)

    # Диаметр сечения цилиндрического смесительного участка
    section_diameter = calculate_circle_diameter(section_area)

    # Скорость газа в конце смесительного участка
    exit_velocity = ((active.mass_flow + passive.mass_flow) /
                     (mixture_density * section_area))

    # Длина струи
    jet_length = calculate_circle_diameter(
        calculate_section_area(section_area, ejection.m)
    ) * (4 * (1 + ejection.entrainment_ratio) - 1.8)

    # Длина смесительного участка
    section_length = 2.5 * section_diameter

    # Расстояние от сопла до начала цилиндрического участка L1
    nozzle_to_inlet_distance = (
        jet_length + section_length) - 0.5 * section_diameter

    # Длина цилиндрического участка L2
    cylinder_length = 3 * section_diameter

    return MixerGeometry(
        section_area=section_area,
        section_diameter=section_diameter,
        exit_velocity=exit_velocity,
        jet_length=jet_length,
        section_length=section_length,
        nozzle_to_inlet_distance=nozzle_to_inlet_distance,
        cylinder_length=cylinder_length)


def calculate_diffuser(ejection: EjectionParams,
                       mixer: MixerGeometry,
                       common_params: CommonParams,
                       diffuser_exit_area: float,
                       mixture_density: float,
                       pressure_recovery_coefficient: float,
                       opening_angle: float) -> DiffuserGeometry:
    """Геометрия диффузора: площадь, длина, давление за диффузором"""
    # Длина диффузора
    length = calculate_diffuser_length(
        common_params.outlet_diameter, mixer.section_diameter, opening_angle)

    # Давление за диффузором
    pressure_after = (ejection.pressure_cyl_section_exit +
                      pressure_recovery_coefficient *
                      (calculate_specific_weight(mixture_density) *
                       mixer.exit_velocity ** 2) / (2 * g))

    return DiffuserGeometry(
        exit_area=diffuser_exit_area,
        length=length,
        pressure_after=pressure_after)


def calculate_pipeline(mixer: MixerGeometry,
                       common_params: CommonParams,
                       mixture_density: float,
                       mixture_dynamic_viscosity: float,
                       s: float) -> PipelineParams:
    """Параметры нагнетательного трубопровода: скорость и число Рейнольдса"""
    # Скорость движения потока за диффузором
    velocity = calculate_section_velocity(mixer.exit_velocity, s)

    # Расчет числа Re в нагнетательном трубопроводе, установленном за диффузором
    reynolds_number = calculate_reynolds_number(
        mixture_density, velocity,
        common_params.outlet_diameter, mixture_dynamic_viscosity)

    return PipelineParams(velocity=velocity, reynolds_number=reynolds_number)


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
    ejection = calculate_ejection_params(active, passive, common_params)

    # Площадь конечного сечения диффузора — нужна раньше calculate_mixer
    diffuser_exit_area = calculate_circle_area(common_params.outlet_diameter)

    mixer = calculate_mixer(
        ejection, active, passive, diffuser_exit_area, mixture_density, s)
    nozzle = calculate_nozzle(ejection, active, mixer, psi)
    diffuser = calculate_diffuser(
        ejection, mixer, common_params, diffuser_exit_area,
        mixture_density, pressure_recovery_coefficient, opening_angle)
    pipeline = calculate_pipeline(
        mixer, common_params, mixture_density, mixture_dynamic_viscosity, s)

    return GasEjector(
        ejection=ejection,
        nozzle=nozzle,
        mixer=mixer,
        diffuser=diffuser,
        pipeline=pipeline)


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

    ejector = calculate_gas_ejector(
        active, passive, common,
        mixture_density=56.05,
        s=2,
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
    critical_pressure = calculate_critical_pressure(
        ejector.ejection.critical_pressure_ratio, active.inlet_pressure)
    critical_temperature = calculate_critical_temperature(
        active.temperature, ejector.ejection.critical_pressure_ratio,
        ejector.ejection.adiabatic_index)
    temperature_cyl_exit = calculate_temperature_cyl_section_exit(
        critical_temperature, ejector.ejection.pressure_cyl_section_exit,
        critical_pressure, ejector.ejection.adiabatic_index)

    _minor_header("ОСНОВНЫЕ ПАРАМЕТРЫ ЭЖЕКЦИИ")
    p("Степень сжатия:", f"{ejector.ejection.compression_ratio:.4f}")
    p("Коэффициент эжекции:", f"{ejector.ejection.entrainment_ratio:.4f}")
    p("m1 (участок струи до стенки):", f"{ejector.ejection.m1:.4f}")
    p("m (основной геометрический параметр):", f"{ejector.ejection.m:.4f}")
    p("n:", f"{ejector.ejection.n:.4f}")

    _minor_header("ГАЗОДИНАМИЧЕСКИЕ ПАРАМЕТРЫ")
    p("Газовая постоянная активной среды R_a:", f"{R_active:.2f}", "Дж/(кг·К)")
    p("Газовая постоянная пассивной среды R_n:",
      f"{R_passive:.2f}", "Дж/(кг·К)")
    p("Теплоемкость активной среды Cp_a:", f"{Cp_active:.2f}", "Дж/(кг·К)")
    p("Теплоемкость пассивной среды Cp_n:", f"{Cp_passive:.2f}", "Дж/(кг·К)")
    p("Показатель адиабаты k:", f"{ejector.ejection.adiabatic_index:.4f}")
    p("Критическое отношение давлений β:",
      f"{ejector.ejection.critical_pressure_ratio:.6f}")

    _minor_header("КРИТИЧЕСКИЕ ПАРАМЕТРЫ СОПЛА")
    p("Критическое давление P_кр:",
      f"{critical_pressure/PA_TO_MPA:.3f}", "МПа")
    p("Критическая температура T_кр:",
      f"{critical_temperature:.2f} К ({critical_temperature - KELVIN_TO_CELSIUS:.0f} °C)")

    _minor_header("ДАВЛЕНИЯ И НАПОРЫ")
    p("Динамический напор на выходе из сопла (I-I):",
      f"{ejector.ejection.dynamic_head_nozzle_exit/PA_TO_MPA:.3f}", "МПа")
    p("Напор эжектора без диффузора:",
      f"{ejector.ejection.ejector_head_no_diff/PA_TO_MPA:.3f}", "МПа")
    p("Давление в конце цилиндрического участка (III-III):",
      f"{ejector.ejection.pressure_cyl_section_exit/PA_TO_MPA:.3f}", "МПа")
    p("Температура в конце цилиндрического участка (III-III):",
      f"{temperature_cyl_exit:.2f} К ({temperature_cyl_exit - KELVIN_TO_CELSIUS:.0f} °C)")
    p("Давление за диффузором:",
      f"{ejector.diffuser.pressure_after/PA_TO_MPA:.3f}", "МПа")

    _minor_header("СКОРОСТИ ПОТОКОВ")
    p("Скорость активной среды в трубопроводе:",
      f"{calculate_gas_outflow_velocity(active.mass_flow, active.temperature,
                                        active.inlet_pressure, active.inlet_diameter,
                                        active.molecular_mass):.2f}", "м/с")
    p("Скорость пассивной среды в трубопроводе:",
      f"{calculate_gas_outflow_velocity(passive.mass_flow, passive.temperature,
                                        passive.inlet_pressure, passive.inlet_diameter,
                                        passive.molecular_mass):.2f}", "м/с")
    p("Скорость истечения газа из сопла:",
      f"{ejector.ejection.nozzle_exit_velocity:.2f}", "м/с")
    p("Скорость газа в конце смесительного участка:",
      f"{ejector.mixer.exit_velocity:.2f}", "м/с")
    p("Скорость движения потока за диффузором:",
      f"{ejector.pipeline.velocity:.2f}", "м/с")

    _minor_header("ГЕОМЕТРИЯ СОПЛА")
    p("Площадь выходного сечения сопла F1:",
      f"{ejector.nozzle.exit_area:.4f}", "м²")
    p("Диаметр выходного сечения сопла D1:",
      f"{ejector.nozzle.exit_diameter * M_TO_MM:.2f}", "мм")
    p("Площадь узкой части сопла Fкр:",
      f"{ejector.nozzle.throat_area:.4f}", "м²")
    p("Диаметр узкой части сопла Dкр:",
      f"{ejector.nozzle.throat_diameter * M_TO_MM:.2f}", "мм")

    _minor_header("ГЕОМЕТРИЯ СМЕСИТЕЛЬНОГО УЧАСТКА")
    p("Площадь сечения смесительного участка F3:",
      f"{ejector.mixer.section_area:.4f}", "м²")
    p("Диаметр смесительного участка D3:",
      f"{ejector.mixer.section_diameter * M_TO_MM:.2f}", "мм")
    p("Длина струи Lx'':",
      f"{ejector.mixer.jet_length * M_TO_MM:.2f}", "мм")
    p("Длина смесительного участка Lсм:",
      f"{ejector.mixer.section_length * M_TO_MM:.2f}", "мм")
    p("Расстояние от сопла до цилиндрического участка L1:",
      f"{ejector.mixer.nozzle_to_inlet_distance * M_TO_MM:.2f}", "мм")
    p("Длина цилиндрического участка L2:",
      f"{ejector.mixer.cylinder_length * M_TO_MM:.2f}", "мм")

    _minor_header("ГЕОМЕТРИЯ ДИФФУЗОРА")
    p("Площадь конечного сечения диффузора F4:",
      f"{ejector.diffuser.exit_area:.4f}", "м²")
    p("Длина диффузора L3:",
      f"{ejector.diffuser.length * M_TO_MM:.2f}", "мм")

    _minor_header("НАГНЕТАТЕЛЬНЫЙ ТРУБОПРОВОД")
    p("Скорость потока w4:", f"{ejector.pipeline.velocity:.2f}", "м/с")
    p("Число Рейнольдса Re:", f"{ejector.pipeline.reynolds_number:.0f}")

    _major_header("РАСЧЕТ ЗАВЕРШЕН")
