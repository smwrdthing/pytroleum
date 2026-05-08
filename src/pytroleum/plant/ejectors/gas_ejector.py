from scipy.constants import g
import numpy as np
from dataclasses import dataclass
from pytroleum.plant.ejectors.inputs import (ActiveMediumData,
                                             PassiveMediumData,
                                             CommonParams)
from pytroleum.plant.ejectors.equations import (
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
    cyl_section_exit = ejector_head_no_diff + passive.inlet_pressure

    # Давление в критическом сечении сопла
    critical = calculate_critical_pressure(
        ejection.critical_pressure_ratio, active.inlet_pressure)

    # Давление за диффузором
    ejector_outlet = (cyl_section_exit +
                      pressure_recovery_coefficient *
                      (calculate_specific_weight(mixture_density) *
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
