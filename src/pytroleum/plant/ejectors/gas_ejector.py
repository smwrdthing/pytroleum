import numpy as np
from scipy.constants import g

from pytroleum.plant.ejectors.equations import (calculate_circle_area,
                                                calculate_section_area,
                                                calculate_circle_diameter,
                                                calculate_nozzle_throat_area,
                                                calculate_diffuser_length,
                                                calculate_section_velocity,
                                                calculate_critical_pressure,
                                                calculate_critical_temperature,
                                                calculate_section_temperature,
                                                calculate_reynolds_number)

from pytroleum.plant.ejectors.inputs import (OperationConditions,
                                             Requirements,
                                             ACTIVE, PASSIVE)
from pytroleum.plant.ejectors.base_ejector import BaseEjector

# NOTE как и в других местах - я бы предпочёл уйти от силовых единиц расхода и удельного
# NOTE веса и пользоваться вместо этого обычными единицами измерения расхода и плотностью
# NOTE во всех расчётных формулах


class GasEjector(BaseEjector):

    def __init__(self, conditions: OperationConditions,
                 req: Requirements):
        super().__init__(conditions, req)

        # Основное уравнение эжекции для участка струи от сопла до места
        # соприкосновения со стенкой
        self.m1 = 2 * (1 + self.entrainment_ratio) ** 2

        # Основной геометрический параметр эжектора m
        self.m = self.m1 / (1 + (2 * self.entrainment_ratio ** 2) / self.m1)

        # Отношение геометрических параметров
        self.n = self.m / self.m1

    def calculate_geometry_params(self, psi: float, opening_angle: float,
                                  s: float) -> None:
        """Геометрические размеры эжектора"""
        # Площадь конечного сечения диффузора
        self.diffuser_exit_area = calculate_circle_area(
            self.req.outlet_diameter)

        # Площадь сечения цилиндрического смесительного участка
        self.mixing_section_area = calculate_section_area(
            self.diffuser_exit_area, s)

        # Диаметр сечения цилиндрического смесительного участка
        self.mixing_section_diameter = calculate_circle_diameter(
            self.mixing_section_area)

        # Площадь выходного сечения сопла
        self.nozzle_exit_area = calculate_section_area(
            self.mixing_section_area, self.m)

        # Диаметр выходного сечения сопла
        self.nozzle_exit_diameter = calculate_circle_diameter(
            self.nozzle_exit_area)

        # Площадь сечения узкой части сопла
        self.nozzle_throat_area = calculate_nozzle_throat_area(
            self.conditions, psi)

        # Диаметр сечения узкой части сопла
        self.nozzle_throat_diameter = calculate_circle_diameter(
            self.nozzle_throat_area)

        # Длина струи
        self.jet_length = (self.nozzle_exit_diameter *
                           (4 * (1 + self.entrainment_ratio) - 1.8))

        # Длина смесительного участка
        self.mixing_section_length = 2.5 * self.mixing_section_diameter

        # Расстояние от сопла до начала цилиндрического участка L1
        self.nozzle_to_inlet_distance = (self.jet_length -
                                         0.5 * self.mixing_section_diameter)

        # Длина цилиндрического участка L2
        self.cylinder_length = (self.jet_length + self.mixing_section_length -
                                self.nozzle_to_inlet_distance)

        # Длина диффузора
        self.diffuser_length = calculate_diffuser_length(
            self.req.outlet_diameter,
            self.mixing_section_diameter,
            opening_angle)

    def calculate_velocity_params(self, mixture_density: float,
                                  s: float) -> None:
        """Скорости по сечениям эжектора"""

        # Скорость истечения газа из сопла (w1)
        nozzle_exit_pressure = self.conditions.pressure[ACTIVE] / 1.1
        # NOTE напор будет "head", чтобы не путаться со статическим давлением
        self.velocity_nozzle_exit = np.sqrt(
            2 * nozzle_exit_pressure / self.conditions.phase[ACTIVE].rhomass())

        # Скорость газа в конце смесительного участка (w3)
        self.velocity_cyl_section_exit = (
            (self.conditions.mass_flow_rate[ACTIVE] +
             self.conditions.mass_flow_rate[PASSIVE]) /
            (mixture_density * self.mixing_section_area))

        # Скорость на выходе из эжектора (w4)
        self.velocity_ejector_outlet = calculate_section_velocity(
            self.velocity_cyl_section_exit, s)

    def calculate_pressure_params(self, mixture_density: float,
                                  pressure_recovery_coefficient: float) -> None:
        """Давления по сечениям эжектора"""
        # Динамический напор эжектирующей струи на выходе из сопла (сечение I-I)
        self.dynamic_head_nozzle_exit = self.conditions.pressure[ACTIVE] / 1.1
        # NOTE дважды считаем одно и то же (см. функцию выше)

        # Напор, создаваемый эжектором без диффузора
        self.ejector_head_no_diff = self.dynamic_head_nozzle_exit / self.m

        # Давление в конце цилиндрического участка (сечение III-III)
        self.pressure_cyl_section_exit = (self.ejector_head_no_diff +
                                          self.conditions.pressure[PASSIVE])

        # Давление в критическом сечении сопла
        self.pressure_critical = calculate_critical_pressure(
            self.critical_pressure_ratio, self.conditions.pressure[ACTIVE])

        # Давление за диффузором
        self.pressure_ejector_outlet = (
            self.pressure_cyl_section_exit +
            pressure_recovery_coefficient *
            (mixture_density *
             self.velocity_cyl_section_exit ** 2) / 2)

    def calculate_temperature_params(self) -> None:
        """Температуры по сечениям эжектора"""
        # Температура в критическом сечении сопла (T1)
        self.temperature_nozzle_exit = calculate_critical_temperature(
            self.conditions.temperature[ACTIVE], self.critical_pressure_ratio,
            self.adiabatic_index)

        # Температура в конце цилиндрического участка (T3)
        self.temperature_cyl_section_exit = calculate_section_temperature(
            self.temperature_nozzle_exit, self.pressure_cyl_section_exit,
            self.pressure_critical, self.adiabatic_index)

        # Температура на выходе из диффузора (T4)
        self.temperature_diffuser_exit = calculate_section_temperature(
            self.temperature_cyl_section_exit, self.pressure_ejector_outlet,
            self.pressure_cyl_section_exit, self.adiabatic_index)

    def calculate(self, s: float, mixture_density: float,
                  pressure_recovery_coefficient: float, psi: float,
                  opening_angle: float,
                  mixture_dynamic_viscosity: float) -> None:
        """Полный расчёт газового эжектора"""
        self.calculate_geometry_params(psi, opening_angle, s)
        self.calculate_velocity_params(mixture_density, s)
        self.calculate_pressure_params(mixture_density,
                                       pressure_recovery_coefficient)
        self.calculate_temperature_params()

        # Число Рейнольдса в нагнетательном трубопроводе за диффузором
        self.reynolds_number = calculate_reynolds_number(
            mixture_density, self.velocity_ejector_outlet,
            self.req.outlet_diameter, mixture_dynamic_viscosity)
