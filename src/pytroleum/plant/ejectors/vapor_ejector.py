import numpy as np
from scipy.constants import g
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from pytroleum.plant.ejectors.equations import *
from pytroleum.plant.ejectors.base_ejector import BaseEjector
from pytroleum.plant.ejectors.utils import PA_TO_MPA, KCAL_TO_J

NOZZLE_VELOCITY_COEFF = 0.95


class VaporEjector(BaseEjector):

    def calculate_nozzle_params(self,
                                entropy_lower_boundary: float,
                                entropy_upper_boundary: float,
                                enthalpy_lower_boundary: float,
                                phi: float = NOZZLE_VELOCITY_COEFF) -> None:
        """Расчёт сопла пароструйного эжектора.

        Параметры:
            entropy_lower_boundary  — энтропия на нижней пограничной кривой
                                      при давлении газа в конце сопла
            entropy_upper_boundary  — энтропия на верхней пограничной кривой
                                      при давлении газа в конце сопла
            enthalpy_lower_boundary — энтальпия на нижней пограничной кривой
                                      при давлении газа в конце сопла
            phi                     — скоростной коэффициент сопла (0.95
                                      для сопел с большой степенью расширения)
        """
        # Табличные значения при давлении p1
        self.entropy_lower_boundary = entropy_lower_boundary
        self.entropy_upper_boundary = entropy_upper_boundary
        self.enthalpy_lower_boundary = enthalpy_lower_boundary

        # Температура активной среды в сопле
        self.temperature_nozzle_exit = 0.5045 * self.active.temperature

        # Давление газа в конце сопла p1
        self.pressure_nozzle_exit = 0.68 * self.active.inlet_pressure

        # Степень льдистости газа в выходном сечении сопла
        self.ice_quality_nozzle_exit = (
            (self.active.entropy - entropy_lower_boundary) /
            (entropy_upper_boundary - entropy_lower_boundary))

        # Скрытая степень льдистости газа  в выходном сечении сопла
        # в зависимости от давления газа в конце сопла, p1
        self.latent_heat_nozzle_exit = 0.986 * self.active.enthalpy

        # Скоростной коэффициент сопла
        self.phi = phi

        # Энтальпия расширившейся в сопле среды
        self.enthalpy_nozzle_exit = (enthalpy_lower_boundary +
                                     self.latent_heat_nozzle_exit *
                                     self.ice_quality_nozzle_exit)
        # Действительная скорость истечения газа из сопла
        self.velocity_nozzle_exit = (91.5 * phi *
                                     np.sqrt(self.active.enthalpy/KCAL_TO_J -
                                             self.enthalpy_nozzle_exit/KCAL_TO_J))

        # Потери тепла в сопле
        self.heat_loss_nozzle = ((1 - phi ** 2) * (self.active.enthalpy -
                                 self.enthalpy_nozzle_exit))

        # Энтальпия газа в конце сопла с учётом потерь
        self.enthalpy_nozzle_exit_actual = (self.enthalpy_nozzle_exit +
                                            self.heat_loss_nozzle)

        # Степень льдистости газа в конце действительного расширения
        self.ice_quality_nozzle_actual = ((self.enthalpy_nozzle_exit_actual -
                                           self.enthalpy_lower_boundary) /
                                          self.latent_heat_nozzle_exit)

        # Давление в критическом сечении сопла
        self.pressure_critical = calculate_critical_pressure(
            self.critical_pressure_ratio, self.active.inlet_pressure)

        # Температура в критическом сечении сопла (T1)
        self.temperature_critical = calculate_critical_temperature(
            self.active.temperature, self.critical_pressure_ratio,
            self.adiabatic_index)

        # Удельный объём насыщенного пара при давлении p1
        R_active = calculate_gas_constant(self.active.molecular_mass)
        self.vapor_specific_vol = (R_active * self.temperature_critical /
                                   self.pressure_nozzle_exit)

        # Удельный объём льдистого газа на выходе из сопла
        self.specific_volume_nozzle_exit = (self.ice_quality_nozzle_actual *
                                            self.vapor_specific_vol)

        # Динамический напор эжектирующей струи
        self.dynamic_head_nozzle = (self.velocity_nozzle_exit ** 2 /
                                    (2 * self.specific_volume_nozzle_exit))

    def calculate_mixture_parameters(self,
                                     entrainment_ratios: list[float],
                                     pressure_recovery_coefficient: float,
                                     mach_number: float,
                                     enthalpies_cyl_exit: list[float],
                                     pressure_delta: float) -> None:
        """Расчёт основного геометрического параметра ступени."""
        self.stage_entrainment_ratios = entrainment_ratios
        self.pressure_recovery_coefficient = pressure_recovery_coefficient
        self.mach_number = mach_number
        self.stage_enthalpies_cyl_exit = enthalpies_cyl_exit

        self.stage_adiabatic_indices = []
        self.stage_pressures_cyl_exit = []
        self.stage_partial_pressures_active = []
        self.stage_gas_constants_mixture = []
        self.stage_temperatures_cyl_exit = []   # T3, К
        self.stage_sound_velocities = []         # a(3), м/с
        self.stage_mixture_velocities = []       # w(3), м/с
        self.stage_specific_volumes = []         # v(3), м³/кг
        self.stage_geometric_params = []         # m, основной геометрический параметр
        self.stage_mixture_pressures = []  # p(3)

        gas_constant_active = calculate_gas_constant(
            self.active.molecular_mass)
        gas_constant_passive = calculate_gas_constant(
            self.passive.molecular_mass)

        # Удельные теплоёмкости, Дж/(кг·К)
        specific_heat_capacity_passive = calculate_specific_heat_capacity(
            self.passive.heat_capacity, self.passive.molecular_mass)

        for entrainment_ratio, enthalpy_cyl_exit in zip(entrainment_ratios,
                                                        enthalpies_cyl_exit):
            # Показатель адиабаты смеси
            adiabatic_index_mixture = calculate_adiabatic_index(
                self.active, self.passive, entrainment_ratio)

            # Газовая постоянная смеси
            gas_constant_mixture = ((gas_constant_active +
                                    entrainment_ratio * gas_constant_passive) /
                                    (1 + entrainment_ratio))

            # Давление смеси в конце цилиндрического участка (сечение III)
            pressure_cyl_exit = (
                self.common_params.outlet_pressure /
                (1 + pressure_recovery_coefficient *
                 adiabatic_index_mixture * mach_number ** 2 / 2))

            # Парциальное давление активной среды в конце цилиндрического участка
            partial_pressure_active = (
                pressure_cyl_exit /
                (1 + gas_constant_passive * entrainment_ratio / gas_constant_active))

            # Уравнение теплового баланса:
            def heat_balance(temperature_cyl_exit_guess: np.ndarray) -> list[float]:
                T3 = temperature_cyl_exit_guess[0]
                return [
                    self.active.enthalpy
                    + entrainment_ratio * specific_heat_capacity_passive * self.passive.temperature
                    - entrainment_ratio * specific_heat_capacity_passive * T3
                    - enthalpy_cyl_exit
                    - THERMAL_EQUIVALENT_OF_WORK
                    * (adiabatic_index_mixture * mach_number ** 2 / 2)
                    * (gas_constant_active + entrainment_ratio * gas_constant_passive)
                    * T3
                ]

            t3_initial_guess = self.passive.temperature

            # Температура смеси в конце цилиндрического участка T3, К
            temperature_cyl_exit = float(
                fsolve(heat_balance, [t3_initial_guess])[0])

            # Расчет местной скорости звука
            sound_velocity = np.sqrt(adiabatic_index_mixture *
                                     gas_constant_mixture * temperature_cyl_exit)

            # Скорость смеси в конце цилиндрического участка, м/с
            mixture_velocity = mach_number * sound_velocity

            # Удельный объём смеси в конце цилиндрического участка, м³/кг
            specific_volume = gas_constant_mixture * \
                temperature_cyl_exit / pressure_cyl_exit

            # Основной геометрический параметр ступени m
            geometric_param = (
                (1 + entrainment_ratio) *
                specific_volume *
                self.velocity_nozzle_exit /
                (self.specific_volume_nozzle_exit * mixture_velocity)
            )

            # Давление смеси
            pressure_mixture = (
                self.pressure_nozzle_exit +
                2 * self.dynamic_head_nozzle *
                (pressure_delta - (1 + entrainment_ratio) *
                 mixture_velocity / self.velocity_nozzle_exit) /
                geometric_param
            )

            self.stage_adiabatic_indices.append(adiabatic_index_mixture)
            self.stage_pressures_cyl_exit.append(pressure_cyl_exit)
            self.stage_partial_pressures_active.append(partial_pressure_active)
            self.stage_gas_constants_mixture.append(gas_constant_mixture)
            self.stage_temperatures_cyl_exit.append(temperature_cyl_exit)
            self.stage_sound_velocities.append(sound_velocity)
            self.stage_mixture_velocities.append(mixture_velocity)
            self.stage_specific_volumes.append(specific_volume)
            self.stage_geometric_params.append(geometric_param)
            self.stage_mixture_pressures.append(pressure_mixture)

    def plot_mixture_pressure_vs_entrainment(self) -> None:
        """График 1 — зависимость давления смеси p(3) от коэффициента эжекции q."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.stage_entrainment_ratios,
                [p / PA_TO_MPA for p in self.stage_mixture_pressures], marker='o')
        ax.set_xlabel('Коэффициент эжекции q')
        ax.set_ylabel('Давление смеси p(3), МПа')
        ax.grid(True, linestyle='--', alpha=0.5)
        for entrainment_ratio, pressure in zip(self.stage_entrainment_ratios,
                                               self.stage_mixture_pressures):
            ax.annotate(f'{pressure / PA_TO_MPA:.2f}', xy=(entrainment_ratio, pressure / PA_TO_MPA),
                        textcoords='offset points', xytext=(0, 10), ha='center')
        ax.set_ylim(7, 9)
        plt.tight_layout()

    def calculate_geometry(self,
                           nozzle_expansion_ratio: float,
                           psi: float) -> None:
        """Расчёт геометрических параметров сопла пароструйного эжектора."""
        # Расчётная площадь выходного сечения сопла F1*, м²
        self.nozzle_exit_area_theoretical = (
            self.active.mass_flow * self.specific_volume_nozzle_exit /
            self.velocity_nozzle_exit
        )

        # Площадь сечения цилиндрического участка:

        # Диаметр сечения цилиндрического участка:

        # Площадь критического сечения сопла Fкр, м²
        self.nozzle_throat_area = calculate_nozzle_throat_area(
            self.active, psi)

        # Диаметр критического сечения (без учёта погранслоя), м
        self.nozzle_throat_diameter = calculate_circle_diameter(
            self.nozzle_throat_area)

        # Диаметр критического сечения с учётом пограничного слоя, м
        self.nozzle_throat_diameter_corrected = self.nozzle_throat_diameter * 1.3

        # Действительная площадь выходного сечения сопла F1, м²
        self.nozzle_exit_area = nozzle_expansion_ratio * self.nozzle_exit_area_theoretical

        # Действительный диаметр выходного сечения сопла D1, м
        self.nozzle_exit_diameter = calculate_circle_diameter(
            self.nozzle_exit_area)

        # Диаметр сопла принимаем равным диаметру труопровода эжектирующего газа
        self.nozzle_diameter = self.active.inlet_diameter

        # Длина сужающийся части сопла:

        # Длина расширяющейся части сопла:

        # Длину критического сечения Lкр принимают равной Dкр,
        # а затем при окончательной обработке доводят до
        # половины диаметра Dкр с получением плавного скругления
        # в местах перехода на конус:

        # Диаметр камеры разряжения (D2) принимаем
        # равным диаметру трубопровода эжектируемой среды (D(п))
        self.vacuum_chamber_diameter = self.passive.inlet_diameter

        # Входной диаметр камеры смешения (Dк')
        # принимаем равным диаметру камеры разряжения (D2)
        self.inlet_mixing_diameter = self.vacuum_chamber_diameter

        # Переходная часть от диаметра (Dк') к диаметру (Dк)
        self.mixing_diameter = 0.9 * self.inlet_mixing_diameter

        # Радиус кривизны камеры смешения
        self.curvature_radius = 0.3 * self.mixing_diameter

        # Длина камеры смешения
        self.mixing_length = 0.25 * self.mixing_diameter

        # Площадь кольцевого сечния F между диаметрами Dк и D1':

        # Диаметр концевого участка сопла:

        # Скорость газа в кольцевом сечении:

        # Длина сужающейся части диффузора:

        # Отношение площадей F4/F3 можно вычислить по формуле:

        # Длина расщиряющейся части диффузора:

        # Расстояние от сопла до начала цилиндрического участка:

        # Длина цилиндрического участка:

        # Положение сопла относительно диффузора:
