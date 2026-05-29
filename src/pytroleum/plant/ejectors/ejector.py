import numpy as np
from scipy.constants import g
from scipy.optimize import fsolve
from scipy.constants import R as UNIVERSAL_GAS_CONSTANT

from pytroleum.plant.ejectors.equations import (calculate_adiabatic_index,
                                                calculate_critical_pressure_ratio,
                                                calculate_gas_outflow_velocity,
                                                calculate_circle_area,
                                                calculate_circle_diameter,
                                                calculate_nozzle_throat_area,
                                                calculate_diffuser_length,
                                                calculate_section_temperature,
                                                calculate_reynolds_number,
                                                THERMAL_EQUIVALENT_OF_WORK)
from pytroleum.plant.ejectors.inputs import (OperationConditions,
                                             Requirements,
                                             ACTIVE, PASSIVE)
from pytroleum.plant.ejectors.utils import KCAL_TO_J

NOZZLE_VELOCITY_COEFF = 0.95


class BaseEjector:

    def __init__(self, conditions: OperationConditions,
                 req: Requirements):
        self.conditions = conditions
        self.req = req

        # Степень сжатия установки
        self.compression_ratio = (req.outlet_pressure /
                                  conditions.pressure[PASSIVE])

        # Коэффициент эжекции
        self.entrainment_ratio = (conditions.mass_flow_rate[PASSIVE] /
                                  conditions.mass_flow_rate[ACTIVE])

        # Скорость активной среды в трубопроводе w(a)
        self.velocity_active_inlet = calculate_gas_outflow_velocity(
            conditions.mass_flow_rate[ACTIVE], req.active_inlet_diameter,
            conditions.phase[ACTIVE])

        # Скорость пассивной среды в трубопроводе w(n)
        self.velocity_passive_inlet = calculate_gas_outflow_velocity(
            conditions.mass_flow_rate[PASSIVE], req.passive_inlet_diameter,
            conditions.phase[PASSIVE])

        # Показатель адиабаты
        self.adiabatic_index = calculate_adiabatic_index(
            conditions, self.entrainment_ratio)

        # Критическое отношение давлений
        self.critical_pressure_ratio = calculate_critical_pressure_ratio(
            conditions.phase[ACTIVE])


class GasEjector(BaseEjector):

    def __init__(self, conditions: OperationConditions,
                 req: Requirements):
        super().__init__(conditions, req)

        # Основное уравнение эжекции для участка струи от сопла до места
        # соприкосновения со стенкой
        self.m1 = 2 * (1 + self.entrainment_ratio) ** 2
        # NOTE атрибутам лучше давать говорящие имена, так как это часть публичного
        # NOTE интерфейса
        # NOTE (то, что видит пользователь, в том числе с помощью автодополнения)
        # NOTE говорящее имя = проще пользоваться

        # Основной геометрический параметр эжектора m
        self.m = self.m1 / (1 + (2 * self.entrainment_ratio ** 2) / self.m1)

        # Отношение геометрических параметров
        self.n = self.m / self.m1

        # Динамический напор эжектирующей струи на выходе из сопла (сечение I-I)
        self.dynamic_head_nozzle_exit = self.conditions.pressure[ACTIVE] / 1.1

        # Напор, создаваемый эжектором без диффузора
        self.ejector_head_no_diff = self.dynamic_head_nozzle_exit / self.m

    # NOTE Объект класса GasEjector должен содержать в себе атрибуты самого эжектора
    # NOTE а не рабочих условий и технических требований, эжектор может существовать
    # NOTE сам по себе, без них, тут в конструктор надо передавать размеры эжектора
    # NOTE
    # NOTE Проектный расчёт под конкретные условия уже требует ТЗ/рабочих условий, туда
    # NOTE эти объекты и нужно будет передавать

    def calculate_geometry_params(self, psi: float, opening_angle: float,
                                  s: float) -> None:
        """Геометрические размеры эжектора"""
        # Площадь конечного сечения диффузора
        self.diffuser_exit_area = calculate_circle_area(
            self.req.outlet_diameter)

        # Площадь сечения цилиндрического смесительного участка
        self.mixing_section_area = self.diffuser_exit_area / s

        # Диаметр сечения цилиндрического смесительного участка
        self.mixing_section_diameter = calculate_circle_diameter(
            self.mixing_section_area)

        # Площадь выходного сечения сопла
        self.nozzle_exit_area = self.mixing_section_area / self.m

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
        self.velocity_nozzle_exit = np.sqrt(
            2 * self.dynamic_head_nozzle_exit / self.conditions.phase[ACTIVE].rhomass())

        # Скорость газа в конце смесительного участка (w3)
        self.velocity_cyl_section_exit = (
            (self.conditions.mass_flow_rate[ACTIVE] +
             self.conditions.mass_flow_rate[PASSIVE]) /
            (mixture_density * self.mixing_section_area))

        # Скорость на выходе из эжектора (w4)
        self.velocity_ejector_outlet = self.velocity_cyl_section_exit / s

    def calculate_pressure_params(self, mixture_density: float,
                                  pressure_recovery_coefficient: float) -> None:
        """Давления по сечениям эжектора"""
        # Давление в конце цилиндрического участка (сечение III-III)
        self.pressure_cyl_section_exit = (self.ejector_head_no_diff +
                                          self.conditions.pressure[PASSIVE])

        # Давление в критическом сечении сопла
        self.pressure_critical = (self.critical_pressure_ratio *
                                  self.conditions.phase[ACTIVE].p())

        # Давление за диффузором
        self.pressure_ejector_outlet = (
            self.pressure_cyl_section_exit +
            pressure_recovery_coefficient *
            (mixture_density *
             self.velocity_cyl_section_exit ** 2) / 2)

    def calculate_temperature_params(self) -> None:
        """Температуры по сечениям эжектора"""
        # Температура в критическом сечении сопла (T1)
        self.temperature_nozzle_exit = (
            self.conditions.phase[ACTIVE].T() *
            self.critical_pressure_ratio **
            ((self.adiabatic_index - 1) / self.adiabatic_index))

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


class VaporEjector(BaseEjector):

    def __init__(self, conditions: OperationConditions,
                 req: Requirements):
        super().__init__(conditions, req)

    def calculate_nozzle_params(self,
                                active_enthalpy: float,
                                active_entropy: float,
                                entropy_lower_boundary: float,
                                entropy_upper_boundary: float,
                                enthalpy_lower_boundary: float,
                                phi: float = NOZZLE_VELOCITY_COEFF) -> None:
        """Расчёт сопла пароструйного эжектора.

        Параметры:
            active_enthalpy         — энтальпия активной среды, Дж/кг
            active_entropy          — энтропия активной среды, Дж/(кг·К)
            entropy_lower_boundary  — энтропия на нижней пограничной кривой
                                      при давлении газа в конце сопла
            entropy_upper_boundary  — энтропия на верхней пограничной кривой
                                      при давлении газа в конце сопла
            enthalpy_lower_boundary — энтальпия на нижней пограничной кривой
                                      при давлении газа в конце сопла
            phi                     — скоростной коэффициент сопла (0.95
                                      для сопел с большой степенью расширения)

        """

        # NOTE Возможно энтропии выйдет снимать через CoolProp, не нужно будет искать
        # NOTE табличные значения для данных условий
        # Табличные значения при давлении p1
        self.entropy_lower_boundary = entropy_lower_boundary
        self.entropy_upper_boundary = entropy_upper_boundary
        self.enthalpy_lower_boundary = enthalpy_lower_boundary

        # Температура активной среды в сопле
        self.temperature_nozzle_exit = (0.5045 *  # NOTE магическая константа
                                        self.conditions.temperature[ACTIVE])

        # Давление газа в конце сопла p1
        self.pressure_nozzle_exit = (0.68 *  # NOTE магическая константа
                                     self.conditions.pressure[ACTIVE])

        # Степень льдистости газа в выходном сечении сопла
        self.ice_quality_nozzle_exit = (
            (active_entropy - entropy_lower_boundary) /
            (entropy_upper_boundary - entropy_lower_boundary))

        # Скрытая степень льдистого газа в выходном сечении сопла
        # в зависимости от давления газа в конце сопла p1
        self.latent_heat_nozzle_exit = 0.986 * active_enthalpy
        # NOTE магическая константа

        # Энтальпия расширившейся в сопле среды
        self.enthalpy_nozzle_exit = (enthalpy_lower_boundary +
                                     self.latent_heat_nozzle_exit *
                                     self.ice_quality_nozzle_exit)

        # Действительная скорость истечения газа из сопла
        self.velocity_nozzle_exit = (91.5 * phi *  # NOTE магическая константа
                                     np.sqrt(active_enthalpy / KCAL_TO_J -
                                             self.enthalpy_nozzle_exit / KCAL_TO_J))

        # Потери тепла в сопле
        self.heat_loss_nozzle = ((1 - phi ** 2) *
                                 (active_enthalpy - self.enthalpy_nozzle_exit))

        # Энтальпия газа в конце сопла с учётом потерь
        self.enthalpy_nozzle_exit_actual = (self.enthalpy_nozzle_exit +
                                            self.heat_loss_nozzle)

        # Степень льдистости газа в конце действительного расширения
        self.ice_quality_nozzle_actual = ((self.enthalpy_nozzle_exit_actual -
                                           enthalpy_lower_boundary) /
                                          self.latent_heat_nozzle_exit)

        # Давление в критическом сечении сопла
        self.pressure_critical = (self.critical_pressure_ratio *
                                  self.conditions.pressure[ACTIVE])

        # Температура в критическом сечении сопла
        self.temperature_critical = (
            self.conditions.temperature[ACTIVE] *
            self.critical_pressure_ratio **
            ((self.adiabatic_index - 1) / self.adiabatic_index))

        # Газовая постоянная активной среды
        R_active = (UNIVERSAL_GAS_CONSTANT /
                    self.conditions.phase[ACTIVE].molar_mass())

        # Удельный объём насыщенного пара при давлении p1
        self.vapor_specific_vol = (R_active * self.temperature_critical /
                                   self.pressure_nozzle_exit)

        # Удельный объём льдистого газа на выходе из сопла
        self.specific_volume_nozzle_exit = (self.ice_quality_nozzle_actual *
                                            self.vapor_specific_vol)

        # Динамический напор эжектирующей струи
        self.dynamic_head_nozzle = (self.velocity_nozzle_exit ** 2 /
                                    (2 * self.specific_volume_nozzle_exit))

    def calculate_mixture_parameters(self,
                                     active_enthalpy: float,
                                     entrainment_ratios: list[float],
                                     pressure_recovery_coefficient: float,
                                     mach_number: float,
                                     enthalpies_cyl_exit: list[float],
                                     pressure_delta: float) -> None:
        """Расчёт основного геометрического параметра ступени."""

        # NOTE функция очень большая, обычно это означает, что мы пытаемся сделать
        # NOTE в ней слишком много всего и сразу, надо разбить на отдельные функции

        self.stage_entrainment_ratios = entrainment_ratios
        self.pressure_recovery_coefficient = pressure_recovery_coefficient
        self.mach_number = mach_number
        self.stage_enthalpies_cyl_exit = enthalpies_cyl_exit

        self.stage_adiabatic_indices = []
        self.stage_pressures_cyl_exit = []
        self.stage_partial_pressures_active = []
        self.stage_gas_constants_mixture = []
        self.stage_temperatures_cyl_exit = []
        self.stage_sound_velocities = []
        self.stage_mixture_velocities = []
        self.stage_specific_volumes = []
        self.stage_geometric_params = []
        self.stage_mixture_pressures = []

        R_active = (UNIVERSAL_GAS_CONSTANT /
                    self.conditions.phase[ACTIVE].molar_mass())
        R_passive = (UNIVERSAL_GAS_CONSTANT /
                     self.conditions.phase[PASSIVE].molar_mass())
        Cp_passive = self.conditions.phase[PASSIVE].cpmass()

        for entrainment_ratio, enthalpy_cyl_exit in zip(entrainment_ratios,
                                                        enthalpies_cyl_exit):
            # Показатель адиабаты смеси
            adiabatic_index_mixture = calculate_adiabatic_index(
                self.conditions, entrainment_ratio)

            # Газовая постоянная смеси
            gas_constant_mixture = ((R_active +
                                     entrainment_ratio * R_passive) /
                                    (1 + entrainment_ratio))

            # Давление смеси в конце цилиндрического участка (сечение III)
            pressure_cyl_exit = (
                self.req.outlet_pressure /
                (1 + pressure_recovery_coefficient *
                 adiabatic_index_mixture * mach_number ** 2 / 2))

            # Парциальное давление активной среды в конце цилиндрического участка
            partial_pressure_active = (
                pressure_cyl_exit /
                (1 + R_passive * entrainment_ratio / R_active))

            # Уравнение теплового баланса
            def heat_balance(temperature_cyl_exit_guess: np.ndarray,
                             _entrainment_ratio=entrainment_ratio,
                             _enthalpy_cyl_exit=enthalpy_cyl_exit,
                             _adiabatic_index_mixture=adiabatic_index_mixture
                             ) -> list[float]:
                T3 = temperature_cyl_exit_guess[0]
                return [
                    active_enthalpy + _entrainment_ratio * Cp_passive *
                    self.conditions.temperature[PASSIVE] - _entrainment_ratio *
                    Cp_passive * T3 -
                    _enthalpy_cyl_exit - THERMAL_EQUIVALENT_OF_WORK *
                    (_adiabatic_index_mixture * mach_number ** 2 / 2) *
                    (R_active + _entrainment_ratio * R_passive) * T3
                ]

            # Температура смеси в конце цилиндрического участка T3, К
            temperature_cyl_exit = float(
                fsolve(heat_balance,
                       [self.conditions.temperature[PASSIVE]])[0])

            # Местная скорость звука
            sound_velocity = np.sqrt(adiabatic_index_mixture *
                                     gas_constant_mixture *
                                     temperature_cyl_exit)

            # Скорость смеси в конце цилиндрического участка, м/с
            mixture_velocity = mach_number * sound_velocity

            # Удельный объём смеси в конце цилиндрического участка, м³/кг
            specific_volume = (gas_constant_mixture *
                               temperature_cyl_exit / pressure_cyl_exit)

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

    def calculate_geometry(self,
                           nozzle_expansion_ratio: float,
                           psi: float) -> None:
        """Расчёт геометрических параметров сопла пароструйного эжектора."""
        # Расчётная площадь выходного сечения сопла F1*, м²
        self.nozzle_exit_area_theoretical = (
            self.conditions.mass_flow_rate[ACTIVE] *
            self.specific_volume_nozzle_exit /
            self.velocity_nozzle_exit
        )

        # Площадь сечения цилиндрического участка:

        # Диаметр сечения цилиндрического участка:

        # Площадь критического сечения сопла Fкр, м²
        self.nozzle_throat_area = calculate_nozzle_throat_area(
            self.conditions, psi)

        # Диаметр критического сечения (без учёта погранслоя), м
        self.nozzle_throat_diameter = calculate_circle_diameter(
            self.nozzle_throat_area)

        # Диаметр критического сечения с учётом пограничного слоя, м
        self.nozzle_throat_diameter_corrected = (self.nozzle_throat_diameter *
                                                 1.3)

        # Действительная площадь выходного сечения сопла F1, м²
        self.nozzle_exit_area = (nozzle_expansion_ratio *
                                 self.nozzle_exit_area_theoretical)

        # Действительный диаметр выходного сечения сопла D1, м
        self.nozzle_exit_diameter = calculate_circle_diameter(
            self.nozzle_exit_area)

        # Диаметр сопла принимаем равным диаметру трубопровода активной среды
        self.nozzle_diameter = self.req.active_inlet_diameter

        # Длина сужающийся части сопла:

        # Длина расширяющейся части сопла:

        # Длину критического сечения Lкр принимают равной Dкр,
        # а затем при окончательной обработке доводят до
        # половины диаметра Dкр с получением плавного скругления
        # в местах перехода на конус

        # Диаметр камеры разряжения принимаем равным диаметру
        # трубопровода пассивной среды
        self.vacuum_chamber_diameter = self.req.passive_inlet_diameter

        # Входной диаметр камеры смешения принимаем равным
        # диаметру камеры разряжения
        self.inlet_mixing_diameter = self.vacuum_chamber_diameter

        # Переходная часть от Dк' к Dк
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
