import numpy as np
from scipy.constants import g

from pytroleum.plant.ejectors.equations import *
from pytroleum.plant.ejectors.inputs import (ActiveMediumData,
                                             PassiveMediumData,
                                             CommonParams)
from pytroleum.plant.ejectors.gas_ejector import BaseEjector

KCAL_TO_J = 4186.8
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

    def calculate_stage_geometry(self,
                                 entrainment_ratios: list[float],
                                 pressure_recovery_coefficient: float,
                                 mach_number: float) -> None:
        """Расчёт основного геометрического параметра ступени."""

        self.stage_entrainment_ratios = entrainment_ratios
        self.pressure_recovery_coefficient = pressure_recovery_coefficient
        self.mach_number = mach_number

        self.stage_adiabatic_indices = []
        self.stage_pressures_cyl_exit = []

        for q in entrainment_ratios:
            # Показатель адиабаты смеси для данного q (k3)
            k3 = calculate_adiabatic_index(self.active, self.passive, q)

            # Давление смеси в конце цилиндрического участка (сечение III)
            p3 = (self.common_params.outlet_pressure /
                  (1 + pressure_recovery_coefficient * k3 * mach_number ** 2 / 2))

            self.stage_adiabatic_indices.append(k3)
            self.stage_pressures_cyl_exit.append(p3)
