import numpy as np
from pytroleum.plant.tps.utils import (_major_header, _minor_header, _minor_divider,
                                       _TO_MM, _TO_MICRON, PERCENT, SECONDS_PER_MINUTE)
from pytroleum.plant.tps.inputs import (SeparatorDesign,
                                        OperationConditions,
                                        flow_based_water_cut,
                                        flow_velocity,
                                        STANDARD_STATE,
                                        VAPOR, OIL, WATER, N_FLOWS)
from scipy.constants import g
from typing import Iterable

FIRST_SECTION = 0
SECOND_SECTION = 1
TOTAL = 2
# (первая секция, вторая секция, суммарный)
FILL_COEFFS = (0.858, 0.858, 0.858)


def compute_settling_velocity(drop_diameter: float,
                              continuous_phase_density: float,
                              continuous_phase_viscosity: float,
                              dispersed_phase_density: float) -> float:
    """Скорость осаждения/всплытия капли по закону Стокса, м/с.

    v_ст = g * d_к² * (ρ_нф - ρ_дф) / (18 * μ_нф)

    где d_к — диаметр капли, ρ_нф, ρ_дф — плотности непрерывной и дисперсной фаз,
    μ_нф — динамическая вязкость непрерывной фазы.
    Положительное значение — капля всплывает, отрицательное — оседает.
    """
    density_diff = continuous_phase_density - dispersed_phase_density
    return g * drop_diameter**2 * density_diff / (18 * continuous_phase_viscosity)


class Separator:
    def __init__(self, design: SeparatorDesign,
                 conditions: OperationConditions):
        self.design = design
        self.conditions = conditions
        self.velocity = np.zeros(N_FLOWS)

    def compute_flow_areas(self) -> tuple[float, float, float]:
        """Площади поперечного сечения для газа, нефти и воды, м².

        F_ж = F_сеч * к_зап,  F_в = F_ж * w,  F_н = F_ж - F_в,  F_г = F_сеч - F_ж

        где F_сеч — площадь поперечного сечения аппарата, к_зап — коэффициент
        заполнения, w — обводнённость.
        """
        liquid_area = self.design.section_area * FILL_COEFFS[FIRST_SECTION]
        water_area = liquid_area * flow_based_water_cut(self.conditions)
        oil_area = liquid_area - water_area
        gas_area = self.design.section_area - liquid_area
        return gas_area, oil_area, water_area

    def compute_velocities(self) -> None:
        """Скорости движения фаз в поперечном сечении сепаратора, м/с.

        u_г = Q_г_ру / F_г,  u_н = Q_н / F_н,  u_в = Q_в / F_в

        где Q_г_ру, Q_н, Q_в — объёмные расходы газа (при р.у.), нефти и воды,
        F_г, F_н, F_в — площади сечения для каждой фазы.
        """
        areas = self.compute_flow_areas()
        self.velocity = flow_velocity(self.conditions, np.array(areas))

    def residence_time(self) -> tuple[float, float, float]:
        """Время пребывания жидкости в секциях сепаратора, с.

        τ_пр = V_сек * к_зап / Q_ж

        где V_сек — объём секции, к_зап — коэффициент заполнения, Q_ж — расход жидкости.
        Суммарное: τ_общ = τ_пр_1 + τ_пр_2.
        """
        vol_flow_liquid = (self.conditions.vol_flow_rate[OIL] +
                           self.conditions.vol_flow_rate[WATER])
        rt_first = (self.design.volume[FIRST_SECTION] *
                    FILL_COEFFS[FIRST_SECTION] / vol_flow_liquid)
        rt_total = (self.design.volume_separator * FILL_COEFFS[TOTAL] /
                    vol_flow_liquid)
        rt_second = rt_total - rt_first
        return rt_first, rt_second, rt_total

    def transit_time(self, phase: int) -> float:
        """Время прохождения фазой расстояния от распределительной решётки до
        сливной перегородки, с.

        t_тр = L_c / u_ф

        где L_c — расстояние от решётки до перегородки, u_ф — скорость фазы.
        """
        return self.design.length_to_baffle / self.velocity[phase]

    def settling_height(self, drop_diameter: float,
                        continuous_phase_density: float,
                        continuous_phase_viscosity: float,
                        dispersed_phase_density: float,
                        phase: int) -> float:
        """Высота осаждения/всплытия капель за время прохождения L_c, м.

        h_ос = |v_ст| * t_тр

        где v_ст — скорость Стокса, t_тр — время прохождения фазой расстояния
        от распределительной решётки до сливной перегородки.
        """
        velocity = compute_settling_velocity(
            drop_diameter, continuous_phase_density,
            continuous_phase_viscosity, dispersed_phase_density,
        )
        return abs(velocity) * self.transit_time(phase)

    def capacity(self, fill_coeffs: Iterable[float] = FILL_COEFFS) -> tuple[float, float]:
        """Пропускная способность сепаратора по жидкости для каждой секции, м³/с.

        Q_доп = V_сек * к_зап / τ_пр

        где V_сек — объём секции, к_зап — коэффициент заполнения, τ_пр — время пребывания.
        """
        fill_coeffs = tuple(fill_coeffs)
        rt = self.residence_time()
        first_section_capacity = (self.design.volume[FIRST_SECTION] *
                                  fill_coeffs[FIRST_SECTION] / rt[FIRST_SECTION])
        second_section_capacity = (self.design.volume[SECOND_SECTION] *
                                   fill_coeffs[SECOND_SECTION] / rt[SECOND_SECTION])
        return first_section_capacity, second_section_capacity
