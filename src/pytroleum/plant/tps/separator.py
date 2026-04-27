import numpy as np
from pytroleum.plant.tps.utils import (_major_header, _minor_header, _minor_divider,
                                       _TO_MM, _TO_MICRON, PERCENT, SECONDS_PER_MINUTE)
from pytroleum.plant.tps.inputs import (SeparatorDesign,
                                        OperationConditions,
                                        flow_based_water_cut,
                                        flow_velocity,
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
        q_liquid = (self.conditions.vol_flow_rate[OIL] +
                    self.conditions.vol_flow_rate[WATER])
        rt_first = (self.design.volume[FIRST_SECTION] *
                    FILL_COEFFS[FIRST_SECTION] / q_liquid)
        rt_total = (self.design.volume_separator * FILL_COEFFS[TOTAL] /
                    q_liquid)
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


if __name__ == "__main__":
    from CoolProp import constants as CoolConst
    from pytroleum.plant.tps.utils import SECONDS_PER_DAY

    pressure_work = 4e6
    temperature_work = 353
    flow_oil = 200 / SECONDS_PER_DAY
    flow_water = 300 / SECONDS_PER_DAY
    flow_gas_norm = 300_000 / SECONDS_PER_DAY

    conditions = OperationConditions()
    gas_density_norm = conditions.phase[VAPOR].rhomass()
    conditions.phase[OIL].change(933, 3.073e-3)  # type: ignore
    conditions.update_state((CoolConst.PT_INPUTS, pressure_work, temperature_work),
                            upd_containers=True)
    mass_flow_gas = flow_gas_norm * gas_density_norm
    conditions.vol_flow_rate = np.array([
        mass_flow_gas / conditions.phase[VAPOR].rhomass(),
        flow_oil, flow_water,
    ])

    design = SeparatorDesign(
        inner_diameter=2.0,
        length_cylindrical_part=9.5,
        length_semiaxis=0.618,
        length_first_section=8.2,
        length_second_section=1.3,
        length_to_baffle=4.7
    )
    diameter_water_droplet = 100e-6
    diameter_oil_droplet = 50e-6

    separator = Separator(design=design, conditions=conditions)

    _major_header("РАСЧЁТ ПРОПУСКНОЙ СПОСОБНОСТИ СЕПАРАТОРА ПО ЖИДКОСТИ")

    print(
        f"Внутренний диаметр сепаратора: {design.inner_diameter * _TO_MM:.0f} мм")
    print(
        f"Длина цилиндрической части: {design.length_cylindrical_part:.1f} м")
    print(
        f"Длина полуоси эллиптического днища: {design.length_semiaxis:.3f} м")
    q_liquid = conditions.vol_flow_rate[OIL] + conditions.vol_flow_rate[WATER]
    print(f"Объёмный расход жидкости: "
          f"{q_liquid * SECONDS_PER_DAY:.1f} м³/сут")

    print(
        f"Коэффициент заполнения: {FILL_COEFFS[TOTAL] * PERCENT:.1f} %")
    print(
        f"Номинальный объём сепаратора: {design.volume_separator:.3f} м³")

    separator.compute_velocities()
    rt = separator.residence_time()
    capacities = separator.capacity()

    print(
        f"Время пребывания жидкости: {rt[TOTAL] / SECONDS_PER_MINUTE:.2f} мин")

    _minor_header("ПЕРВАЯ СЕКЦИЯ (НЕФТЬ + ВОДА)")
    print(
        f"Длина первой секции: {design.length_first_section:.1f} м")
    print(
        f"Коэффициент заполнения: {FILL_COEFFS[FIRST_SECTION] * PERCENT:.1f} %")
    print(
        f"Объём первой секции: {design.volume[FIRST_SECTION]:.3f} м³")
    print(
        f"Время пребывания (Н+В): {rt[FIRST_SECTION] / SECONDS_PER_MINUTE:.3f} мин")
    print(f"Пропускная способность: "
          f"{capacities[FIRST_SECTION] * SECONDS_PER_DAY:.2f} м³/сут")

    _minor_header("СБОРНИК НЕФТИ (ПОСЛЕ ПЕРЕГОРОДКИ)")
    print(
        f"Длина секции после перегородки: {design.length_second_section:.1f} м")
    print(
        f"Коэффициент заполнения: {FILL_COEFFS[SECOND_SECTION] * PERCENT:.1f} %")
    print(
        f"Объём секции после перегородки: {design.volume[SECOND_SECTION]:.3f} м³")
    print(
        f"Время пребывания: {rt[SECOND_SECTION] / SECONDS_PER_MINUTE:.3f} мин")
    print(f"Пропускная способность: "
          f"{capacities[SECOND_SECTION] * SECONDS_PER_DAY:.3f} м³/сут")

    _minor_header("СКОРОСТИ ДВИЖЕНИЯ ФАЗ В СЕЧЕНИИ СЕПАРАТОРА")
    areas = separator.compute_flow_areas()
    print(
        f"Площадь сечения для прохода жидкости: {areas[OIL] + areas[WATER]:.3f} м²")
    print(f"Площадь сечения для прохода газа: {areas[VAPOR]:.3f} м²")
    print(f"Площадь сечения для прохода нефти: {areas[OIL]:.3f} м²")
    print(f"Площадь сечения для прохода воды: {areas[WATER]:.3f} м²")
    _minor_divider()
    print(f"Скорость движения газа: {separator.velocity[VAPOR]:.4f} м/с")
    print(
        f"Скорость движения нефти: {separator.velocity[OIL] * _TO_MM:.4f} мм/с")
    print(
        f"Скорость движения воды: {separator.velocity[WATER] * _TO_MM:.4f} мм/с")

    _minor_header("ОСАЖДЕНИЕ КАПЕЛЬ ВОДЫ В СЛОЕ НЕФТИ")
    rho_oil = conditions.phase[OIL].rhomass()
    rho_water = conditions.phase[WATER].rhomass()
    mu_oil = conditions.phase[OIL].viscosity()
    mu_water = conditions.phase[WATER].viscosity()
    print(
        f"Расстояние решётка — перегородка: {design.length_to_baffle:.1f} м")
    print(
        f"Диаметр капли воды: {diameter_water_droplet * _TO_MICRON:.0f} мкм")

    velocity_water = abs(compute_settling_velocity(
        diameter_water_droplet, rho_oil, mu_oil, rho_water))
    print(f"Скорость осаждения капель воды: "
          f"{velocity_water * _TO_MM:.4f} мм/с")
    print(
        f"Время прохождения нефтью: {separator.transit_time(OIL):.2f} с")

    height_water = separator.settling_height(
        diameter_water_droplet, rho_oil, mu_oil, rho_water, OIL)
    print(f"Высота осаждения капель воды: "
          f"{height_water * _TO_MM:.2f} мм")

    _minor_header("ВСПЛЫТИЕ КАПЕЛЬ НЕФТИ В СЛОЕ ВОДЫ")
    print(
        f"Расстояние решётка — перегородка: {design.length_to_baffle:.1f} м")
    print(
        f"Диаметр капли нефти: {diameter_oil_droplet * _TO_MICRON:.0f} мкм")
    velocity_oil = compute_settling_velocity(
        diameter_oil_droplet, rho_water, mu_water, rho_oil)
    print(f"Скорость подъёма капель нефти: "
          f"{velocity_oil * _TO_MM:.4f} мм/с")
    print(
        f"Время прохождения водой: {separator.transit_time(WATER):.2f} с")
    height_oil = separator.settling_height(
        diameter_oil_droplet, rho_water, mu_water, rho_oil, WATER)
    print(f"Высота подъёма капель нефти: "
          f"{height_oil * _TO_MM:.2f} мм")
