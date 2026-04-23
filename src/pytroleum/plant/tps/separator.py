from pytroleum.plant.tps.utils import (_major_header, _minor_header, _minor_divider,
                                       _TO_MM, _TO_MICRON, PERCENT, SECONDS_PER_MINUTE)
from pytroleum.plant.tps.inputs import (SeparatorDesign,
                                        OperationConditions,
                                        PhysicalProperties,
                                        FlowRates)
from scipy.constants import g
from typing import Iterable

FIRST_SECTION = 0
SECOND_SECTION = 1
TOTAL = 2
# (первая секция, вторая секция, суммарный)
FILL_COEFFS = (0.858, 0.858, 0.858)

VAPOR = 0
OIL = 1
WATER = 2


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
                 flows: FlowRates):
        self.design = design
        self.flows = flows  # NOTE информацию о расходах можно держать в conditions

    def compute_flow_areas(self) -> tuple[float, float, float]:
        """Площади поперечного сечения для газа, нефти и воды, м².

        F_ж = F_сеч * к_зап,  F_в = F_ж * w,  F_н = F_ж - F_в,  F_г = F_сеч - F_ж

        где F_сеч — площадь поперечного сечения аппарата, к_зап — коэффициент
        заполнения, w — обводнённость.
        """
        liquid_area = self.design.section_area * FILL_COEFFS[FIRST_SECTION]
        water_area = liquid_area * self.flows.properties.water_cut
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
        self.flows.velocity[VAPOR] = self.flows.flow_rate[VAPOR] / areas[VAPOR]
        self.flows.velocity[OIL] = self.flows.flow_rate[OIL] / areas[OIL]
        self.flows.velocity[WATER] = self.flows.flow_rate[WATER] / areas[WATER]

    def residence_time(self,
                       conditions: OperationConditions) -> tuple[float, float, float]:
        """Время пребывания жидкости в секциях сепаратора, с.

        τ_пр = V_сек * к_зап / Q_ж

        где V_сек — объём секции, к_зап — коэффициент заполнения, Q_ж — расход жидкости.
        Суммарное: τ_общ = τ_пр_1 + τ_пр_2.
        """
        rt_first = (self.design.volume[FIRST_SECTION] *
                    FILL_COEFFS[FIRST_SECTION] / conditions.flow_liquid)
        rt_total = (self.design.volume_separator * FILL_COEFFS[TOTAL] /
                    conditions.flow_liquid)
        rt_second = rt_total - rt_first
        return rt_first, rt_second, rt_total

    def transit_time(self, phase: int) -> float:
        """Время прохождения фазой расстояния от распределительной
        решётки до сливной перегородки, с.

        t_тр = L_c / u_ф

        где L_c — расстояние от решётки до перегородки, u_ф — скорость фазы.
        """
        return self.design.length_to_baffle / self.flows.velocity[phase]

    def settling_height(self, drop_diameter: float,
                        continuous_phase_density: float,
                        continuous_phase_viscosity: float,
                        dispersed_phase_density: float,
                        phase: int) -> float:
        """Высота осаждения/всплытия капель за время прохождения L_c, м.

        h_ос = |v_ст| * t_тр

        где v_ст — скорость Стокса, t_тр — Время прохождения фазой расстояния
        от распределительной решётки до сливной перегородки
        """
        velocity = compute_settling_velocity(
            drop_diameter, continuous_phase_density,
            continuous_phase_viscosity, dispersed_phase_density,
        )
        return abs(velocity) * self.transit_time(phase)

    def capacity(self, conditions: OperationConditions,
                 fill_coeffs: Iterable[float] = FILL_COEFFS) -> tuple[float, float]:
        """Пропускная способность сепаратора по жидкости для каждой секции, м³/с.

        Q_доп = V_сек * к_зап / τ_пр

        где V_сек — объём секции, к_зап — коэффициент заполнения, τ_пр — время пребывания.
        """
        fill_coeffs = tuple(fill_coeffs)
        rt = self.residence_time(conditions)
        first_section_capacity = (self.design.volume[FIRST_SECTION] *
                                  fill_coeffs[FIRST_SECTION] / rt[FIRST_SECTION])
        second_section_capacity = (self.design.volume[SECOND_SECTION] *
                                   fill_coeffs[SECOND_SECTION] / rt[SECOND_SECTION])
        return first_section_capacity, second_section_capacity


if __name__ == "__main__":
    from pytroleum.plant.tps.utils import SECONDS_PER_DAY

    design = SeparatorDesign(
        inner_diameter=2.0,
        length_cylindrical_part=9.5,
        length_semiaxis=0.618,
        length_first_section=8.2,
        length_second_section=1.3,
        length_to_baffle=4.7
    )
    conditions = OperationConditions(
        pressure=4e6,
        temperature=353,
        flow_gas_norm=300000 / SECONDS_PER_DAY,
        flow_liquid=500 / SECONDS_PER_DAY,
    )

    properties = PhysicalProperties(
        gas_density_norm=0.94,
        oil_density=933,
        water_density=966,
        water_cut=0.6,
        gas_factor=267.9,
        oil_surface_tension=0.02848,
        viscosity_oil=3.073e-3,
        viscosity_water=0.544e-3
    )

    flows = FlowRates(conditions=conditions, properties=properties)

    diameter_water_droplet = 100e-6
    diameter_oil_droplet = 50e-6

    separator = Separator(design=design, flows=flows)

    _major_header("РАСЧЁТ ПРОПУСКНОЙ СПОСОБНОСТИ СЕПАРАТОРА ПО ЖИДКОСТИ")

    print(f"Внутренний диаметр сепаратора: "
          f"{design.inner_diameter * _TO_MM:.0f} мм")
    print(f"Длина цилиндрической части сепаратора: "
          f"{design.length_cylindrical_part:.1f} м")
    print(f"Длина полуоси эллиптического днища: "
          f"{design.length_semiaxis:.3f} м")
    print(f"Объёмный расход жидкости: "
          f"{conditions.flow_liquid * SECONDS_PER_DAY:.1f} м³/сут")
    print(f"Коэффициент заполнения: "
          f"{FILL_COEFFS[TOTAL] * PERCENT:.1f} %")
    print(f"Номинальный объём сепаратора: "
          f"{design.volume_separator:.3f} м³")

    separator.compute_velocities()
    rt = separator.residence_time(conditions)
    capacities = separator.capacity(conditions)

    print(f"Время пребывания жидкости: "
          f"{rt[TOTAL] / SECONDS_PER_MINUTE:.2f} мин")
    print(f"Пропускная способность: "
          f"{conditions.flow_liquid * SECONDS_PER_DAY:.2f} м³/сут")

    _minor_header("ПЕРВАЯ СЕКЦИЯ (НЕФТЬ + ВОДА)")
    print(f"Длина первой секции: "
          f"{design.length_first_section:.1f} м")
    print(f"Коэффициент заполнения: "
          f"{FILL_COEFFS[FIRST_SECTION] * PERCENT:.1f} %")

    print(f"Объём первой секции: "
          f"{design.volume[FIRST_SECTION]:.3f} м³")
    print(f"Время пребывания (Н+В): "
          f"{rt[FIRST_SECTION] / SECONDS_PER_MINUTE:.3f} мин")
    print(f"Пропускная способность: "
          f"{capacities[FIRST_SECTION] * SECONDS_PER_DAY:.2f} м³/сут")

    _minor_header("СБОРНИК НЕФТИ (ПОСЛЕ ПЕРЕГОРОДКИ)")
    print(f"Длина секции после перегородки: "
          f"{design.length_second_section:.1f} м")
    print(f"Коэффициент заполнения: "
          f"{FILL_COEFFS[SECOND_SECTION] * PERCENT:.1f} %")
    print(f"Объём секции после перегородки: "
          f"{design.volume[SECOND_SECTION]:.3f} м³")
    print(f"Время пребывания: "
          f"{rt[SECOND_SECTION] / SECONDS_PER_MINUTE:.3f} мин")
    print(f"Пропускная способность: "
          f"{capacities[SECOND_SECTION] * SECONDS_PER_DAY:.3f} м³/сут")

    _minor_header(
        "РАСЧЕТ СКОРОСТЕЙ ДВИЖЕНИЯ ЖИДКОЙ ФАЗЫ И ГАЗОВОЙ ФАЗЫ В СЕЧЕНИИ СЕПАРАТОРА")

    areas = separator.compute_flow_areas()

    print(
        f"Площадь сечения для прохода жидкости: {areas[OIL] + areas[WATER]:.3f} м²")
    print(f"Площадь сечения для прохода газа: {areas[VAPOR]:.3f} м²")
    print(f"Площадь сечения для прохода воды: {areas[WATER]:.3f} м²")
    print(f"Площадь сечения для прохода нефти: {areas[OIL]:.3f} м²")

    _minor_divider()
    print(f"Скорость движения газа: {flows.velocity[VAPOR]:.4f} м/с")
    print(f"Скорость движения нефти: {flows.velocity[OIL] * _TO_MM:.4f} мм/с")
    print(f"Скорость движения воды: {flows.velocity[WATER] * _TO_MM:.4f} мм/с")

    _minor_header("ОСАЖДЕНИЕ КАПЕЛЬ ВОДЫ В СЛОЕ НЕФТИ")
    print(f"Расстояние от распределительной решетки до сливной перегородки: "
          f"{design.length_to_baffle:.1f} м")
    print(f"Диаметр капли воды: "
          f"{diameter_water_droplet * _TO_MICRON:.0f} мкм")
    print(f"Скорость осаждения капель воды: "
          f"{abs(compute_settling_velocity(diameter_water_droplet, properties.oil_density,
                                           properties.viscosity_oil,
                                           properties.water_density)) * _TO_MM:.4f} мм/с")
    print(f"Время прохождения нефтью расстояния: "
          f"{separator.transit_time(OIL):.2f} с")
    print(f"Высота осаждения капель воды: "
          f"{separator.settling_height(diameter_water_droplet,
                                       properties.oil_density,
                                       properties.viscosity_oil,
                                       properties.water_density, OIL) * _TO_MM:.2f} мм")

    _minor_header("ВСПЛЫТИЕ КАПЕЛЬ НЕФТИ В СЛОЕ ВОДЫ")
    print(f"Расстояние от распределительной решетки до сливной перегородки: "
          f"{design.length_to_baffle:.1f} м")
    print(f"Диаметр капли нефти:"
          f"{diameter_oil_droplet * _TO_MICRON:.0f} мкм")
    print(f"Скорость подъёма капель нефти: "
          f"{compute_settling_velocity(diameter_oil_droplet, properties.water_density,
                                       properties.viscosity_water,
                                       properties.oil_density) * _TO_MM:.4f} мм/с")
    print(f"Время прохождения водой расстояния: "
          f"{separator.transit_time(WATER):.2f} с")
    print(f"Высота подъёма капель нефти: "
          f"{separator.settling_height(diameter_oil_droplet,
                                       properties.water_density,
                                       properties.viscosity_water,
                                       properties.oil_density, WATER) * _TO_MM:.2f} мм")
