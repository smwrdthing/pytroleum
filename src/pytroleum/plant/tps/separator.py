from pytroleum.plant.tps.utils import (_major_header, _minor_header, _minor_divider,
                                       _TO_MM, _TO_MICRON, PERCENT, SECONDS_PER_MINUTE)
from pytroleum.plant.tps.inputs import (SeparatorDesign,
                                        OperationConditions,
                                        PhysicalProperties,
                                        FlowRates)
from pytroleum.plant.tps.wire_mesh_demister import WireMeshDemister
from pytroleum.plant.tps.nozzle import Nozzle

import numpy as np
from scipy.constants import g
from typing import Iterable

FIRST_SECTION = 0
SECOND_SECTION = 1
TOTAL = 2
FILL_COEFFS = (0.858, 0.858, 0.858)  # (первая секция, вторая секция, суммарный)

VAPOR = 0
OIL = 1
WATER = 2


def compute_settling_velocity(drop_diameter: float,
                              continuous_phase_density: float,
                              continuous_phase_viscosity: float,
                              dispersed_phase_density: float) -> float:
    """Скорость осаждения/всплытия капли по закону Стокса, м/с. Положительное значение
    — капля всплывает (дисперсная фаза легче непрерывной),отрицательное — оседает."""
    density_diff = continuous_phase_density - dispersed_phase_density
    return g * drop_diameter**2 * density_diff / (18 * continuous_phase_viscosity)


class Separator:
    def __init__(self, design: SeparatorDesign,
                 conditions: OperationConditions,
                 properties: PhysicalProperties,
                 flows: FlowRates,
                 diameter_water_droplet: float,
                 diameter_oil_droplet: float):
        self.design = design
        self.conditions = conditions
        self.properties = properties
        self.flows = flows
        self.diameter_water_droplet = diameter_water_droplet
        self.diameter_oil_droplet = diameter_oil_droplet

    def compute_flow_areas(self) -> tuple[float, float, float]:
        """Площади поперечного сечения для газа, нефти и воды, м²."""
        section_area = np.pi * self.design.inner_diameter ** 2 / 4
        liquid_area = section_area * FILL_COEFFS[FIRST_SECTION]
        water_area = liquid_area * self.flows.properties.water_cut
        oil_area = liquid_area - water_area
        gas_area = section_area - liquid_area
        return gas_area, oil_area, water_area

    def compute_velocities(self) -> None:
        """Вычисляет скорости газа, нефти и воды, м/с."""
        areas = self.compute_flow_areas()
        self.flows.velocity[VAPOR] = self.flows.flow_rate[VAPOR] / areas[VAPOR]
        self.flows.velocity[OIL] = self.flows.flow_rate[OIL] / areas[OIL]
        self.flows.velocity[WATER] = self.flows.flow_rate[WATER] / areas[WATER]

    def residence_time(self) -> tuple[float, float, float]:
        """Время пребывания жидкости в секциях сепаратора, с."""
        rt_first = (self.design.volume[FIRST_SECTION] *
                    FILL_COEFFS[FIRST_SECTION] / self.conditions.flow_liquid)
        rt_total = (self.design.volume_separator * FILL_COEFFS[TOTAL] /
                    self.conditions.flow_liquid)
        rt_second = rt_total - rt_first
        return rt_first, rt_second, rt_total

    def transit_time(self, phase: int) -> float:
        """Время прохождения фазой расстояния от распределительной
        решётки до сливной перегородки, с."""
        return self.design.L_c / self.flows.velocity[phase]

    def settling_height(self, drop_diameter: float,
                        continuous_phase_density: float,
                        continuous_phase_viscosity: float,
                        dispersed_phase_density: float,
                        phase: int) -> float:
        """Высота осаждения/всплытия капель, м."""
        velocity = compute_settling_velocity(
            drop_diameter, continuous_phase_density,
            continuous_phase_viscosity, dispersed_phase_density,
        )
        return abs(velocity) * self.transit_time(phase)

    def capacity(self, fill_coeffs: Iterable[float] = FILL_COEFFS) -> tuple[float, float]:
        """Пропускная способность сепаратора по жидкости для каждой секции, м³/с."""
        fill_coeffs = tuple(fill_coeffs)
        rt = self.residence_time()
        first_section_capacity = (self.design.volume[FIRST_SECTION] *
                                  fill_coeffs[FIRST_SECTION] / rt[FIRST_SECTION])
        second_section_capacity = (self.design.volume[SECOND_SECTION] *
                                   fill_coeffs[SECOND_SECTION] / rt[SECOND_SECTION])
        return first_section_capacity, second_section_capacity

    def resistance(self, losses_unaccounted: float,
                   demister: WireMeshDemister,
                   liquidgasnozzle: Nozzle,
                   gasnozzle: Nozzle) -> tuple[float, float, float, float]:
        """Сопротивления сепаратора, Па."""

        assert liquidgasnozzle.resistance_coeff is not None
        assert gasnozzle.resistance_coeff is not None
        gas_density = self.properties.gas_density_work(self.conditions)

        pressure_drop_mesh_demister = (
            demister.mesh_resistance_coefficient * gas_density *
            demister.actual_velocity() ** 2 / 2)

        pressure_drop_inlet_nozzle = (
            liquidgasnozzle.resistance_coeff * gas_density *
            liquidgasnozzle.flow_velocity(self.flows.flow_gas_work) ** 2 / 2)

        pressure_drop_outlet_nozzle = (
            gasnozzle.resistance_coeff * gas_density *
            gasnozzle.flow_velocity(self.flows.flow_gas_work) ** 2 / 2)

        pressure_drop_total = losses_unaccounted * (pressure_drop_mesh_demister +
                                                    pressure_drop_inlet_nozzle +
                                                    pressure_drop_outlet_nozzle)

        return (pressure_drop_mesh_demister, pressure_drop_inlet_nozzle,
                pressure_drop_outlet_nozzle, pressure_drop_total)


if __name__ == "__main__":
    from pytroleum.plant.tps.utils import SECONDS_PER_DAY
    from pytroleum.plant.tps.nozzle import design_gas_nozzle, design_liquid_gas_nozzle

    design = SeparatorDesign(
        inner_diameter=2.0,
        length_cylindrical_part=9.5,
        length_semiaxis=0.618,
        length_first_section=8.2,
        length_second_section=1.3,
        L_c=4.7
    )
    conditions = OperationConditions(
        pressure_work=4e6,
        temperature_work=353,
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

    separator = Separator(design=design, conditions=conditions,
                          properties=properties, flows=flows,
                          diameter_water_droplet=diameter_water_droplet,
                          diameter_oil_droplet=diameter_oil_droplet)

    demister = WireMeshDemister(properties, flows,
                                area_reduction_coefficient=1.05,
                                mesh_resistance_coefficient=70)
    gasnozzle = design_gas_nozzle(flows=flows,
                                  speed=10.0,
                                  resistance_coeff=0.5)
    liquidgasnozzle = design_liquid_gas_nozzle(flows=flows,
                                               gas_speed=10.0,
                                               liquid_speed=1.0,
                                               resistance_coeff=1.0)
    losses_unaccounted = 1.2

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
    rt = separator.residence_time()
    capacities = separator.capacity()

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

    _minor_header("РАСЧЕТ СКОРОСТЕЙ ДВИЖЕНИЯ ЖИДКОЙ ФАЗЫ И ГАЗОВОЙ ФАЗЫ В СЕЧЕНИИ СЕПАРАТОРА")

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
          f"{design.L_c:.1f} м")
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
          f"{design.L_c:.1f} м")
    print(f"Диаметр капли нефти: "
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

    _major_header("РАСЧЁТ СОПРОТИВЛЕНИЯ СЕПАРАТОРА")

    _minor_divider()
    print(f"Плотность газа при рабочих условиях: "
          f"{properties.gas_density_work(conditions):.3f} кг/м³")

    _minor_divider()
    print(f"Диаметр штуцера входа ГЖС: "
          f"{liquidgasnozzle.nominal_diameter * _TO_MM:.0f} мм")
    print(f"Диаметр штуцера выхода газа: "
          f"{gasnozzle.nominal_diameter * _TO_MM:.0f} мм")

    _minor_divider()
    print(f"Скорость во входном штуцере ГЖС: "
          f"{liquidgasnozzle.flow_velocity(flows.flow_gas_work):.3f} м/с")
    print(f"Скорость в выходном штуцере газа: "
          f"{gasnozzle.flow_velocity(flows.flow_gas_work):.3f} м/с")

    _minor_divider()
    print(f"Коэффициент сопротивления входного патрубка: "
          f"{liquidgasnozzle.resistance_coeff}")
    print(f"Коэффициент сопротивления выходного патрубка: "
          f"{gasnozzle.resistance_coeff}")
    print(f"Коэффициент сопротивления сетчатого отбойника: "
          f"{demister.mesh_resistance_coefficient}")
    print(f"Коэффициент неучтенных потерь: {losses_unaccounted}")

    (pressure_drop_mesh_demister, pressure_drop_inlet_nozzle,
     pressure_drop_outlet_nozzle, pressure_drop_total) = separator.resistance(
        losses_unaccounted, demister, liquidgasnozzle, gasnozzle)

    _minor_divider()
    print(
        f"Падение давления на отбойнике: {pressure_drop_mesh_demister:.3f} Па")
    print(
        f"Потери давления на штуцере входа: {pressure_drop_inlet_nozzle:.3f} Па")
    print(
        f"Потери давления на штуцере выхода: {pressure_drop_outlet_nozzle:.3f} Па")

    _minor_divider()
    print(f"Сопротивление сепаратора: {pressure_drop_total:.3f} Па")
    _minor_divider()
