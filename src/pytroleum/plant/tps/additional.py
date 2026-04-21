from pytroleum.plant.tps.utils import (_major_header, _minor_header, _minor_divider,
                                       SECONDS_PER_DAY,
                                       _TO_MM)
from pytroleum.plant.tps.inputs import (PhysicalProperties,
                                        OperationConditions,
                                        FlowRates, GeometryCyclone,
                                        CoalescerPacking)
from pytroleum.plant.tps.separator import compute_settling_velocity, Separator, OIL, WATER

import numpy as np


class Coalescer:
    def __init__(self, coalescer_packing: CoalescerPacking,
                 separator: Separator) -> None:
        self.coalescer_packing = coalescer_packing
        self.separator = separator

    def droplet_settling_time(self, plate_spacing: float, drop_diameter: float,
                              continuous_phase_density: float,
                              continuous_phase_viscosity: float,
                              dispersed_phase_density: float) -> float:
        """Время осаждения/всплытия капли в зазоре, с."""
        velocity = compute_settling_velocity(
            drop_diameter, continuous_phase_density,
            continuous_phase_viscosity, dispersed_phase_density,
        )
        return (plate_spacing / (abs(velocity) *
                                 np.cos(np.radians(self.coalescer_packing.angle))))

    def channel_length(self, phase_velocity: float, settling_time: float) -> float:
        """Длина канала коалесцера, м."""
        return phase_velocity * settling_time


class Cyclone:
    """ Расчет скорости газа в сепарационном элементе (спиральный канал)"""

    def __init__(self, flows: FlowRates, geometry_cyclone: GeometryCyclone):
        self.flows = flows
        self.geometry_cyclone = geometry_cyclone

    def velocity_gas_in_spiral_channel(self):
        return self.flows.flow_gas_work / (self.geometry_cyclone.number_of_cyclones *
                                           self.geometry_cyclone.area_spiral_channel)

# ============================================================
# Пример использования
# ============================================================


if __name__ == "__main__":
    from pytroleum.plant.tps.inputs import SeparatorDesign
    from pytroleum.plant.tps.utils import SECONDS_PER_MINUTE

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

    design = SeparatorDesign(
        inner_diameter=2.0,
        length_cylindrical_part=9.5,
        length_semiaxis=0.618,
        length_first_section=8.2,
        length_second_section=1.3,
        L_c=4.7
    )
    diameter_water_droplet = 100e-6
    diameter_oil_droplet = 50e-6
    coalescer_packing = CoalescerPacking(
        coalescer_top_gap=15e-3,
        coalescer_bottom_gap=25e-3,
    )

    geometry_cyclone = GeometryCyclone(width_inlet_cyclone=47.5e-3,
                                       height_inlet_cyclone=75e-3, number_of_cyclones=4)

    separator = Separator(design=design, conditions=conditions,
                          properties=properties, flows=flows,
                          diameter_water_droplet=diameter_water_droplet,
                          diameter_oil_droplet=diameter_oil_droplet)
    separator.compute_velocities()

    coalescer = Coalescer(
        coalescer_packing=coalescer_packing, separator=separator)
    cyclone = Cyclone(flows=flows, geometry_cyclone=geometry_cyclone)

    _major_header("РАСЧЁТ КОАЛЕСЦЕРА")

    _minor_header("ВЕРХНИЙ КОАЛЕСЦЕР")
    print(f"Угол наклона пластин: {coalescer_packing.angle:.0f}°")
    print(
        f"Зазор между пластинами: {coalescer_packing.coalescer_top_gap * _TO_MM:.0f} мм")
    t_top = coalescer.droplet_settling_time(
        coalescer_packing.coalescer_top_gap,
        diameter_water_droplet,
        properties.oil_density,
        properties.viscosity_oil,
        properties.water_density,
    )
    print(
        f"Время осаждения капель воды в зазоре: {t_top / SECONDS_PER_MINUTE:.2f} мин")
    print(
        f"Длина канала: {coalescer.channel_length(flows.velocity[OIL], t_top):.4f} м")

    _minor_header("НИЖНИЙ КОАЛЕСЦЕР")
    print(f"Угол наклона пластин: {coalescer_packing.angle:.0f}°")
    print(f"Зазор между пластинами: "
          f"{coalescer_packing.coalescer_bottom_gap * _TO_MM:.0f} мм")

    t_bottom = coalescer.droplet_settling_time(
        coalescer_packing.coalescer_bottom_gap,
        diameter_oil_droplet,
        properties.water_density,
        properties.viscosity_water,
        properties.oil_density,
    )
    print(
        f"Время всплытия капель нефти в зазоре: {t_bottom / SECONDS_PER_MINUTE:.2f} мин")
    print(f"Длина канала: "
          f"{coalescer.channel_length(flows.velocity[WATER], t_bottom):.4f} мм")

    _major_header(
        "РАСЧЁТ СКОРОСТИ ГАЗА В СЕПАРАЦИОННОМ ЭЛЕМЕНТЕ (СПИРАЛЬНЫЙ КАНАЛ)")

    _minor_header("ГЕОМЕТРИЯ ЦИКЛОНА")
    print(
        f"Ширина входа в циклон: {geometry_cyclone.width_inlet_cyclone * _TO_MM:.1f} мм")
    print(
        f"Высота входа в циклон: {geometry_cyclone.height_inlet_cyclone * _TO_MM:.1f} мм")
    print(f"Количество циклонов: {geometry_cyclone.number_of_cyclones}")

    _minor_divider()
    print(f"Расход газа при рабочих условиях: "
          f"{flows.flow_gas_work * SECONDS_PER_DAY:.1f} м³/сут")
    print(f"Площадь сечения спирального канала: "
          f"{geometry_cyclone.area_spiral_channel:.4f} м²")
    print(f"Скорость газа в спиральном канале: "
          f"{cyclone.velocity_gas_in_spiral_channel():.3f} м/с")
