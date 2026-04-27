import numpy as np
from pytroleum.plant.tps.utils import (_major_header, _minor_header, _minor_divider,
                                       SECONDS_PER_DAY, _TO_MM)
from pytroleum.plant.tps.inputs import (OperationConditions,
                                        CoalescerPacking,
                                        GeometryCyclone,
                                        STANDARD_STATE,
                                        VAPOR, OIL, WATER)
from pytroleum.plant.tps.separator import compute_settling_velocity, Separator


class Coalescer:
    def __init__(self, coalescer_packing: CoalescerPacking,
                 separator: Separator) -> None:
        self.coalescer_packing = coalescer_packing
        self.separator = separator

    def droplet_settling_time(self, plate_spacing: float, drop_diameter: float,
                              continuous_phase_density: float,
                              continuous_phase_viscosity: float,
                              dispersed_phase_density: float) -> float:
        """Время осаждения/всплытия капли в зазоре между пластинами, с.

        t_к = h / (|v_ст| * cos(α))

        где h — расстояние между пластинами, v_ст — скорость Стокса,
        α — угол наклона пластин.
        """
        velocity = compute_settling_velocity(
            drop_diameter, continuous_phase_density,
            continuous_phase_viscosity, dispersed_phase_density,
        )
        return (plate_spacing / (abs(velocity) *
                                 np.cos(np.radians(self.coalescer_packing.angle))))

    def required_length_for(self, phase_velocity: float, settling_time: float) -> float:
        """Длина канала коалесцера, м.

        L_кан = u_ф * t_к

        где u_ф — скорость фазы в канале, t_к — время осаждения/всплытия капли.
        """
        return phase_velocity * settling_time


class Cyclone:

    def __init__(self, conditions: OperationConditions,
                 geometry_cyclone: GeometryCyclone):
        self.conditions = conditions
        self.geometry_cyclone = geometry_cyclone

    def vapor_velocity(self, number_of_cyclones: int) -> float:
        """Скорость газа в спиральном канале, м/с.

        uг_сп = Qг_ру / (n * F_кан)

        где Qг_ру — расход газа при р.у., n — число циклонов,
        F_кан — площадь сечения спирального канала одного циклона.
        """
        return self.conditions.vol_flow_rate[VAPOR] / (
            number_of_cyclones * self.geometry_cyclone.area_spiral_channel)


# ============================================================
# Пример использования
# ============================================================

if __name__ == "__main__":
    from CoolProp import constants as CoolConst
    from pytroleum.plant.tps.inputs import SeparatorDesign
    from pytroleum.plant.tps.utils import SECONDS_PER_MINUTE

    pressure = 4e6
    temperature = 353
    flow_gas_norm = 300_000 / SECONDS_PER_DAY
    flow_oil = 200 / SECONDS_PER_DAY
    flow_water = 300 / SECONDS_PER_DAY

    conditions = OperationConditions()
    conditions.phase[VAPOR].update(*STANDARD_STATE)
    gas_density_norm = conditions.phase[VAPOR].rhomass()
    conditions.phase[OIL].change(933, 3.073e-3)  # type: ignore
    conditions.update_state((CoolConst.PT_INPUTS, pressure, temperature),
                            upd_containers=True)
    conditions.vol_flow_rate = np.array([
        flow_gas_norm * gas_density_norm / conditions.phase[VAPOR].rhomass(),
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
    coalescer_packing = CoalescerPacking(coalescer_top_gap=15e-3,
                                         coalescer_bottom_gap=25e-3)

    number_of_cyclones = 4
    geometry_cyclone = GeometryCyclone(width_inlet_cyclone=47.5e-3,
                                       height_inlet_cyclone=75e-3)

    separator = Separator(design=design, conditions=conditions)
    separator.compute_velocities()

    coalescer = Coalescer(
        coalescer_packing=coalescer_packing, separator=separator)
    cyclone = Cyclone(conditions=conditions, geometry_cyclone=geometry_cyclone)

    oil_density = conditions.phase[OIL].rhomass()
    water_density = conditions.phase[WATER].rhomass()
    oil_viscosity = conditions.phase[OIL].viscosity()
    water_viscosity = conditions.phase[WATER].viscosity()

    _major_header("РАСЧЁТ КОАЛЕСЦЕРА")

    _minor_header("ВЕРХНИЙ КОАЛЕСЦЕР")
    print(f"Угол наклона пластин: {coalescer_packing.angle:.0f}°")
    print(
        f"Зазор между пластинами: {coalescer_packing.coalescer_top_gap * _TO_MM:.0f} мм")
    t_top = coalescer.droplet_settling_time(
        coalescer_packing.coalescer_top_gap,
        diameter_water_droplet,
        oil_density, oil_viscosity, water_density,
    )
    print(
        f"Время осаждения капель воды в зазоре: {t_top / SECONDS_PER_MINUTE:.2f} мин")
    print(f"Длина канала: "
          f"{coalescer.required_length_for(separator.velocity[OIL], t_top):.4f} м")

    _minor_header("НИЖНИЙ КОАЛЕСЦЕР")
    print(f"Угол наклона пластин: {coalescer_packing.angle:.0f}°")
    print(f"Зазор между пластинами: "
          f"{coalescer_packing.coalescer_bottom_gap * _TO_MM:.0f} мм")
    t_bottom = coalescer.droplet_settling_time(
        coalescer_packing.coalescer_bottom_gap,
        diameter_oil_droplet,
        water_density, water_viscosity, oil_density,
    )
    print(
        f"Время всплытия капель нефти в зазоре: {t_bottom / SECONDS_PER_MINUTE:.2f} мин")
    print(f"Длина канала: "
          f"{coalescer.required_length_for(separator.velocity[WATER], t_bottom):.4f} мм")

    _major_header(
        "РАСЧЁТ СКОРОСТИ ГАЗА В СЕПАРАЦИОННОМ ЭЛЕМЕНТЕ (СПИРАЛЬНЫЙ КАНАЛ)")

    _minor_header("ГЕОМЕТРИЯ ЦИКЛОНА")
    print(
        f"Ширина входа в циклон: {geometry_cyclone.width_inlet_cyclone * _TO_MM:.1f} мм")
    print(
        f"Высота входа в циклон: {geometry_cyclone.height_inlet_cyclone * _TO_MM:.1f} мм")
    print(f"Количество циклонов: {number_of_cyclones}")

    _minor_divider()
    print(f"Расход газа при р.у.: "
          f"{conditions.vol_flow_rate[VAPOR] * SECONDS_PER_DAY:.1f} м³/сут")
    print(f"Площадь сечения спирального канала: "
          f"{geometry_cyclone.area_spiral_channel:.4f} м²")
    print(f"Скорость газа в спиральном канале:  "
          f"{cyclone.vapor_velocity(number_of_cyclones):.3f} м/с")
