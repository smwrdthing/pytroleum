import numpy as np
from pytroleum.plant.tps.inputs import (OperationConditions,
                                        STANDARD_STATE, VAPOR, OIL, WATER)
from pytroleum.plant.tps.utils import _TO_MM, _TO_M

MAX_GAS_VELOCITY = 20  # м/с
MIN_GAS_VELOCITY = 10  # м/с

# СТК 542 РР1 Сепаратор режим 2 (высоковязкая нефть)
MAX_LIQUID_VELOCITY = 3  # м/с
MIN_LIQUID_VELOCITY = 1  # м/с

# Таблица с номинальными диаметрами:
# https://dpva.ru/guide/guideequipment/connections/diameters/pipeoutsidediametercorrespondance/

NOMINAL_DIAMETERS = np.array([
    10, 15, 20, 25, 32, 40, 50, 65, 80, 90, 100, 125, 150,
    200, 225, 250, 300, 400, 500, 600, 800, 1000, 1200]) * _TO_M


def select_nominal_diameter(diameter: float, nominal_diameters) -> float:
    """Выбор ближайшего большего номинального диаметра, м."""
    for d_nom in sorted(nominal_diameters):
        if d_nom >= diameter:
            return d_nom
    raise ValueError(
        f"Расчётный диаметр {diameter * _TO_MM:.1f} мм "
        f"больше {max(nominal_diameters) * _TO_MM:.0f} мм")


def _validate_velocity(name: str, velocity: float,
                       min_velocity: float, max_velocity: float) -> None:
    if not min_velocity <= velocity <= max_velocity:
        print(f"Предупреждение: скорость в {name} "
              f"{velocity} м/с вне диапазона [{min_velocity}, {max_velocity}] м/с")


class Nozzle:
    """Класс описывает патрубок"""

    def __init__(self, diameter: float) -> None:
        self.diameter = diameter
        self.area = np.pi * diameter ** 2 / 4

        self.nominal_diameter = select_nominal_diameter(
            diameter, NOMINAL_DIAMETERS)
        self.nominal_area = np.pi * self.nominal_diameter ** 2 / 4

    def flow_velocity(self, volumetric_flow_rate: float) -> float:
        """Скорость потока в штуцере по номинальному диаметру, м/с.

        u_шт = Q / F_ном,  F_ном = π * D_ном² / 4

        где Q — объёмный расход фазы, D_ном — принятый номинальный диаметр.
        """
        return volumetric_flow_rate / self.nominal_area


def design_nozzle(volumetric_flow_rate, target_velocity) -> Nozzle:
    """Расчётный диаметр штуцера по заданному расходу и скорости.

    Однофазный:  D_расч = √(4 * Q / (π * u_зад))

    Двухфазный (ГЖС):  D_расч = 1.13 * √(Σ Q_i / (1.3 * u_зад_i))

    где Q_i — объёмные расходы фаз, u_зад_i — заданные скорости фаз.
    """
    volumetric_flow_rate = np.atleast_1d(volumetric_flow_rate)
    target_velocity = np.atleast_1d(target_velocity)

    if len(volumetric_flow_rate) == 1:
        diameter = np.sqrt(4 * volumetric_flow_rate[0] /
                           (np.pi * target_velocity[0]))
    else:
        diameter = 1.13 * np.sqrt(
            np.sum(volumetric_flow_rate / (1.3 * target_velocity))
        )

    return Nozzle(diameter)


def design_two_phase_nozzle(conditions: OperationConditions, gas_speed: float,
                            liquid_speed: float) -> Nozzle:
    """Штуцер для двухфазного потока (ГЖС)."""
    _validate_velocity("Штуцер ГЖС (газ)", gas_speed,
                       MIN_GAS_VELOCITY, MAX_GAS_VELOCITY)
    _validate_velocity("Штуцер ГЖС (жидкость)", liquid_speed,
                       MIN_LIQUID_VELOCITY, MAX_LIQUID_VELOCITY)
    return design_nozzle(
        [conditions.vol_flow_rate[VAPOR],
         conditions.vol_flow_rate[OIL] + conditions.vol_flow_rate[WATER]],
        [gas_speed, liquid_speed])


# ============================================================
# Пример использования
# ============================================================

if __name__ == "__main__":
    from CoolProp import constants as CoolConst
    from pytroleum.plant.tps.utils import SECONDS_PER_DAY, _major_header, _minor_header

    pressure = 2e6
    temperature = 273.15 + 50
    vol_flow_gas_norm = 300_000 / SECONDS_PER_DAY
    vol_flow_oil = 200 / SECONDS_PER_DAY
    vol_flow_water = 300 / SECONDS_PER_DAY

    conditions = OperationConditions()
    conditions.phase[OIL].change(933, 3.073e-3)  # type: ignore
    conditions.phase[VAPOR].update(*STANDARD_STATE)
    gas_density_norm = conditions.phase[VAPOR].rhomass()
    conditions.update_state((CoolConst.PT_INPUTS, pressure, temperature),
                            upd_containers=True)
    conditions.vol_flow_rate = np.array([
        vol_flow_gas_norm * gas_density_norm /
        conditions.phase[VAPOR].rhomass(),
        vol_flow_oil, vol_flow_water,
    ])

    gas_speed = 10.0
    liquid_speed = 1.0

    gasnozzle = design_nozzle(conditions.vol_flow_rate[VAPOR], gas_speed)
    oil_nozzle = design_nozzle(conditions.vol_flow_rate[OIL], liquid_speed)
    water_nozzle = design_nozzle(conditions.vol_flow_rate[WATER], liquid_speed)
    vol_flow_liquid = conditions.vol_flow_rate[OIL] + \
        conditions.vol_flow_rate[WATER]
    liquid_nozzle = design_nozzle(vol_flow_liquid, liquid_speed)
    liquidgasnozzle = design_two_phase_nozzle(conditions=conditions,
                                              gas_speed=gas_speed,
                                              liquid_speed=liquid_speed)

    _major_header("РАСЧЕТ ШТУЦЕРОВ")

    _minor_header("Штуцер газа")
    print(f"Скорость: {gas_speed:.2f} м/с")
    print(f"Расчетный диаметр: {gasnozzle.diameter * _TO_MM:.1f} мм")
    print(f"Стандартный диаметр: {gasnozzle.nominal_diameter * _TO_MM:.0f} мм")
    print(f"Площадь сечения: {gasnozzle.nominal_area:.4f} м²")
    print(f"Фактическая скорость: "
          f"{gasnozzle.flow_velocity(conditions.vol_flow_rate[VAPOR]):.4f} м/с")
    print()
    _minor_header("Штуцер нефти")
    print(f"Скорость: {liquid_speed:.2f} м/с")
    print(f"Расчетный диаметр: {oil_nozzle.diameter * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {oil_nozzle.nominal_diameter * _TO_MM:.0f} мм")
    print(f"Площадь сечения: {oil_nozzle.nominal_area:.4f} м²")
    print(f"Фактическая скорость: "
          f"{oil_nozzle.flow_velocity(conditions.vol_flow_rate[OIL]):.4f} м/с")

    print()
    _minor_header("Штуцер воды")
    print(f"Скорость: {liquid_speed:.2f} м/с")
    print(f"Расчетный диаметр: {water_nozzle.diameter * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {water_nozzle.nominal_diameter * _TO_MM:.0f} мм")
    print(f"Площадь сечения: {water_nozzle.nominal_area:.4f} м²")
    print(f"Фактическая скорость: "
          f"{water_nozzle.flow_velocity(conditions.vol_flow_rate[WATER]):.4f} м/с")

    print()
    _minor_header("Штуцер жидкости")
    print(f"Скорость: {liquid_speed:.2f} м/с")
    print(f"Расчетный диаметр: {liquid_nozzle.diameter * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {liquid_nozzle.nominal_diameter * _TO_MM:.0f} мм")
    print(f"Площадь сечения: {liquid_nozzle.nominal_area:.4f} м²")
    print(f"Фактическая скорость: "
          f"{liquid_nozzle.flow_velocity(vol_flow_liquid):.4f} м/с")

    print()
    _minor_header("Штуцер ГЖС")
    print(f"Скорость газа: {gas_speed:.2f} м/с")
    print(f"Скорость жидкости: {liquid_speed:.2f} м/с")
    print(f"Расчетный диаметр: {liquidgasnozzle.diameter * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {liquidgasnozzle.nominal_diameter * _TO_MM:.0f} мм")
    print(f"Площадь сечения: {liquidgasnozzle.nominal_area:.4f} м²")
    print(f"Фактическая скорость: "
          f"{liquidgasnozzle.flow_velocity(conditions.vol_flow_rate[VAPOR]):.4f} м/с")
