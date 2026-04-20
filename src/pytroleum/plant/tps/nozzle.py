import numpy as np
from pytroleum.plant.tps.inputs import FlowRates
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


def _select_nominal_diameter(diameter: float) -> float:
    for d_nom in sorted(NOMINAL_DIAMETERS):
        if d_nom >= diameter:
            return d_nom
    raise ValueError(
        f"Расчетный диаметр {diameter * _TO_MM:.1f} мм "
        f"больше {NOMINAL_DIAMETERS[-1] * _TO_MM:.0f} мм")


def _validate_velocity(name: str, velocity: float,
                       min_velocity: float, max_velocity: float) -> None:
    if not min_velocity <= velocity <= max_velocity:
        print(f"Предупреждение: скорость в {name} "
              f"{velocity} м/с вне диапазона [{min_velocity}, {max_velocity}] м/с")


class Nozzle:
    """Класс описывает патрубок"""

    def __init__(self, diameter: float, resistance_coeff: float | None = None) -> None:
        self.diameter = diameter
        self.area = np.pi * diameter ** 2 / 4

        self.resistance_coeff = resistance_coeff
        self.nominal_diameter = _select_nominal_diameter(diameter)
        self.nominal_area = np.pi * self.nominal_diameter ** 2 / 4

    def flow_velocity(self, volumetric_flow_rate: float) -> float:
        """Скорость потока в штуцере по номинальному диаметру, м/с"""
        return volumetric_flow_rate / self.nominal_area


def design_nozzle(volumetric_flow_rate, target_velocity,
                  resistance_coeff: float | None = None) -> Nozzle:
    """Для определения штуцера по рабочим параметрам отдельная функция"""

    volumetric_flow_rate = np.atleast_1d(volumetric_flow_rate)
    target_velocity = np.atleast_1d(target_velocity)

    if len(volumetric_flow_rate) == 1:
        # одно значение, однофазный поток
        diameter = np.sqrt(4 * volumetric_flow_rate[0] /
                           (np.pi * target_velocity[0]))
    else:
        # длина больше 1 => двухфазный поток
        diameter = 1.13 * np.sqrt(
            np.sum(volumetric_flow_rate / (1.3 * target_velocity))
        )

    return Nozzle(diameter, resistance_coeff)


def design_gas_nozzle(flows: FlowRates, speed: float,
                      resistance_coeff: float) -> Nozzle:
    _validate_velocity("Штуцер газа", speed,
                       MIN_GAS_VELOCITY, MAX_GAS_VELOCITY)
    return design_nozzle(flows.flow_gas_work, speed, resistance_coeff)


def design_liquid_nozzle(flows: FlowRates, speed: float) -> Nozzle:
    _validate_velocity("Штуцер жидкости", speed,
                       MIN_LIQUID_VELOCITY, MAX_LIQUID_VELOCITY)
    return design_nozzle(flows.conditions.flow_liquid, speed)


def design_oil_nozzle(flows: FlowRates, speed: float) -> Nozzle:
    _validate_velocity("Штуцер нефти", speed,
                       MIN_LIQUID_VELOCITY, MAX_LIQUID_VELOCITY)
    return design_nozzle(flows.flow_oil, speed)


def design_water_nozzle(flows: FlowRates, speed: float) -> Nozzle:
    _validate_velocity("Штуцер воды", speed,
                       MIN_LIQUID_VELOCITY, MAX_LIQUID_VELOCITY)
    return design_nozzle(flows.flow_water, speed)


def design_liquid_gas_nozzle(flows: FlowRates, gas_speed: float,
                             liquid_speed: float,
                             resistance_coeff: float) -> Nozzle:
    _validate_velocity("Штуцер ГЖС (газ)", gas_speed,
                       MIN_GAS_VELOCITY, MAX_GAS_VELOCITY)
    _validate_velocity("Штуцер ГЖС (жидкость)", liquid_speed,
                       MIN_LIQUID_VELOCITY, MAX_LIQUID_VELOCITY)
    return design_nozzle(
        [flows.flow_gas_work, flows.conditions.flow_liquid],
        [gas_speed, liquid_speed],
        resistance_coeff,
    )

# ============================================================
# Пример использования
# ============================================================


if __name__ == "__main__":
    from pytroleum.plant.tps.inputs import (
        OperationConditions, PhysicalProperties, FlowRates,
    )
    from pytroleum.plant.tps.utils import (
        SECONDS_PER_DAY, _major_header, _minor_header,
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

    gas_speed = 10.0
    liquid_speed = 1.0

    gasnozzle = design_gas_nozzle(flows=flows,
                                  speed=gas_speed,
                                  resistance_coeff=0.5)
    oil_nozzle = design_oil_nozzle(flows=flows, speed=liquid_speed)
    water_nozzle = design_water_nozzle(flows=flows, speed=liquid_speed)
    liquid_nozzle = design_liquid_nozzle(flows=flows, speed=liquid_speed)
    liquidgasnozzle = design_liquid_gas_nozzle(flows=flows,
                                               gas_speed=gas_speed,
                                               liquid_speed=liquid_speed,
                                               resistance_coeff=1.0)

    _major_header("РАСЧЕТ ШТУЦЕРОВ")

    _minor_header("Штуцер газа")
    print(f"Скорость: {gas_speed:.2f} м/с")
    print(f"Расчетный диаметр: {gasnozzle.diameter * _TO_MM:.1f} мм")
    print(f"Стандартный диаметр: {gasnozzle.nominal_diameter * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {gasnozzle.nominal_area:.4f} м²")
    print(
        f"Фактическая скорость: {gasnozzle.flow_velocity(flows.flow_gas_work):.4f} м/с")

    print()
    _minor_header("Штуцер нефти")
    print(f"Скорость: {liquid_speed:.2f} м/с")
    print(f"Расчетный диаметр:  {oil_nozzle.diameter * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {oil_nozzle.nominal_diameter * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {oil_nozzle.nominal_area:.4f} м²")
    print(
        f"Фактическая скорость: {oil_nozzle.flow_velocity(flows.flow_oil):.4f} м/с")

    print()
    _minor_header("Штуцер воды")
    print(f"Скорость: {liquid_speed:.2f} м/с")
    print(f"Расчетный диаметр: {water_nozzle.diameter * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {water_nozzle.nominal_diameter * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {water_nozzle.nominal_area:.4f} м²")
    print(
        f"Фактическая скорость: {water_nozzle.flow_velocity(flows.flow_water):.4f} м/с")

    print()
    _minor_header("Штуцер жидкости")
    print(f"Скорость: {liquid_speed:.2f} м/с")
    print(f"Расчетный диаметр: {liquid_nozzle.diameter * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {liquid_nozzle.nominal_diameter * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {liquid_nozzle.nominal_area:.4f} м²")
    print(f"Фактическая скорость: "
          f"{liquid_nozzle.flow_velocity(flows.conditions.flow_liquid):.4f} м/с")

    print()
    _minor_header("Штуцер ГЖС")
    print(f"Скорость газа: {gas_speed:.2f} м/с")
    print(f"Скорость жидкости: {liquid_speed:.2f} м/с")
    print(f"Расчетный диаметр: {liquidgasnozzle.diameter * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {liquidgasnozzle.nominal_diameter * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {liquidgasnozzle.nominal_area:.4f} м²")
    print(f"Фактическая скорость: "
          f"{liquidgasnozzle.flow_velocity(flows.flow_gas_work):.4f} м/с")
