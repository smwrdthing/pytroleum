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


class Nozzle:
    """Расчёт пропускной способности штуцера"""

    def __init__(self, name: str, flow_rate: float, speed: float,
                 min_speed: float, max_speed: float):
        self.name = name
        self.flow_rate = flow_rate
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed = speed

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float):
        if not self.min_speed <= value <= self.max_speed:
            print(f"Предупреждение: скорость в {self.name} "
                  f"{value} м/с вне диапазона [{self.min_speed}, {self.max_speed}] м/с")
        self._speed = value

    def calculate_diameter(self) -> float:
        return np.sqrt(4 * self.flow_rate / (np.pi * self.speed))

    def select_nominal_diameter(self) -> float:
        diameter = self.calculate_diameter()
        for d_nom in sorted(NOMINAL_DIAMETERS):
            if d_nom >= diameter:
                return d_nom
        raise ValueError(
            f"Расчетный диаметр {diameter*_TO_MM:.1f} мм "
            f"больше {NOMINAL_DIAMETERS[-1]*_TO_MM:.0f} мм")

    def nozzle_area(self) -> float:
        """Площадь сечения штуцера по номинальному диаметру, м²"""
        diameter_nominal = self.select_nominal_diameter()
        # NOTE атрибутом?
        return np.pi * diameter_nominal ** 2 / 4

    def actual_speed(self) -> float:
        """Фактическая скорость в штуцере, м/с"""
        return self.flow_rate / self.nozzle_area()


class GasNozzle(Nozzle):
    """Расчёт штуцера газа"""

    def __init__(self, flows: FlowRates, speed: float):
        super().__init__("Штуцер газа", flows.flow_gas_work(), speed,
                         min_speed=MIN_GAS_VELOCITY, max_speed=MAX_GAS_VELOCITY)


class LiquidNozzle(Nozzle):
    """Расчёт штуцера жидкости"""

    def __init__(self, flows: FlowRates, speed: float):
        super().__init__("Штуцер жидкости", flows.conditions.flow_liquid,
                         speed, min_speed=MIN_LIQUID_VELOCITY,
                         max_speed=MAX_LIQUID_VELOCITY)


class OilNozzle(LiquidNozzle):
    """Расчёт штуцера нефти"""

    def __init__(self, flows: FlowRates, speed: float):
        Nozzle.__init__(self, "Штуцер нефти", flows.flow_oil(),
                        speed, min_speed=MIN_LIQUID_VELOCITY,
                        max_speed=MAX_LIQUID_VELOCITY)


class WaterNozzle(LiquidNozzle):
    """Расчёт штуцера воды"""

    def __init__(self, flows: FlowRates, speed: float):
        Nozzle.__init__(self, "Штуцер воды", flows.flow_water(),
                        speed, min_speed=MIN_LIQUID_VELOCITY,
                        max_speed=MAX_LIQUID_VELOCITY)


class LiquidGasNozzle(Nozzle):
    """Расчёт штуцера газожидкостной смеси."""

    def __init__(self, flows: FlowRates, gas_speed: float, liquid_speed: float):
        self.name = "Штуцер ГЖС"
        self.flow_rate = flows.flow_gas_work()
        self.liquid_flow_rate = flows.conditions.flow_liquid
        self.gas_speed = gas_speed
        self.liquid_speed = liquid_speed

    def calculate_diameter(self) -> float:
        return 1.13 * np.sqrt(self.flow_rate / (1.3 * self.gas_speed) +
                              self.liquid_flow_rate / (1.3 * self.liquid_speed))

    @property
    def gas_speed(self) -> float:
        return self._gas_speed

    @gas_speed.setter
    def gas_speed(self, value: float):
        if not MIN_GAS_VELOCITY <= value <= MAX_GAS_VELOCITY:
            print(f"Предупреждение: скорость газа в {self.name} "
                  f"{value} м/с вне диапазона "
                  f"[{MIN_GAS_VELOCITY}, {MAX_GAS_VELOCITY}] м/с")
        self._gas_speed = value

    @property
    def liquid_speed(self) -> float:
        return self._liquid_speed

    @liquid_speed.setter
    def liquid_speed(self, value: float):
        if not MIN_LIQUID_VELOCITY <= value <= MAX_LIQUID_VELOCITY:
            print(f"Предупреждение: скорость жидкости в {self.name} "
                  f"{value} м/с вне диапазона "
                  f"[{MIN_LIQUID_VELOCITY}, {MAX_LIQUID_VELOCITY}] м/с")
        self._liquid_speed = value

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

    gasnozzle = GasNozzle(flows=flows, speed=10.0)
    oil_nozzle = OilNozzle(flows=flows, speed=1.0)
    water_nozzle = WaterNozzle(flows=flows, speed=1.0)
    liquid_nozzle = LiquidNozzle(flows=flows, speed=1.0)
    liquidgasnozzle = LiquidGasNozzle(
        flows=flows, gas_speed=10.0, liquid_speed=1.0)

    _major_header("РАСЧЕТ ШТУЦЕРОВ")

    _minor_header(gasnozzle.name)
    print(f"Скорость: {gasnozzle.speed:.2f} м/с")
    print(
        f"Расчетный диаметр: {gasnozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {gasnozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {gasnozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость: {gasnozzle.actual_speed():.4f} м/с")

    print()
    _minor_header(oil_nozzle.name)
    print(f"Скорость: {oil_nozzle.speed:.2f} м/с")
    print(
        f"Расчетный диаметр:  {oil_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {oil_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {oil_nozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость: {oil_nozzle.actual_speed():.4f} м/с")

    print()
    _minor_header(water_nozzle.name)
    print(
        f"Скорость: {water_nozzle.speed:.2f} м/с")
    print(
        f"Расчетный диаметр: {water_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {water_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {water_nozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость: {water_nozzle.actual_speed():.4f} м/с")

    print()
    _minor_header(liquid_nozzle.name)
    print(
        f"Скорость: {liquid_nozzle.speed:.2f} м/с")
    print(
        f"Расчетный диаметр: {liquid_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {liquid_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {liquid_nozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость: {liquid_nozzle.actual_speed():.4f} м/с")

    print()
    _minor_header(liquidgasnozzle.name)
    print(
        f"Скорость газа: {liquidgasnozzle.gas_speed:.2f} м/с")
    print(
        f"Скорость жидкости: {liquidgasnozzle.liquid_speed:.2f} м/с")
    print(
        f"Расчетный диаметр: {liquidgasnozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(f"Стандартный диаметр: "
          f"{liquidgasnozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(
        f"Площадь сечения штуцера: {liquidgasnozzle.nozzle_area():.4f} м²")
    print(
        f"Фактическая скорость: {liquidgasnozzle.actual_speed():.4f} м/с")
