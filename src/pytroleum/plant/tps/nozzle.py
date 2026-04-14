import numpy as np
from abc import ABC, abstractmethod
from pytroleum.plant.tps.inputs import FlowRates
from pytroleum.plant.tps.utils import _TO_MM

# Андрей сказал
MAX_GAS_VELOCITY = 20  # м/с
MIN_GAS_VELOCITY = 10  # м/с

# СТК 542 РР1 Сепаратор режим 2 (высоковязкая нефть)
MAX_LIQUID_VELOCITY = 3  # м/с
MIN_LIQUID_VELOCITY = 1  # м/с

# Таблица с номинальными диаметрами:
# https://dpva.ru/guide/guideequipment/connections/diameters/pipeoutsidediametercorrespondance/

NOMINAL_DIAMETERS = np.array([
    10, 15, 20, 25, 32, 40, 50, 65, 80, 90, 100, 125, 150,
    200, 225, 250, 300, 400, 500, 600, 800, 1000, 1200]) * 1e-3


class Nozzle(ABC):
    """Расчёт пропускной способности штуцера"""

    MIN_SPEED: float
    MAX_SPEED: float

    def __init__(self, name: str, flow_rate: float, recommended_speed: float):
        self.name = name
        self.flow_rate = flow_rate
        self.recommended_speed = recommended_speed

    @property
    def recommended_speed(self) -> float:
        return self._recommended_speed

    @recommended_speed.setter
    def recommended_speed(self, value: float):
        if not self.MIN_SPEED <= value <= self.MAX_SPEED:
            print(f"Предупреждение: рекомендуемая скорость в {self.name} "
                  f"{value} м/с вне диапазона [{self.MIN_SPEED}, {self.MAX_SPEED}] м/с")
        self._recommended_speed = value

    @abstractmethod
    def calculate_diameter(self) -> float:
        ...

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

    MIN_SPEED = MIN_GAS_VELOCITY
    MAX_SPEED = MAX_GAS_VELOCITY

    def __init__(self, flows: FlowRates, recommended_speed: float):
        super().__init__("Штуцер газа", flows.flow_gas_work(), recommended_speed)

    def calculate_diameter(self) -> float:
        return np.sqrt(4 * self.flow_rate / (np.pi * self.recommended_speed))


class LiquidNozzle(Nozzle):
    """Расчёт штуцера жидкости"""

    MIN_SPEED = MIN_LIQUID_VELOCITY
    MAX_SPEED = MAX_LIQUID_VELOCITY

    def __init__(self, flows: FlowRates, recommended_speed: float):
        super().__init__("Штуцер жидкости",
                         flows.conditions.flow_liquid, recommended_speed)

    def calculate_diameter(self) -> float:
        return np.sqrt(4 * self.flow_rate / (np.pi * self.recommended_speed))

# NOTE методы calculate_diameter одинаковые у GasNozzle и у LiquidNozzle


class OilNozzle(LiquidNozzle):
    """Расчёт штуцера нефти"""

    def __init__(self, flows: FlowRates, recommended_speed: float):
        Nozzle.__init__(self, "Штуцер нефти",
                        flows.flow_oil(), recommended_speed)


class WaterNozzle(LiquidNozzle):
    """Расчёт штуцера воды"""

    def __init__(self, flows: FlowRates, recommended_speed: float):
        Nozzle.__init__(self, "Штуцер воды",
                        flows.flow_water(), recommended_speed)


class LiquidGasNozzle(Nozzle):
    """Расчёт штуцера газожидкостной смеси."""

    MIN_SPEED = MIN_GAS_VELOCITY
    MAX_SPEED = MAX_GAS_VELOCITY
    MIN_LIQUID_SPEED = MIN_LIQUID_VELOCITY
    MAX_LIQUID_SPEED = MAX_LIQUID_VELOCITY

    def __init__(self, flows: FlowRates, gas_speed: float, liquid_speed: float):
        super().__init__("Штуцер ГЖС", flows.flow_gas_work(), gas_speed)
        self.liquid_flow_rate = flows.conditions.flow_liquid
        self.liquid_speed = liquid_speed

    def calculate_diameter(self) -> float:
        return 1.13 * np.sqrt(self.flow_rate / (1.3 * self.recommended_speed) +
                              self.liquid_flow_rate / (1.3 * self.liquid_speed))

    def actual_speed(self) -> float:
        """Фактическая скорость в штуцере, м/с"""
        return self.flow_rate / self.nozzle_area()

    @property
    def gas_speed(self) -> float:
        return self.recommended_speed

    @property
    def liquid_speed(self) -> float:
        return self._liquid_speed

    # NOTE setter для gas speed?

    @liquid_speed.setter
    def liquid_speed(self, value: float):
        if not self.MIN_LIQUID_SPEED <= value <= self.MAX_LIQUID_SPEED:
            print(f"Предупреждение: скорость жидкости в {self.name} "
                  f"{value} м/с вне диапазона "
                  f"[{self.MIN_LIQUID_SPEED}, {self.MAX_LIQUID_SPEED}] м/с")
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

    gasnozzle = GasNozzle(flows=flows, recommended_speed=10.0)
    oil_nozzle = OilNozzle(flows=flows, recommended_speed=1.0)
    water_nozzle = WaterNozzle(flows=flows, recommended_speed=1.0)
    liquid_nozzle = LiquidNozzle(flows=flows, recommended_speed=1.0)
    liquidgasnozzle = LiquidGasNozzle(
        flows=flows, gas_speed=10.0, liquid_speed=1.0)

    _major_header("РАСЧЕТ ШТУЦЕРОВ")

    _minor_header(gasnozzle.name)
    print(f"Рекомендуемая скорость: {gasnozzle.recommended_speed:.2f} м/с")
    print(
        f"Расчетный диаметр: {gasnozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {gasnozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {gasnozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость: {gasnozzle.actual_speed():.4f} м/с")

    print()
    _minor_header(oil_nozzle.name)
    print(f"Рекомендуемая скорость: {oil_nozzle.recommended_speed:.2f} м/с")
    print(
        f"Расчетный диаметр:  {oil_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {oil_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {oil_nozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость: {oil_nozzle.actual_speed():.4f} м/с")

    print()
    _minor_header(water_nozzle.name)
    print(
        f"Рекомендуемая скорость: {water_nozzle.recommended_speed:.2f} м/с")
    print(
        f"Расчетный диаметр: {water_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {water_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {water_nozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость: {water_nozzle.actual_speed():.4f} м/с")

    print()
    _minor_header(liquid_nozzle.name)
    print(
        f"Рекомендуемая скорость: {liquid_nozzle.recommended_speed:.2f} м/с")
    print(
        f"Расчетный диаметр: {liquid_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр: {liquid_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера: {liquid_nozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость: {liquid_nozzle.actual_speed():.4f} м/с")

    print()
    _minor_header(liquidgasnozzle.name)
    print(
        f"Рекомендуемая скорость газа: {liquidgasnozzle.gas_speed:.2f} м/с")
    print(
        f"Рекомендуемая скорость жидкости: {liquidgasnozzle.liquid_speed:.2f} м/с")
    print(
        f"Расчетный диаметр: {liquidgasnozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(f"Стандартный диаметр: "
          f"{liquidgasnozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(
        f"Площадь сечения штуцера: {liquidgasnozzle.nozzle_area():.4f} м²")
    print(
        f"Фактическая скорость: {liquidgasnozzle.actual_speed():.4f} м/с")
