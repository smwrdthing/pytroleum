import numpy as np
from abc import ABC, abstractmethod
from pytroleum.plant.tps.inputs import FlowRates
from pytroleum.plant.tps.report import _minor_divider, _major_divider

_TO_MM = 1000

# Андрей сказал
MAX_GAS_VELOCITY = 20  # м/с
MIN_GAS_VELOCITY = 10  # м/с

# СТК 542 РР1 Сепаратор режим 2 (высоковязкая нефть)
MAX_LIQUID_VELOCITY = 3  # м/с
MIN_LIQUID_VELOCITY = 1  # м/с

# Таблица с номинальными диаметрами:
# https://dpva.ru/guide/guideequipment/connections/diameters/pipeoutsidediametercorrespondance/

NOMINAL_DIAMETERS = [
    10e-3, 15e-3, 20e-3, 25e-3, 32e-3, 40e-3, 50e-3, 65e-3, 80e-3,
    90e-3, 100e-3, 125e-3, 150e-3, 200e-3, 225e-3,
    250e-3, 300e-3, 400e-3, 500e-3, 600e-3, 800e-3, 1000e-3, 1200e-3
]


class Nozzle(ABC):
    """Calculation of the nozzle capacity"""

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
        pass

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
        return np.pi * diameter_nominal ** 2 / 4

    def actual_speed(self) -> float:
        """Фактическая скорость в штуцере, м/с"""
        return self.flow_rate / self.nozzle_area()


class GasNozzle(Nozzle):
    """Calculation of the gas nozzle capacity."""

    MIN_SPEED = MIN_GAS_VELOCITY
    MAX_SPEED = MAX_GAS_VELOCITY

    def __init__(self, flows: FlowRates, recommended_speed: float):
        super().__init__("Штуцер газа", flows.flow_gas_work(), recommended_speed)

    def calculate_diameter(self) -> float:
        return np.sqrt(4 * self.flow_rate / (np.pi * self.recommended_speed))


class LiquidNozzle(Nozzle):
    """Calculation of the liquid nozzle capacity."""

    MIN_SPEED = MIN_LIQUID_VELOCITY
    MAX_SPEED = MAX_LIQUID_VELOCITY

    def __init__(self, flows: FlowRates, recommended_speed: float):
        super().__init__("Штуцер жидкости",
                         flows.conditions.flow_liquid, recommended_speed)

    def calculate_diameter(self) -> float:
        return np.sqrt(4 * self.flow_rate / (np.pi * self.recommended_speed))


class OilNozzle(LiquidNozzle):
    """Calculation of the oil nozzle capacity."""

    def __init__(self, flows: FlowRates, recommended_speed: float):
        Nozzle.__init__(self, "Штуцер нефти",
                        flows.flow_oil(), recommended_speed)


class WaterNozzle(LiquidNozzle):
    """Calculation of the water nozzle capacity."""

    def __init__(self, flows: FlowRates, recommended_speed: float):
        Nozzle.__init__(self, "Штуцер воды",
                        flows.flow_water(), recommended_speed)


class LiquidGasNozzle(Nozzle):
    """Расчёт штуцера для газожидкостной смеси."""

    MIN_SPEED = MIN_GAS_VELOCITY
    MAX_SPEED = MAX_GAS_VELOCITY
    MIN_LIQUID_SPEED = MIN_LIQUID_VELOCITY
    MAX_LIQUID_SPEED = MAX_LIQUID_VELOCITY

    def __init__(self, flows: FlowRates, gas_speed: float, liquid_speed: float):
        super().__init__("Штуцер ГЖС", flows.flow_gas_work(), gas_speed)
        self.liquid_flow_rate = flows.conditions.flow_liquid
        self.liquid_speed = liquid_speed

    def calculate_diameter(self) -> float:
        """Расчёт диаметра для двухфазной среды по специальной формуле."""
        return 1.13 * np.sqrt(self.flow_rate / (1.3 * self.recommended_speed) +
                              self.liquid_flow_rate / (1.3 * self.liquid_speed))

    def actual_speed(self) -> float:
        """Фактическая скорость газа в штуцере, м/с"""
        return self.flow_rate / self.nozzle_area()

    @property
    def gas_speed(self) -> float:
        return self.recommended_speed

    @property
    def liquid_speed(self) -> float:
        return self._liquid_speed

    @liquid_speed.setter
    def liquid_speed(self, value: float):
        if not self.MIN_LIQUID_SPEED <= value <= self.MAX_LIQUID_SPEED:
            print(f"Предупреждение: скорость жидкости в {self.name} "
                  f"{value} м/с вне диапазона "
                  f"[{self.MIN_LIQUID_SPEED}, {self.MAX_LIQUID_SPEED}] м/с")
        self._liquid_speed = value


if __name__ == '__main__':
    from pytroleum.plant.tps.inputs import (OperationConditions,
                                            PhysicalProperties,
                                            FlowRates)
    con = OperationConditions(
        pressure_work=1e6,
        temperature_work=353,
        flow_gas_norm=300000 / 86400,
        flow_liquid=500 / 86400,
    )
    props = PhysicalProperties(
        gas_density_norm=0.94,
        oil_density=933,
        water_density=966,
        water_cut=0.6,
        gas_factor=267.9,
    )

    flows = FlowRates(conditions=con, properties=props)

    gas_nozzle = GasNozzle(flows=flows, recommended_speed=10.0)
    oil_nozzle = OilNozzle(flows=flows, recommended_speed=1.0)
    water_nozzle = WaterNozzle(flows=flows, recommended_speed=1.0)
    liquid_nozzle = LiquidNozzle(flows=flows, recommended_speed=1.0)
    liquid_gas_nozzle = LiquidGasNozzle(
        flows=flows, gas_speed=10.0, liquid_speed=1.0)

    _major_divider()
    print(gas_nozzle.name)
    _major_divider()

    print(f"Рекомендуемая скорость: {gas_nozzle.recommended_speed:.2f} м/с")
    print(
        f"Расчетный диаметр:{gas_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр:{gas_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера:{gas_nozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость:{gas_nozzle.actual_speed():.4f} м/с")

    _major_divider()
    print(oil_nozzle.name)
    _major_divider()

    print(f"Рекомендуемая скорость:{oil_nozzle.recommended_speed:.2f} м/с")
    print(
        f"Расчетный диаметр:{oil_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр:{oil_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера:{oil_nozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость:{oil_nozzle.actual_speed():.4f} м/с")

    _major_divider()
    print(water_nozzle.name)
    _major_divider()

    print(f"Рекомендуемая скорость:{water_nozzle.recommended_speed:.2f} м/с")
    print(
        f"Расчетный диаметр:{water_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр:{water_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера:{water_nozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость:{water_nozzle.actual_speed():.4f} м/с")

    _major_divider()
    print(liquid_nozzle.name)
    _major_divider()

    print(f"Рекомендуемая скорость:{liquid_nozzle.recommended_speed:.2f} м/с")
    print(
        f"Расчетный диаметр:{liquid_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(
        f"Стандартный диаметр:{liquid_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера:{liquid_nozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость:{liquid_nozzle.actual_speed():.4f} м/с")

    _major_divider()
    print(liquid_gas_nozzle.name)
    _major_divider()

    print(f"Рекомендуемая скорость газа:{liquid_gas_nozzle.gas_speed:.2f} м/с")
    print(
        f"Рекомендуемая скорость жидкости:{liquid_gas_nozzle.liquid_speed:.2f} м/с")
    print(
        f"Расчетный диаметр:{liquid_gas_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(f"Стандартный диаметр: "
          f"{liquid_gas_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Площадь сечения штуцера:{liquid_gas_nozzle.nozzle_area():.4f} м²")
    print(f"Фактическая скорость:{liquid_gas_nozzle.actual_speed():.4f} м/с")
