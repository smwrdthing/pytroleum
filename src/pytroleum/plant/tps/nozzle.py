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

NOMINAL_DIAMETERS = [
    10e-3, 15e-3, 20e-3, 25e-3, 32e-3, 40e-3, 50e-3, 65e-3, 80e-3,
    90e-3, 100e-3, 125e-3, 150e-3, 200e-3, 225e-3,
    250e-3, 300e-3, 400e-3, 500e-3, 600e-3, 800e-3, 1000e-3, 1200e-3
]


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

    @liquid_speed.setter
    def liquid_speed(self, value: float):
        if not self.MIN_LIQUID_SPEED <= value <= self.MAX_LIQUID_SPEED:
            print(f"Предупреждение: скорость жидкости в {self.name} "
                  f"{value} м/с вне диапазона "
                  f"[{self.MIN_LIQUID_SPEED}, {self.MAX_LIQUID_SPEED}] м/с")
        self._liquid_speed = value
