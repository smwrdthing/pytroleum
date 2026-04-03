import numpy as np


class Nozzle:
    """Calculation of the nozzle capacity
    """
    STANDARD_DIAMETERS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    def __init__(self, name: str, flow_rate: float, recommended_speed: float):
        self.name = name
        self.flow_rate = flow_rate
        self.recommended_speed = recommended_speed

    def calculate_diameter(self) -> float:
        return np.sqrt(4 * self.flow_rate / (np.pi * self.recommended_speed))

    def select_standard_diameter(self) -> float:
        d = self.calculate_diameter()
        for std in sorted(self.STANDARD_DIAMETERS):
            if std >= d:
                return std
        raise ValueError(
            f"Расчетный диаметр {d*1000:.1f} мм "
            f"больше {self.STANDARD_DIAMETERS[-1]*1000:.0f} мм"
        )

    def actual_speed(self) -> float:
        d = self.select_standard_diameter()
        return 4 * self.flow_rate / (np.pi * d ** 2)


class GasNozzle(Nozzle):
    """Сalculation of the gas nozzle capacity"""

    def __init__(self, flow_gas_work: float, recommended_speed: float = 10.0):
        super().__init__("Штуцер для газ", flow_gas_work, recommended_speed)


class OilNozzle(Nozzle):
    """Сalculation of the oil nozzle capacity"""

    def __init__(self, flow_oil: float, recommended_speed: float = 1.0):
        super().__init__("Штуцер для нефти", flow_oil, recommended_speed)


class LiquidNozzle(Nozzle):
    """Сalculation of the liquid nozzle capacity"""

    def __init__(self, flow_liquid: float, recommended_speed: float = 1.0):
        super().__init__("Штуцер для жидкости", flow_liquid, recommended_speed)


class WaterNozzle(Nozzle):
    """Сalculation of the liquid nozzle capacity"""

    def __init__(self, flow_water: float, recommended_speed: float = 1.0):
        super().__init__("Штуцер для жидкости", flow_water, recommended_speed)


class LiquidGasNozzle(Nozzle):
    pass
