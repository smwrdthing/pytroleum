from dataclasses import dataclass

DEFAULT_PRESSURE = 0.1e6      # Па
DEFAULT_TEMPERATURE = 293     # К (20°C)
SECONDS_PER_DAY = 86400
KG_PER_TON = 1000
KG_S_TO_T_H = 3.6
PERCENT = 100
PA_TO_MPA = 1e6
SECONDS_PER_HOUR = 3600


@dataclass
class OperationConditions:
    pressure_work: float        # Па - рабочее давление
    temperature_work: float     # К - рабочая температура
    flow_gas_norm: float        # м³/с - объемный расход газа при н.у.
    flow_liquid: float          # м³/с - объемный расход жидкости


@dataclass
class PhysicalProperties:
    gas_density_norm: float     # кг/м³ - плотность газа
    oil_density: float          # кг/м³ - плотность нефти
    water_density: float        # кг/м³ - плотность воды
    water_cut: float            # Обводненность
    gas_factor: float           # Газовый фактор, м³/т
    oil_surface_tension: float  # Поверхностное натяжение нефти, Н/м

    def liquid_density(self) -> float:
        """Плотность жидкости (Н+В) при заданной обводненности"""
        return (self.water_density * self.water_cut +
                self.oil_density * (1 - self.water_cut))

    def gas_density_work(self, conditions: OperationConditions) -> float:
        """Плотность газа в рабочих условиях"""
        return self.gas_density_norm * (conditions.pressure_work / DEFAULT_PRESSURE) * \
            (DEFAULT_TEMPERATURE / conditions.temperature_work)


@dataclass
class Coefficients:
    # Коэффициент учитывающий снижение площади сечения элементами насадки
    area_reduction_coefficient: float
    mesh_resistance_coefficient: float
    inlet_resistance_coefficient: float
    outlet_resistance_coefficient: float
    losses_unaccounted: float


@dataclass
class FlowRates:
    conditions: OperationConditions
    properties: PhysicalProperties

    def flow_oil(self) -> float:
        """ Расход нефти, м³/с """
        return self.conditions.flow_liquid * (1 - self.properties.water_cut)

    def flow_water(self) -> float:
        """ Расход воды, м³/с """
        return self.conditions.flow_liquid * self.properties.water_cut

    def flow_gas_work(self) -> float:
        """Расход газа при рабочих условиях"""
        return (self.conditions.flow_gas_norm*DEFAULT_PRESSURE *
                self.conditions.temperature_work) / \
            ((self.conditions.pressure_work+DEFAULT_PRESSURE) *
             DEFAULT_TEMPERATURE)

    def mass_flow_gas(self) -> float:
        """Массовый расход газа, кг/с"""
        return self.conditions.flow_gas_norm * self.properties.gas_density_norm

    def mass_flow_oil(self) -> float:
        """Массовый расход нефти, кг/с"""
        return self.flow_oil() * self.properties.oil_density

    def mass_flow_water(self) -> float:
        """Массовый расход воды, кг/с"""
        return self.flow_water() * self.properties.water_density

    def mass_flow_liquid(self) -> float:
        """Массовый расход жидкости (нефть + вода) в кг/с"""
        return self.mass_flow_oil() + self.mass_flow_water()

    def mass_flow_total(self) -> float:
        """Суммарный массовый расход (Г+Н+В) в кг/с"""
        return self.mass_flow_gas() + self.mass_flow_oil() + self.mass_flow_water()

    def mass_flow_oil_ton_per_day(self) -> float:
        """Массовый расход нефти, т/сут"""
        return self.mass_flow_oil() * SECONDS_PER_DAY / KG_PER_TON

    def flow_gas_from_gas_factor(self) -> float:
        """Объемный расход по газу из условия газового фактора, м³/сут"""
        return self.properties.gas_factor * self.mass_flow_oil_ton_per_day()
