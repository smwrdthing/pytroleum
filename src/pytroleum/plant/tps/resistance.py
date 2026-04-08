from pytroleum.plant.tps.utils import (_major_header,
                                       _minor_divider,
                                       SECONDS_PER_DAY,
                                       _TO_MM)
from pytroleum.plant.tps.inputs import (PhysicalProperties,
                                        OperationConditions,
                                        Coefficients,
                                        FlowRates)
from pytroleum.plant.tps.wire_mesh_demister import WireMeshDemister
from pytroleum.plant.tps.nozzle import LiquidGasNozzle, GasNozzle


class Resistance:
    """Сопротивление сепаратора"""

    def __init__(self, properties: PhysicalProperties,
                 coefficients: Coefficients,
                 demister: WireMeshDemister,
                 conditions: OperationConditions,
                 liquidgasnozzle: LiquidGasNozzle,
                 gasnozzle: GasNozzle):
        self.properties = properties
        self.coefficients = coefficients
        self.demister = demister
        self.conditions = conditions
        self.liquidgasnozzle = liquidgasnozzle
        self.gasnozzle = gasnozzle

    def pressure_drop_mesh_demister(self):
        """Падение давления на отбойнике"""
        return self.coefficients.mesh_resistance_coefficient *\
            self.properties.gas_density_work(self.conditions) *\
            self.demister.actual_velocity()**2/2

    def pressure_drop_inlet_nozzle(self):
        """Потери давления на штуцере входа ГЖС"""
        return self.coefficients.inlet_resistance_coefficient *\
            self.properties.gas_density_work(self.conditions) *\
            self.liquidgasnozzle.actual_speed()**2/2

    def pressure_drop_outlet_nozzle(self):
        """ Потери давления на штуцере газа"""
        return self.coefficients.outlet_resistance_coefficient *\
            self.properties.gas_density_work(self.conditions) *\
            self.gasnozzle.actual_speed()**2/2

    def separator_resistance(self):
        return self.coefficients.losses_unaccounted*(self.pressure_drop_mesh_demister() +
                                                     self.pressure_drop_inlet_nozzle() +
                                                     self.pressure_drop_outlet_nozzle())


if __name__ == "__main__":
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
        oil_surface_tension=0.02848
    )
    coefficients = Coefficients(
        area_reduction_coefficient=1.05,
        mesh_resistance_coefficient=70,
        inlet_resistance_coefficient=1,
        outlet_resistance_coefficient=0.5,
        losses_unaccounted=1.2
    )
    flows = FlowRates(conditions=conditions, properties=properties)
    demister = WireMeshDemister(properties, flows, coefficients)

    gasnozzle = GasNozzle(flows=flows, recommended_speed=10.0)
    liquidgasnozzle = LiquidGasNozzle(
        flows=flows, gas_speed=10.0, liquid_speed=1.0)

    resistance = Resistance(
        properties=properties,
        coefficients=coefficients,
        demister=demister,
        conditions=conditions,
        liquidgasnozzle=liquidgasnozzle,
        gasnozzle=gasnozzle,
    )
    # ВЫВОД РЕЗУЛЬТАТОВ
    _major_header("РАСЧЁТ СОПРОТИВЛЕНИЯ СЕПАРАТОРА")

    _minor_divider()
    print(f"Коэффициент сопротивления сетчатого отбойника: "
          f"{coefficients.mesh_resistance_coefficient}")
    print(f"Падение давления на отбойнике: "
          f"{resistance.pressure_drop_mesh_demister():.3f} Па")

    _minor_divider()
    print(f"Диаметр штуцера входа ГЖС: "
          f"{liquidgasnozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"Диаметр штуцера выхода газа: "
          f"{gasnozzle.select_nominal_diameter() * _TO_MM:.0f} мм")

    _minor_divider()
    print(f"Скорость во входном штуцере ГЖС: "
          f"{liquidgasnozzle.actual_speed():.3f} м/с")
    print(f"Скорость в выходном штуцере газа: "
          f"{gasnozzle.actual_speed():.3f} м/с")

    _minor_divider()
    print(f"Коэффициент сопротивления входного патрубка: "
          f"{coefficients.inlet_resistance_coefficient}")
    print(f"Коэффициент сопротивления выходного патрубка: "
          f"{coefficients.outlet_resistance_coefficient}")

    _minor_divider()
    print(f"Потери давления на штуцере входа: "
          f"{resistance.pressure_drop_inlet_nozzle():.3f} Па")
    print(f"Потери давления на штуцере выхода: "
          f"{resistance.pressure_drop_outlet_nozzle():.3f} Па")

    _minor_divider()
    print(f"Коэффициент неучтенных потерь: "
          f"{coefficients.losses_unaccounted}")

    _minor_divider()
    print(f"Сопротивление сепаратора: "
          f"{resistance.separator_resistance():.3f} Па")
