from pytroleum.plant.tps.utils import (_major_header,
                                       _minor_divider,
                                       SECONDS_PER_DAY,
                                       _TO_MM)
from pytroleum.plant.tps.inputs import (PhysicalProperties,
                                        OperationConditions,
                                        FlowRates, GeometryCyclone)
from pytroleum.plant.tps.wire_mesh_demister import WireMeshDemister
from pytroleum.plant.tps.nozzle import Nozzle, design_gas_nozzle, design_liquid_gas_nozzle

# NOTE возможно тут тоже часть методов получится перевести в атрибуты


class Resistance:
    """Расчет сопротивление сепаратора"""
    # NOTE может методом в классе сепаратора?

    def __init__(self, properties: PhysicalProperties,
                 flows: FlowRates,
                 losses_unaccounted: float,
                 demister: WireMeshDemister,
                 conditions: OperationConditions,
                 liquidgasnozzle: Nozzle,
                 gasnozzle: Nozzle):
        self.properties = properties
        self.flows = flows
        self.losses_unaccounted = losses_unaccounted  # Коэффициент неучтенных потерь
        self.demister = demister
        self.conditions = conditions
        self.liquidgasnozzle = liquidgasnozzle
        self.gasnozzle = gasnozzle

    def pressure_drop_mesh_demister(self):
        """Падение давления на отбойнике"""
        return (self.demister.mesh_resistance_coefficient *
                self.properties.gas_density_work(self.conditions) *
                self.demister.actual_velocity()**2 / 2)

    def pressure_drop_inlet_nozzle(self):
        """Потери давления на штуцере входа ГЖС"""
        assert self.liquidgasnozzle.resistance_coeff is not None
        return (self.liquidgasnozzle.resistance_coeff *
                self.properties.gas_density_work(self.conditions) *
                self.liquidgasnozzle.flow_velocity(self.flows.flow_gas_work)**2 / 2)

    def pressure_drop_outlet_nozzle(self):
        """Потери давления на штуцере газа"""
        assert self.gasnozzle.resistance_coeff is not None
        return (self.gasnozzle.resistance_coeff *
                self.properties.gas_density_work(self.conditions) *
                self.gasnozzle.flow_velocity(self.flows.flow_gas_work)**2 / 2)

    def separator_resistance(self):
        return self.losses_unaccounted * (self.pressure_drop_mesh_demister() +
                                          self.pressure_drop_inlet_nozzle() +
                                          self.pressure_drop_outlet_nozzle())


class Cyclone:
    """ Расчет скорости газа в сепарационном элементе (спиральный канал)"""

    def __init__(self, flows: FlowRates, geometry_cyclone: GeometryCyclone):
        self.flows = flows
        self.geometry_cyclone = geometry_cyclone

    def area_spiral_channel(self) -> float:
        return self.geometry_cyclone.width_inlet_cyclone *\
            self.geometry_cyclone.height_inlet_cyclone

    def velocity_gas_in_spiral_channel(self):
        return self.flows.flow_gas_work / (self.geometry_cyclone.number_of_cyclones *
                                           self.area_spiral_channel())

# ============================================================
# Пример использования
# ============================================================


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
        oil_surface_tension=0.02848,
        viscosity_oil=3.073e-3,
        viscosity_water=0.544e-3
    )
    flows = FlowRates(conditions=conditions, properties=properties)
    demister = WireMeshDemister(properties, flows,
                                area_reduction_coefficient=1.05,
                                mesh_resistance_coefficient=70)

    gasnozzle = design_gas_nozzle(flows=flows,
                                  speed=10.0,
                                  resistance_coeff=0.5)
    liquidgasnozzle = design_liquid_gas_nozzle(flows=flows,
                                               gas_speed=10.0,
                                               liquid_speed=1.0,
                                               resistance_coeff=1.0)

    resistance = Resistance(
        properties=properties,
        flows=flows,
        losses_unaccounted=1.2,
        demister=demister,
        conditions=conditions,
        liquidgasnozzle=liquidgasnozzle,
        gasnozzle=gasnozzle,
    )

    geometry_cyclone = GeometryCyclone(width_inlet_cyclone=47.5e-3,
                                       height_inlet_cyclone=75e-3, number_of_cyclones=4)

    # ВЫВОД РЕЗУЛЬТАТОВ
    _major_header("РАСЧЁТ СОПРОТИВЛЕНИЯ СЕПАРАТОРА")

    _minor_divider()
    print(f"Плотность газа при рабочих условиях: "
          f"{properties.gas_density_work(conditions):.3f} кг/м³")

    _minor_divider()
    print(f"Диаметр штуцера входа ГЖС: "
          f"{liquidgasnozzle.nominal_diameter * _TO_MM:.0f} мм")
    print(f"Диаметр штуцера выхода газа: "
          f"{gasnozzle.nominal_diameter * _TO_MM:.0f} мм")

    _minor_divider()
    print(f"Скорость во входном штуцере ГЖС: "
          f"{liquidgasnozzle.flow_velocity(flows.flow_gas_work):.3f} м/с")
    print(f"Скорость в выходном штуцере газа: "
          f"{gasnozzle.flow_velocity(flows.flow_gas_work):.3f} м/с")

    _minor_divider()
    print(f"Коэффициент сопротивления входного патрубка: "
          f"{liquidgasnozzle.resistance_coeff}")
    print(f"Коэффициент сопротивления выходного патрубка: "
          f"{gasnozzle.resistance_coeff}")
    print(f"Коэффициент сопротивления сетчатого отбойника: "
          f"{demister.mesh_resistance_coefficient}")
    print(f"Коэффициент неучтенных потерь: "
          f"{resistance.losses_unaccounted}")

    _minor_divider()
    print(f"Падение давления на отбойнике: "
          f"{resistance.pressure_drop_mesh_demister():.3f} Па")
    print(f"Потери давления на штуцере входа: "
          f"{resistance.pressure_drop_inlet_nozzle():.3f} Па")
    print(f"Потери давления на штуцере выхода: "
          f"{resistance.pressure_drop_outlet_nozzle():.3f} Па")

    _minor_divider()
    print(f"Сопротивление сепаратора: "
          f"{resistance.separator_resistance():.3f} Па")

    cyclone = Cyclone(flows=flows, geometry_cyclone=geometry_cyclone)

    _major_header(
        "РАСЧЁТ СКОРОСТИ ГАЗА В СЕПАРАЦИОННОМ ЭЛЕМЕНТЕ (СПИРАЛЬНЫЙ КАНАЛ)")

    _minor_divider()
    print(
        f"Ширина входа в циклон: {geometry_cyclone.width_inlet_cyclone * _TO_MM:.1f} мм")
    print(
        f"Высота входа в циклон: {geometry_cyclone.height_inlet_cyclone * _TO_MM:.1f} мм")
    print(f"Количество циклонов: {geometry_cyclone.number_of_cyclones}")

    _minor_divider()
    print(f"Расход газа при рабочих условиях: "
          f"{flows.flow_gas_work * SECONDS_PER_DAY:.1f} м³/сут")
    print(f"Площадь сечения спирального канала: "
          f"{cyclone.area_spiral_channel():.4f} м²")
    print(f"Скорость газа в спиральном канале: "
          f"{cyclone.velocity_gas_in_spiral_channel():.3f} м/с")
