from pytroleum.plant.ejectors.equations import (calculate_adiabatic_index,
                                                calculate_critical_pressure_ratio,
                                                calculate_gas_outflow_velocity)
from pytroleum.plant.ejectors.inputs import (OperationConditions,
                                             Requirements,
                                             ACTIVE, PASSIVE)


class BaseEjector:

    def __init__(self, conditions: OperationConditions,
                 req: Requirements):
        self.conditions = conditions
        self.req = req

        # Степень сжатия установки
        self.compression_ratio = (req.outlet_pressure /
                                  conditions.pressure[PASSIVE])

        # Коэффициент эжекции
        self.entrainment_ratio = (conditions.mass_flow_rate[PASSIVE] /
                                  conditions.mass_flow_rate[ACTIVE])

        # Скорость активной среды в трубопроводе w(a)
        self.velocity_active_inlet = calculate_gas_outflow_velocity(
            conditions.mass_flow_rate[ACTIVE], conditions.temperature[ACTIVE],
            conditions.pressure[ACTIVE], req.active_inlet_diameter,
            conditions.phase[ACTIVE].molar_mass())

        # Скорость пассивной среды в трубопроводе w(n)
        self.velocity_passive_inlet = calculate_gas_outflow_velocity(
            conditions.mass_flow_rate[PASSIVE], conditions.temperature[PASSIVE],
            conditions.pressure[PASSIVE], req.passive_inlet_diameter,
            conditions.phase[PASSIVE].molar_mass())

        # Показатель адиабаты
        self.adiabatic_index = calculate_adiabatic_index(
            conditions, self.entrainment_ratio)

        # Критическое отношение давлений
        self.critical_pressure_ratio = calculate_critical_pressure_ratio(
            self.adiabatic_index)
