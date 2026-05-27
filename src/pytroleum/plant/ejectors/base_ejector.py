from pytroleum.plant.ejectors.equations import (calculate_adiabatic_index,
                                                calculate_critical_pressure_ratio,
                                                calculate_gas_outflow_velocity)
from pytroleum.plant.ejectors.inputs import (ActiveMediumData,
                                             PassiveMediumData,
                                             Requirements)


class BaseEjector:

    def __init__(self, active: ActiveMediumData,
                 passive: PassiveMediumData,
                 req: Requirements):
        self.active = active
        self.passive = passive
        self.req = req

        # Степень сжатия установки
        self.compression_ratio = (req.outlet_pressure /
                                  passive.inlet_pressure)

        # Коэффициент эжекции
        self.entrainment_ratio = passive.mass_flow / active.mass_flow

        # Скорость активной среды в трубопроводе w(a)
        self.velocity_active_inlet = calculate_gas_outflow_velocity(
            self.active.mass_flow, self.active.temperature,
            self.active.inlet_pressure, self.active.inlet_diameter,
            self.active.molecular_mass)

        # Скорость пассивной среды в трубопроводе w(n)
        self.velocity_passive_inlet = calculate_gas_outflow_velocity(
            self.passive.mass_flow, self.passive.temperature,
            self.passive.inlet_pressure, self.passive.inlet_diameter,
            self.passive.molecular_mass)

        # NOTE расчёт скорости здесь должен быть верный, так как он использует определение
        # NOTE массового расхода через среднюю скорость в сечении и уравнение состояния
        # NOTE идеального газа
        # NOTE
        # NOTE С CoolProp будет точнее, т.к. для высоких давлений уравнение состояния
        # NOTE идеального газа работает хуже
        # NOTE
        # NOTE Происхождение формулы из таблицы в excel остаётся неизвестным

        # Показатель адиабаты
        self.adiabatic_index = calculate_adiabatic_index(
            active, passive, self.entrainment_ratio)

        # Критическое отношение давлений
        self.critical_pressure_ratio = calculate_critical_pressure_ratio(
            self.adiabatic_index)
