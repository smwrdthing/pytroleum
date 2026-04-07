import numpy as np
from scipy import interpolate
from scipy.constants import g
import matplotlib.pyplot as plt
from pytroleum.plant.tps.utils import _major_header, _minor_divider
from pytroleum.plant.tps.inputs import PhysicalProperties, FlowRates, OperationConditions
from pytroleum.plant.tps.inputs import (SECONDS_PER_DAY,
                                        SECONDS_PER_HOUR,
                                        PA_TO_MPA,
                                        PERCENT)


class WireMeshDemister:
    """Расчёт сетчатого каплеуловителя

    Данные взяты из рис.1 График зависимости коэффициента устойчивости
    режимов течения газожидкостной смеси от давления (РД 0352-92-85 Методика
    технологического расчета газосепараторов сетчатых)
    """

    _pressure_pa = np.array([1e5, 1.25e5, 1.5e5, 2e5, 2.5e5, 3e5, 3.5e5, 4e5, 4.5e5, 5e5,
                             5.5e5, 6e5, 6.5e5, 7e5, 8e5, 9e5, 10e5, 15e5, 20e5, 25e5,
                             30e5, 35e5, 40e5, 45e5, 50e5, 55e5, 60e5, 65e5, 70e5, 75e5,
                             80e5, 85e5, 90e5, 95e5, 100e5, 110e5, 115e5, 120e5, 125e5,
                             130e5, 135e5, 140e5])  # Па

    _flow_stability_coefficient = np.array([0.98, 0.9, 0.85, 0.725, 0.675, 0.625, 0.59,
                                            0.56, 0.54, 0.525, 0.51, 0.505, 0.502, 0.4907,
                                            0.49, 0.495, 0.5, 0.509, 0.525, 0.54, 0.56,
                                            0.59, 0.614, 0.63, 0.64, 0.638, 0.627, 0.623,
                                            0.6, 0.575, 0.55, 0.525, 0.509, 0.5, 0.48,
                                            0.465, 0.45, 0.43, 0.41, 0.4, 0.38, 0.375])

    _stability_coefficient_interpolator = interpolate.interp1d(
        _pressure_pa,
        _flow_stability_coefficient,
        kind='cubic')

    def __init__(self, properties: PhysicalProperties, flow_rates: FlowRates):
        self.properties = properties
        self.flow_rates = flow_rates

    def get_flow_stability_coefficient(self) -> float:
        """Коэффициент устойчивости режимов течения при текущем давлении"""
        pressure = self.flow_rates.conditions.pressure_work
        return self._stability_coefficient_interpolator(pressure)

    def calculate_critical_velocity(self) -> float:
        """Критическая скорость, м/с"""
        return self.get_flow_stability_coefficient() * np.sqrt(
            np.sqrt((g * self.properties.oil_surface_tension *
                    (self.properties.liquid_density() -
                     self.properties.gas_density_work(self.flow_rates.conditions))) /
                    self.properties.gas_density_work(self.flow_rates.conditions)**2)
        )

    def plot_stability_coefficient(self):
        """Построение графика зависимости коэффициента устойчивости от давления"""

        plt.figure(figsize=(10, 6))
        plt.plot(self._pressure_pa / PA_TO_MPA, self._flow_stability_coefficient,
                 'o', markersize=4, label='Данные по графику')

        # Кривая интерполяции
        pressure_smooth = np.linspace(self._pressure_pa.min(),
                                      self._pressure_pa.max(), 200)
        flow_stability_coefficient_smooth = self._stability_coefficient_interpolator(
            pressure_smooth)

        plt.plot(pressure_smooth / PA_TO_MPA, flow_stability_coefficient_smooth, '--',
                 alpha=0.7, label='Интерполяция')
        plt.xlabel('Давление, МПа', fontsize=12)
        plt.ylabel('Коэффициент устойчивости', fontsize=12)
        plt.title('Коэффициент устойчивости режимов течения от давления')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Текущая точка
        current_pressure = self.flow_rates.conditions.pressure_work
        current_flow_stability_coefficient = self.get_flow_stability_coefficient()
        plt.plot(current_pressure / PA_TO_MPA, current_flow_stability_coefficient,
                 'ro', markersize=8,
                 label=(f'Рабочее давление: {current_pressure/PA_TO_MPA:.2f} МПа, '
                        f'k={current_flow_stability_coefficient:.3f}'))
        plt.legend()

        plt.tight_layout()
        plt.show()

# ============================================================
# РАСЧЁТ
# ============================================================


if __name__ == "__main__":

    con = OperationConditions(
        pressure_work=2.48e6,
        temperature_work=353,                  # К
        flow_gas_norm=300000 / SECONDS_PER_DAY,  # м³/с
        flow_liquid=500 / SECONDS_PER_DAY,      # м³/с
    )

    props = PhysicalProperties(
        gas_density_norm=0.94,      # кг/м³
        oil_density=933,            # кг/м³
        water_density=966,          # кг/м³
        water_cut=0.6,              # 60% обводнённость
        gas_factor=267.9,           # м³/т
        oil_surface_tension=0.02848  # Н/м
    )

    flow_rates = FlowRates(conditions=con, properties=props)
    demister = WireMeshDemister(props, flow_rates)

    # Вывод результатов
    _major_header("РЕЗУЛЬТАТЫ РАСЧЁТА СЕТЧАТОГО КАПЛЕУЛОВИТЕЛЯ")
    print(f"Рабочее давление: {con.pressure_work/PA_TO_MPA} МПа")
    print(f"Рабочая температура: {con.temperature_work} K")

    _minor_divider()
    print(f"Объемный расход газа при н.у.: "
          f"{con.flow_gas_norm * SECONDS_PER_DAY} м³/сут")
    print(f"Объемный расход газа при р.у.: "
          f"{flow_rates.flow_gas_work() * SECONDS_PER_HOUR:.4f} м³/ч")
    print(
        f"Объемный расход жидкости: {con.flow_liquid * SECONDS_PER_DAY} м³/сут")
    print(f"Обводнённость: {props.water_cut * PERCENT} %")

    _minor_divider()
    print(f"Плотность газа в р.у.: {props.gas_density_work(con):.3f} кг/м³")
    print(f"Плотность жидкости (Н+В) при заданной обводненности: "
          f"{props.liquid_density():.2f} кг/м³")
    print(f"Коэффициент k: {demister.get_flow_stability_coefficient()}")
    print(
        f"Критическая скорость: {demister.calculate_critical_velocity():.3f} м/с")

    demister.plot_stability_coefficient()
