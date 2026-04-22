import numpy as np
from scipy import interpolate
from scipy.constants import g
import matplotlib.pyplot as plt
from pytroleum.plant.tps.utils import _major_header, _minor_divider
from pytroleum.plant.tps.nozzle import select_nominal_diameter
from pytroleum.plant.tps.inputs import (PhysicalProperties,
                                        FlowRates,
                                        OperationConditions)
from pytroleum.plant.tps.utils import (SECONDS_PER_DAY,
                                       SECONDS_PER_HOUR,
                                       PA_TO_MPA,
                                       PERCENT,
                                       _TO_MM,
                                       _TO_M,
                                       KELVIN_TO_CELSIUS)

# Данные взяты из рис.5 График зависимости коэффициента устойчивости
# режимов течения газожидкостной смеси от давления для вертикальной
# сетчатой начадки сепаратора (РД 0352-92-85 Методика
# технологического расчета газосепараторов сетчатых)


# Давление в ата (исходные данные из графика)
_PRESSURE_ATA = np.array([
    20, 22.123, 24.554, 23.821, 21.601, 23.149, 25.751, 26.923, 27.749, 30.017,
    28.774, 31.783, 30.930, 29.404, 33.219, 35.067, 34.126, 36.649, 38.196, 40.517,
    42.901, 45.148, 47.522, 49.907, 52.262, 54.651, 57.003, 59.295, 61.565, 62.849,
    64.582, 66.201, 68.097, 70.117, 72.336, 73.954, 75.376, 77.171, 78.695, 80.317,
    82.063, 83.600, 85.442, 84.499, 87.072, 86.382, 88.942, 90.135, 91.367, 94.183,
    92.799, 96.484, 95.472, 98.108, 99.438, 101.291, 103.128, 106.345, 105.207,
    107.251, 108.909, 110.362, 111.899, 113.485, 114.965, 117.884, 116.696, 119.284,
    121.176, 124.032, 122.459, 125.527, 126.909, 128.418, 129.901, 132.074,
    133.385, 135.311, 137.602, 140
])

# Коэффициент устойчивости (данные из графика)
_FLOW_STABILITY_COEFFICIENT = np.array([
    0.802, 0.815, 0.837, 0.829, 0.807, 0.822, 0.847, 0.858, 0.870, 0.894,
    0.879, 0.912, 0.902, 0.886, 0.926, 0.943, 0.934, 0.955, 0.967, 0.976,
    0.986, 0.992, 0.997, 0.998, 0.996, 0.992, 0.988, 0.978, 0.968, 0.962,
    0.954, 0.944, 0.933, 0.919, 0.906, 0.896, 0.885, 0.875, 0.864, 0.851,
    0.841, 0.831, 0.813, 0.822, 0.799, 0.806, 0.789, 0.778, 0.772, 0.754,
    0.761, 0.740, 0.746, 0.728, 0.719, 0.707, 0.694, 0.676, 0.684, 0.669,
    0.661, 0.654, 0.645, 0.633, 0.624, 0.608, 0.616, 0.602, 0.592, 0.577,
    0.584, 0.570, 0.560, 0.554, 0.544, 0.536, 0.527, 0.519, 0.510, 0.501
])

# Перевод в Паскали (1 ата = 1 кгс/см² = 98066.5 Па)
_PRESSURE_PA = _PRESSURE_ATA * 98066.5

NOMINAL_DIAMETERS = np.array([400, 500, 600, 650, 700, 800, 900, 1000,
                              1200, 1400, 1600, 1800, 2000, 2200, 2400,
                              2500, 2600, 2800, 3000, 3200, 3400, 3600,
                              3800, 4000, 4500, 5000, 5500, 5600, 6000,
                              6300, 6400, 7000, 7500, 8000, 8500, 9000,
                              9500, 10000, 11000, 12000, 14000, 16000,
                              18000, 20000]) * _TO_M

_STABILITY_COEFFICIENT_INTERPOLATOR = interpolate.interp1d(
    _PRESSURE_PA,
    _FLOW_STABILITY_COEFFICIENT,
    kind='cubic')


def get_flow_stability_coefficient(pressure: float) -> float:
    """Коэффициент устойчивости режимов течения при заданном давлении."""
    return _STABILITY_COEFFICIENT_INTERPOLATOR(pressure)


def calculate_critical_velocity(properties: PhysicalProperties,
                                conditions: OperationConditions) -> float:
    """Критическая скорость газа для сетчатого каплеуловителя, м/с.

    v_кр = k_уст * √(√(g * σ_н * (ρ_ж - ρ_г_ру) / ρ_г_ру²))

    где k_уст — коэффициент устойчивости (из графика РД 0352-92-85),
    σ_н — поверхностное натяжение нефти,
    ρ_ж, ρ_г_ру — плотности жидкости и газа при р.у.
    """
    k = get_flow_stability_coefficient(conditions.pressure)
    gas_density = properties.gas_density_work(conditions)
    return k * np.sqrt(np.sqrt((g * properties.oil_surface_tension *
                                (properties.liquid_density() - gas_density)) /
                               gas_density ** 2))


class WireMeshDemister:
    """Расчёт сетчатого каплеуловителя"""

    def __init__(self, properties: PhysicalProperties,
                 flows: FlowRates,
                 area_reduction_coefficient: float,
                 mesh_resistance_coefficient: float):
        self.properties = properties
        self.flows = flows
        # Коэффициент учитывающий снижение площади сечения элементами насадки
        self.area_reduction_coefficient = area_reduction_coefficient
        # Коэффициент сопротивления сетчатого отбойника
        self.mesh_resistance_coefficient = mesh_resistance_coefficient

    def compute(self) -> tuple[float, float, float, float]:
        """Расчёт геометрических параметров и производительности каплеуловителя.

        F_расч = k_пл * Q_г_ру / v_кр      — расчётная площадь живого сечения
        D_расч = √(4 * F_расч / π)          — расчётный диаметр

        F_факт = π * D_ном² / (4 * k_пл)   — действительная площадь живого сечения
        u_нб   = Q_г_ру / F_факт            — скорость набегания газа
        Q_доп  = v_кр * F_факт              — производительность каплеуловителя

        где k_пл  — коэффициент снижения живого сечения элементами насадки,
            D_ном — ближайший больший номинальный диаметр,
            v_кр  — критическая скорость газа.
        """
        critical_velocity = calculate_critical_velocity(
            self.properties, self.flows.conditions)

        area = (self.area_reduction_coefficient *
                self.flows.flow_gas_work / critical_velocity)
        diameter = np.sqrt(4 * area / np.pi)

        nominal_diameter = select_nominal_diameter(diameter, NOMINAL_DIAMETERS)
        actual_area = (np.pi * nominal_diameter ** 2 /
                       (4 * self.area_reduction_coefficient))

        actual_velocity = self.flows.flow_gas_work / actual_area
        capacity = critical_velocity * actual_area

        # NOTE можно объединить в объект и возвращать его
        # NOTE return design # <- всё тут
        return diameter, actual_area, actual_velocity, capacity


# ============================================================
# Пример использования
# ============================================================


if __name__ == "__main__":

    conditions = OperationConditions(
        pressure=4e6,
        temperature=353,                  # К
        flow_gas_norm=300000 / SECONDS_PER_DAY,  # м³/с
        flow_liquid=500 / SECONDS_PER_DAY,      # м³/с
    )

    properties = PhysicalProperties(
        gas_density_norm=0.94,      # кг/м³
        oil_density=933,            # кг/м³
        water_density=966,          # кг/м³
        water_cut=0.6,              # 60% обводнённость
        gas_factor=267.9,           # м³/т
        oil_surface_tension=0.02848,  # Н/м
        viscosity_oil=3.073e-3,
        viscosity_water=0.544e-3
    )

    flows = FlowRates(conditions=conditions, properties=properties)
    demister = WireMeshDemister(properties, flows,
                                area_reduction_coefficient=1.05,
                                mesh_resistance_coefficient=70)

    plt.figure(figsize=(10, 6))
    plt.plot(_PRESSURE_PA / PA_TO_MPA, _FLOW_STABILITY_COEFFICIENT,
             'o', markersize=4, label='Данные по графику')

    # Кривая интерполяции
    pressure_smooth = np.linspace(_PRESSURE_PA.min(),
                                  _PRESSURE_PA.max(), 200)
    flow_stability_coefficient_smooth = _STABILITY_COEFFICIENT_INTERPOLATOR(
        pressure_smooth)

    plt.plot(pressure_smooth / PA_TO_MPA, flow_stability_coefficient_smooth, '--',
             alpha=0.7, label='Интерполяция')
    plt.xlabel('Давление, МПа', fontsize=12)
    plt.ylabel('Коэффициент устойчивости', fontsize=12)
    plt.title('Коэффициент устойчивости режимов течения от давления')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.4, 1.1)
    plt.legend()

    # Текущая точка
    current_pressure = conditions.pressure
    current_flow_stability_coefficient = get_flow_stability_coefficient(
        conditions.pressure)
    plt.plot(current_pressure / PA_TO_MPA, current_flow_stability_coefficient,
             'ro', markersize=8,
             label=(f'Рабочее давление: {current_pressure/PA_TO_MPA:.2f} МПа, '
                    f'k={current_flow_stability_coefficient:.3f}'))
    plt.legend()

    # ax = plt.gca()
    # ax.set_xlim((2, 13))
    # ax.set_ylim((0.4, 1.1))

    plt.tight_layout()
    plt.show()

    # Вывод результатов
    _major_header("РЕЗУЛЬТАТЫ РАСЧЁТА СЕТЧАТОГО КАПЛЕУЛОВИТЕЛЯ")
    print(f"Рабочее давление: {conditions.pressure/PA_TO_MPA} МПа")
    print(f"Рабочая температура: {conditions.temperature} К "
          f"({conditions.temperature - KELVIN_TO_CELSIUS} °C)")

    _minor_divider()
    print(f"Объемный расход газа при н.у.: "
          f"{conditions.flow_gas_norm * SECONDS_PER_DAY:,.0f} м³/сут".replace(",", " "))
    print(f"Объемный расход газа при р.у.: "
          f"{flows.flow_gas_work * SECONDS_PER_HOUR:.4f} м³/ч")
    print(f"Объемный расход жидкости: "
          f"{conditions.flow_liquid * SECONDS_PER_DAY} м³/сут")
    print(f"Обводнённость: {properties.water_cut * PERCENT} %")

    _minor_divider()
    print(f"Плотность газа в р.у.: "
          f"{properties.gas_density_work(conditions):.3f} кг/м³")
    print(f"Плотность жидкости (Н+В) при заданной обводненности: "
          f"{properties.liquid_density():.2f} кг/м³")
    print(
        f"Коэффициент k: {get_flow_stability_coefficient(conditions.pressure):.2f}")
    print(f"Критическая скорость: "
          f"{calculate_critical_velocity(properties, conditions):.3f} м/с")

    _minor_divider()
    diameter, actual_area, actual_velocity, capacity = demister.compute()
    nominal_d = select_nominal_diameter(diameter, NOMINAL_DIAMETERS)
    print(f"Диаметр: {diameter * _TO_MM:.1f} мм")
    print(f"Принятый диаметр: {nominal_d * _TO_MM:.0f} мм")
    print(f"Действительная площадь живого сечения: {actual_area:.4f} м²")
    print(f"Действительная скорость набегания: {actual_velocity:.3f} м/с")
    print(f"Производительность: {capacity:.4f} м³/с")
