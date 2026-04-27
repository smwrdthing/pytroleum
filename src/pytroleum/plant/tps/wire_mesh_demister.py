import numpy as np
from dataclasses import dataclass
from scipy import interpolate
from scipy.constants import g
import matplotlib.pyplot as plt
from pytroleum.plant.tps.utils import _major_header, _minor_divider
from pytroleum.plant.tps.nozzle import select_nominal_diameter
from pytroleum.plant.tps.inputs import (OperationConditions,
                                        flow_based_water_cut,
                                        STANDARD_STATE,
                                        VAPOR, OIL, WATER)
from pytroleum.plant.tps.utils import (SECONDS_PER_DAY,
                                       SECONDS_PER_HOUR,
                                       PA_TO_MPA,
                                       PERCENT,
                                       _TO_MM,
                                       _TO_M,
                                       KELVIN_TO_CELSIUS)

# Данные взяты из рис.5 График зависимости коэффициента устойчивости
# режимов течения газожидкостной смеси от давления для вертикальной
# сетчатой насадки сепаратора (РД 0352-92-85 Методика
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


def calculate_critical_velocity(conditions: OperationConditions,
                                oil_surface_tension: float) -> float:
    """Критическая скорость газа для сетчатого каплеуловителя, м/с.

    v_кр = k_уст * √(√(g * σ_н * (ρ_ж - ρ_г_ру) / ρ_г_ру²))

    где k_уст — коэффициент устойчивости (из графика РД 0352-92-85),
    σ_н — поверхностное натяжение нефти,
    ρ_ж, ρ_г_ру — плотности жидкости и газа при р.у.
    """
    k = get_flow_stability_coefficient(conditions.pressure[VAPOR])
    gas_density = conditions.phase[VAPOR].rhomass()
    water_cut = flow_based_water_cut(conditions)
    density_liquid = (conditions.phase[OIL].rhomass() * (1 - water_cut) +
                      conditions.phase[WATER].rhomass() * water_cut)
    return k * np.sqrt(np.sqrt((g * oil_surface_tension *
                                (density_liquid - gas_density)) /
                               gas_density ** 2))


# Коэффициент учитывающий снижение площади сечения элементами насадки
AREA_REDUCTION_COEFF = 1.05


@dataclass
class WireMeshDemister:
    """Результаты расчёта сетчатого каплеуловителя."""
    area: float
    diameter: float
    nominal_diameter: float
    actual_area: float
    actual_velocity: float
    capacity: float


def design_demister(conditions: OperationConditions,
                    oil_surface_tension: float,
                    area_reduction_coeff: float = AREA_REDUCTION_COEFF
                    ) -> WireMeshDemister:
    """Расчёт геометрических параметров и производительности каплеуловителя.

    F_расч = k_пл * Q_г_ру / v_кр — расчётная площадь живого сечения, м²
    D_расч = √(4 * F_расч / π) — расчётный диаметр, м

    F_факт = π * D_ном² / (4 * k_пл) — действительная площадь живого сечения, м²
    u_нб   = Q_г_ру / F_факт  — скорость набегания газа, м/с
    Q_доп  = v_кр * F_факт — производительность каплеуловителя, м³/с

    где k_пл — коэффициент снижения живого сечения элементами насадки,
    D_ном — ближайший больший номинальный диаметр, м,
    v_кр — критическая скорость газа, м/с.
    """
    critical_velocity = calculate_critical_velocity(
        conditions, oil_surface_tension)

    area = area_reduction_coeff * \
        conditions.vol_flow_rate[VAPOR] / critical_velocity
    diameter = np.sqrt(4 * area / np.pi)

    nominal_diameter = select_nominal_diameter(diameter, NOMINAL_DIAMETERS)
    actual_area = (np.pi * nominal_diameter ** 2 /
                   (4 * area_reduction_coeff))

    actual_velocity = conditions.vol_flow_rate[VAPOR] / actual_area
    capacity = critical_velocity * actual_area

    return WireMeshDemister(
        area=area,
        diameter=diameter,
        nominal_diameter=nominal_diameter,
        actual_area=actual_area,
        actual_velocity=actual_velocity,
        capacity=capacity,
    )
