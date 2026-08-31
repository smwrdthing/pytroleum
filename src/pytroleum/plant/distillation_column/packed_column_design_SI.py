"""
=====================================================================
РАСЧЁТ НАСАДОЧНОЙ РЕКТИФИКАЦИОННОЙ КОЛОННЫ
=====================================================================
Реализация методики И. Александров, «Ректификационные и
абсорбционные аппараты» (1971), гл. V «Особенности расчёта насадочных
колонн» + гл. VIII, п.7 «Элементы технологического и гидравлического
расчёта насадочной ректификационной колонны» (стр. 251-255).

Семь расчётных блоков:
  1. Материальный баланс и рабочее флегмовое число
  2. Внутренние потоки пара и жидкости по колонне
  3. Диаметр колонны (скорость захлёбывания)                (V.1 - V.4)
  4. Число единиц переноса (ЧЕП)                             (V.9, V.12)
  5. Высота слоя насадки: метод ВЕП и метод ВЭТТ             (V.10, V.23)
  6. Гидравлическое сопротивление сухой насадки              (V.31-V.33)
  7. Гидравлическое сопротивление орошаемой насадки          (V.36-V.40)
"""

import math
import numpy as np
from scipy import interpolate

_TO_MM = 1000

# =====================================================================
# ДАННЫЕ ИЗ ГРАФИКА V-4 (стр. 158) для отпределения коэффициента k
# при расчете вакуумных насадочных колонн
# =====================================================================


_X_DATA = np.array([
    0.0105, 0.0119, 0.0136, 0.0154, 0.0175, 0.0199, 0.0226, 0.0257,
    0.0293, 0.0333, 0.0378, 0.0430, 0.0488, 0.0555, 0.0631, 0.0717,
    0.0815, 0.0927, 0.1053, 0.1197, 0.1361, 0.1547, 0.1758, 0.1998,
    0.2271, 0.2582, 0.2935, 0.3336, 0.3791, 0.4309, 0.4898, 0.5568,
    0.6322, 0.7193, 0.8176, 0.9339, 1.0059
])

_K_DATA = np.array([
    0.4542, 0.4533, 0.4512, 0.4431, 0.4375, 0.4322, 0.4245, 0.4213,
    0.4153, 0.4081, 0.3996, 0.3914, 0.3867, 0.3808, 0.3756, 0.3663,
    0.3550, 0.3468, 0.3403, 0.3317, 0.3219, 0.3124, 0.3028, 0.2921,
    0.2835, 0.2734, 0.2627, 0.2519, 0.2412, 0.2308, 0.2206, 0.2073,
    0.1981, 0.1912, 0.1803, 0.1687, 0.1627
])

# =====================================================================
# ИНТЕРПОЛЯТОР
# =====================================================================
_K_INTERPOLATOR = interpolate.interp1d(
    _X_DATA,
    _K_DATA,
    kind='cubic')

# =====================================================================
# НОМИНАЛЬНЫЙ РЯД ДИАМЕТРОВ КОЛОННЫ
# =====================================================================
STANDARD_DIAMETERS = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8,
                               3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 5.5, 6.0,
                               6.4, 7.0, 8.0, 9.0])

# =====================================================================
# 1. МАТЕРИАЛЬНЫЙ БАЛАНС И РАБОЧЕЕ ФЛЕГМОВОЕ ЧИСЛО
# =====================================================================


def material_balance(total_flow_rate, xF, yD, xR):
    """Материальный баланс колонны"""

    distillate_flow_rate = total_flow_rate * (xF - xR) / (yD - xR)
    residual_flow_rate = total_flow_rate - distillate_flow_rate
    return distillate_flow_rate, residual_flow_rate


def working_reflux(R_min, beta=1.15):
    """Рабочее флегмовое число"""
    return beta * R_min


# =====================================================================
# 2. ВНУТРЕННИЕ МАТЕРИАЛЬНЫЕ ПОТОКИ (верх / низ колонны)
# =====================================================================
def internal_flows_top(reflux_ratio, distillate_flow_rate):
    """L, G для укрепляющей (верхней) части колонны, кг/с."""
    liquid_flow_rate = reflux_ratio * distillate_flow_rate
    vapor_flow_rate = liquid_flow_rate + distillate_flow_rate
    return liquid_flow_rate, vapor_flow_rate


def internal_flows_bottom(L_top, F_liquid_part, residual_flow_rate):
    """
    L, G для исчерпывающей (нижней) части колонны, кг/с.
    F_liquid_part - жидкая часть питания, поступающая в низ колонны, кг/с.
    """
    liquid_flow_rate = L_top + F_liquid_part
    vapor_flow_rate = liquid_flow_rate - residual_flow_rate
    return liquid_flow_rate, vapor_flow_rate


def vapor_volume_flow(vapor_flow_rate, R, temperature,
                      compressibility_factor,
                      molecular_weight, pressure):
    """Объёмный расход паров по уравнению Клапейрона-Менделеева, м^3/с."""
    return (vapor_flow_rate*R * temperature*compressibility_factor /
            (molecular_weight*pressure))


def vapor_density(vapor_flow_rate, vapor_volume):
    """Плотность паров"""

    return vapor_flow_rate/vapor_volume


def get_k(liquid_flow_rate: float, vapor_flow_rate: float,
          vapor_dencity: float, liquid_dencity: float) -> float:
    """Коэффициент k по известным L, G и плотностям. """

    X = (liquid_flow_rate / vapor_flow_rate) * \
        np.sqrt(vapor_dencity / liquid_dencity)
    return float(_K_INTERPOLATOR(X))


def vapor_velocity(liquid_flow_rate: float,
                   vapor_flow_rate: float,
                   vapor_density: float,
                   liquid_density: float,
                   liquid_viscosity: float,
                   packing_a: float,
                   packing_void: float,
                   water_density: float) -> float:
    """Расчёт рабочей скорости пара."""

    # Коэффициент k по графику V-4
    k = get_k(liquid_flow_rate, vapor_flow_rate, vapor_density, liquid_density)

    density_ratio = vapor_density / liquid_density
    packing_ratio = packing_a / packing_void**3
    psi = liquid_density/water_density

    # Скорость по формуле (V.5)
    return 3.14 * k * (packing_ratio * density_ratio *
                       liquid_viscosity**0.12*psi)**(-0.5)


def column_diameter(V, w):
    """Расчетный диаметр колонны, м"""
    return math.sqrt(V / (0.785 * w))


def select_nominal_diameter(diameter: float, nominal_diameters) -> float:
    """Выбор ближайшего большего номинального диаметра, м."""
    for d_nom in sorted(nominal_diameters):
        if d_nom >= diameter:
            return d_nom
    raise ValueError(
        f"Расчётный диаметр {diameter * _TO_MM:.1f} мм "
        f"больше {max(nominal_diameters) * _TO_MM:.0f} мм")
