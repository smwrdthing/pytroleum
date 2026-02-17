"""
Модель расчёта гидроциклона на основе статей:
- Coelho & Medronho (2001): "A model for performance prediction of hydrocyclones"
- Pana-Suppamassadu et al. (2007): "Size Separation of Rubber Particles from
Natural Rubber Latex by Hydrocyclone Technique"

Все размерности в системе СИ:
- длина: м
- давление: Па
- расход: м³/с
- плотность: кг/м³
- вязкость: Па·с
- концентрации Cv: доля (0-1)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Literal, List


@dataclass
class GeometryParameters:
    """Геометрические параметры гидроциклона"""
    Dc: float  # Диаметр циклона, м
    Di: float  # Диаметр входа, м
    Do: float  # Диаметр вихреуловителя, м
    Du: float  # Диаметр нижнего слива, м
    L: float   # Длина гидроциклона, м
    l_vortex: float  # Выбег вихреуловителя, м


@dataclass
class PhysicalParameters:
    """Физические параметры среды"""
    mu: float   # Динамическая вязкость жидкости, Па·с
    rho: float  # Плотность жидкости, кг/м³
    rhos: float  # Плотность твёрдой фазы, кг/м³


@dataclass
class OperatingParameters:
    """Рабочие параметры гидроциклона"""
    Q: float    # Входной расход, м³/с
    Cv: float   # Концентрация твёрдых частиц на входе


# Константы модели для разных типов гидроциклонов
MODEL_PARAMS = {
    'rietema': (4.23, 2.45),
    'bradley': (5.10, 3.12),
    'demco': (5.40, 3.30)
}

HydrocycloneType = Literal['rietema', 'bradley', 'demco']


def calculate_hydrocyclone_parameters(
    geometry: GeometryParameters,
    physical: PhysicalParameters,
    operating: OperatingParameters,
    model_params: Tuple[float, float]
) -> Dict:
    """
    Расчёт параметров гидроциклона с заданным расходом Q.

    Основано на модели Coelho & Medronho (2001).

    Параметры:
    ----------
    geometry : GeometryParameters
        Геометрические параметры гидроциклона
    physical : PhysicalParameters
        Физические параметры среды
    operating : OperatingParameters
        Рабочие параметры
    model_params : Tuple[float, float]
        Параметры модели (alpha, m)

    Возвращает:
    -----------
    dict
        Словарь с рассчитанными параметрами
    """
    alpha, m = model_params

    # Длина гидроциклона L минус выбег вихреуловителя
    L_minus_l = geometry.L - geometry.l_vortex

    # Число Рейнольдса Re (уравнение (6) Coelho)
    Re = (4 * physical.rho * operating.Q) / (np.pi * physical.mu * geometry.Dc)

    # Рассчитываем перепад давления ΔP из уравнения (16)
    denominator_term = (0.184 * geometry.Dc**(-0.217) * geometry.Di**(1.231) *
                        (geometry.Do**2 + geometry.Du**2)**0.198 *
                        L_minus_l**0.462 *
                        physical.mu**(0.0566) *
                        physical.rho**(-0.528) *
                        np.exp(0.241 * operating.Cv))

    delta_p = (operating.Q / denominator_term)**(1/0.472)

    # Рассчитываем число Эйлера Eu (уравнение 13)
    Eu = (43.5 * geometry.Dc**0.57 *
          (geometry.Dc / geometry.Di)**2.61 *
          (geometry.Dc / (geometry.Do**2 + geometry.Du**2))**0.42 *
          (geometry.Dc / L_minus_l)**0.98 *
          Re**0.12 * np.exp(-0.51 * operating.Cv))

    # Рассчитываем водное отношение Rw (уравнение 14)
    Rw = 1.18 * (geometry.Dc / geometry.Do) ** 5.97 * \
        (geometry.Du / geometry.Dc) ** 3.10 * Eu ** (-0.54)

    # Рассчитываем Stk50_Eu по формуле (12)
    log_term = np.log(1 / Rw)

    Stk50_Eu = (0.12 * (geometry.Dc / geometry.Do)**0.95 *
                (geometry.Dc / L_minus_l)**1.33 *
                (log_term)**0.79 * np.exp(12.0 * operating.Cv))

    # Рассчитываем приведённый отсечной размер частицы (15)
    rhos_minus_rho = physical.rhos - physical.rho

    d50_prime = (1.173 * geometry.Dc**0.64 / (geometry.Do**0.475 * L_minus_l**0.665) *
                 np.sqrt((physical.mu * physical.rho * operating.Q) /
                 (rhos_minus_rho * delta_p)) *
                 np.log(1 / Rw)**0.395 *
                 np.exp(6.0 * operating.Cv))

    return {
        'Q': operating.Q,
        'delta_p': delta_p,
        'Rw': Rw,
        'Re': Re,
        'Eu': Eu,
        'd50_prime': d50_prime,
        'Stk50_Eu': Stk50_Eu,
        'alpha': alpha,
        'm': m
    }


def calculate_grade_efficiency(
    d: np.ndarray,
    d50_prime: float,
    model: Literal['plitt', 'lynch_rao'],
    m: float,
    alpha: float
) -> np.ndarray:
    """
    Расчёт приведённой вероятность уноса.

    Параметры:
    ----------
    d : np.ndarray
        Размеры частиц, м
    d50_prime : float
        Приведённый отсечной размер частицы, м
    model : {'plitt', 'lynch_rao'}
        Модель приведённой вероятность уноса.
    m : float
        Параметр формы (для модели Plitt)
    alpha : float
        Параметр формы (для модели Lynch-Rao)

    Возвращает:
    -----------
    np.ndarray
        Приведённую вероятность уноса G'(d)
    """
    ratio = np.asarray(d) / d50_prime

    if model == 'plitt':
        # Уравнение (9) Coelho - модифицированное Rosin-Rammler
        return 1 - np.exp(-0.693 * ratio**m)
    elif model == 'lynch_rao':
        # Уравнение (8) Coelho
        exp_term = np.exp(alpha * ratio)
        return (exp_term - 1) / (exp_term + np.exp(alpha) - 2)
    else:
        raise ValueError(f"Неизвестная модель: {model}")


def feed_cdf_rosin_rammler(d: np.ndarray, k: float, n: float) -> np.ndarray:
    """
    Кумулятивное распределение частиц y по Розин-Раммлеру.

    Уравнение (20) из статьи Pana-Suppamassadu et al. (2007):
    y(d) = 1 - exp(-(d/k)^n)

    Параметры:
    ----------
    d : np.ndarray
        Размеры частиц, м
    k : float
        Характерный размер, м
    n : float
        Параметр формы распределения

    Возвращает:
    -----------
    np.ndarray
        Кумулятивное распределение y(d)
    """
    d = np.asarray(d, dtype=float)
    return 1.0 - np.exp(-(d / k)**n)


def feed_cdf_rosin_rammler_derivative(d: np.ndarray, k: float, n: float) -> np.ndarray:
    """
    Производная кумулятивного распределения частиц по Розин-Раммлеру.

    Параметры:
    ----------
    d : np.ndarray
        Размеры частиц, м
    k : float
        Характерный размер, м
    n : float
        Параметр формы распределения

    Возвращает:
    -----------
    np.ndarray
        Производная dy/dd
    """
    d = np.asarray(d, dtype=float)
    ratio = d / k
    dy_dd = (n / k) * (ratio)**(n - 1) * np.exp(-(ratio)**n)
    return dy_dd


def calculate_reduced_total_efficiency(
    d: np.ndarray,
    k: float,
    n: float,
    G_prime: np.ndarray
) -> float:
    """
    Расчёт приведённой полной эффективности E_T'.

    E_T' = ∫₀^∞ G'(d) * (dy/dd) dd

    Параметры:
    ----------
    d : np.ndarray
        Размеры частиц, м
    k : float
        Характерный размер распределения Розин-Раммлер, м
    n : float
        Параметр формы распределения Розин-Раммлер
    G_prime : np.ndarray
        Приведённая вероятность уноса G'(d) для каждого размера

    Возвращает:
    -----------
    float
        Приведённая полная эффективность E_T'
    """
    d = np.asarray(d, dtype=float)
    Gp = np.asarray(G_prime, dtype=float)

    dy_dd = feed_cdf_rosin_rammler_derivative(d, k, n)

    return float(np.trapezoid(Gp * dy_dd, d))


def calculate_total_efficiency_from_reduced(
    E_T_prime: float,
    Rw: float
) -> float:
    """
    Расчёт полной эффективности E_T по приведённой E_T'.

    E_T = E_T' * (1 - Rw) + Rw

    Параметры:
    ----------
    E_T_prime : float
        Приведённая полная эффективность
    Rw : float
        Водное отношение

    Возвращает:
    -----------
    float
        Полная эффективность E_T
    """
    return E_T_prime * (1 - Rw) + Rw


def plot_hydrocyclone_analysis():
    """Построение графиков анализа гидроциклонов"""

    # Конфигурации гидроциклонов
    @dataclass
    class HydrocycloneConfig:
        name: str
        geometry: GeometryParameters
        htype: HydrocycloneType

    hydrocyclone_configs: List[HydrocycloneConfig] = [
        HydrocycloneConfig(
            name='Rietema',
            geometry=GeometryParameters(
                Dc=0.075,      # 75 мм
                Di=0.021,      # 21 мм
                Do=0.025,      # 25 мм
                Du=0.0125,     # 12.5 мм
                L=0.375,       # 375 мм
                l_vortex=0.075  # 75 мм
            ),
            htype='rietema'
        ),
        HydrocycloneConfig(
            name='Bradley',
            geometry=GeometryParameters(
                Dc=0.075,      # 75 мм
                Di=0.015,      # 15 мм
                Do=0.020,      # 20 мм
                Du=0.010,      # 10 мм
                L=0.180,       # 180 мм
                l_vortex=0.060  # 60 мм
            ),
            htype='bradley'
        ),
        HydrocycloneConfig(
            name='Demco',
            geometry=GeometryParameters(
                Dc=0.075,      # 75 мм
                Di=0.024,      # 24 мм
                Do=0.028,      # 28 мм
                Du=0.014,      # 14 мм
                L=0.240,       # 240 мм
                l_vortex=0.080  # 80 мм
            ),
            htype='demco'
        )
    ]

    # Физические параметры
    physical = PhysicalParameters(
        mu=0.001,    # Па·с, вязкость воды
        rho=1000,    # кг/м³, плотность воды
        rhos=2650    # кг/м³, плотность песка
    )

    # Рабочие параметры
    operating = OperatingParameters(
        Q=0.001,     # м³/с, расход (60 л/мин)
        Cv=0.05      # 5% объёмная концентрация
    )

    # Параметры распределения Розин-Раммлер
    k = 50e-6        # 50 мкм, характерный размер
    n = 1.5          # параметр формы

    # Диапазон размеров частиц для всех графиков
    d = np.linspace(0, 1e-3, 500)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # Для каждого типа циклона
    for row, config in enumerate(hydrocyclone_configs):
        # Расчёт параметров гидроциклона
        results = calculate_hydrocyclone_parameters(
            geometry=config.geometry,
            physical=physical,
            operating=operating,
            model_params=MODEL_PARAMS[config.htype]
        )

        d50_prime = results['d50_prime']
        Rw = results['Rw']
        m = results['m']
        alpha = results['alpha']

        # Нормированный размер
        d_norm = d / d50_prime

        # 1. График y(d) - кумулятивное распределение от d/d50'
        ax = axes[row, 0]
        y = feed_cdf_rosin_rammler(d, k, n)
        ax.plot(d_norm, y, 'b-', linewidth=2, label='y(d)')
        ax.set_xlabel("$d/d_{50}'$", fontsize=10)
        ax.set_ylabel('y(d)', fontsize=10)
        ax.set_xlim([0, 6])
        ax.set_ylim([0, 1.2])
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        # Текст в правом нижнем углу
        ax.text(0.95, 0.05, f'{config.name}\nk={k*1e6:.0f} мкм\nn={n:.1f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # 2. График dy/dd - функция плотности распределения от d/d50'
        ax = axes[row, 1]
        dy_dd = feed_cdf_rosin_rammler_derivative(d, k, n)
        # Масштабируем производную для отображения
        dy_dd_scaled = dy_dd * d50_prime  # чтобы размерность была 1/(d/d50')
        ax.plot(d_norm, dy_dd_scaled, 'b-', linewidth=2, label='dy/d(d/d50\')')
        ax.set_xlabel("$d/d_{50}'$", fontsize=10)
        ax.set_ylabel("$dy/d(d/d_{50}')$", fontsize=10)
        ax.set_xlim([0, 6])
        ax.set_ylim([0, 0.55])
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        # Текст в правом верхнем углу
        ax.text(0.95, 0.95, f'{config.name}\nk={k*1e6:.0f} мкм\nn={n:.1f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # 3. График G'(d) и G(d) - приведенная и обычная вероятность уноса
        ax = axes[row, 2]
        G_prime = calculate_grade_efficiency(d, d50_prime, 'plitt', m, alpha)
        G = G_prime * (1 - Rw) + Rw

        ax.plot(d_norm, G_prime, 'r-', linewidth=2, label="$G'(d/d_{50}')$")
        ax.plot(d_norm, G, 'b--', linewidth=2, label="$G(d/d_{50}')$")
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax.set_xlabel("$d/d_{50}'$", fontsize=10)
        ax.set_ylabel("$G(d)$, $G'(d)$", fontsize=10)
        ax.set_xlim([0, 6])
        ax.set_ylim([0, 1.2])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        # Текст в правом нижнем углу
        ax.text(0.95, 0.05,
                f'{config.name}\n$d_{{50}}\'$={d50_prime*1e6:.1f} мкм\n'
                f'm={m:.2f}\n$R_w$={Rw:.3f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Расчёт приведённой полной эффективности
        E_T_prime = calculate_reduced_total_efficiency(d, k, n, G_prime)

        # Расчёт полной эффективности
        E_T = calculate_total_efficiency_from_reduced(E_T_prime, Rw)
        print(f"=== {config.name} ===")
        print(f"  E_T' = {E_T_prime:.4f} ({E_T_prime*100:.1f}%)")
        print(f"  E_T  = {E_T:.4f} ({E_T*100:.1f}%)")
        print()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_hydrocyclone_analysis()
