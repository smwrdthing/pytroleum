import numpy as np

from distributions import feed_cdf_rosin_rammler_derivative


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
