import numpy as np


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
