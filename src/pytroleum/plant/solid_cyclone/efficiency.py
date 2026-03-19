"""
Распределение частиц и эффективность гидроциклона.

"""
from typing import Literal

from numpy.typing import NDArray
import numpy as np


def cumulative_size_distribution(
    particle_diameters: NDArray,
    k: float,
    n: float,
) -> NDArray:
    """Кумулятивное распределение частиц по Розин-Раммлеру."""
    return 1.0 - np.exp(-(particle_diameters / k) ** n)


def probability_density(
    particle_diameters: NDArray,
    k: float,
    n: float,
) -> NDArray:
    """Производная кумулятивного распределения Розин-Раммлера (плотность)."""
    ratio = particle_diameters / k
    return (n / k) * ratio ** (n - 1) * np.exp(-ratio ** n)


def calculate_reduced_grade_efficiency(
    particle_diameters: NDArray,
    reduced_cut_size: float,
    model_reduced_grade_efficiency: Literal['plitt', 'lynch_rao'],
    m: float,
    alpha: float,
) -> NDArray:
    """Расчёт приведённой вероятности уноса G'(d)."""
    ratio = particle_diameters / reduced_cut_size

    match model_reduced_grade_efficiency:
        case 'plitt':
            return 1 - np.exp(-0.693 * ratio ** m)
        case 'lynch_rao':
            exp_term = np.exp(alpha * ratio)
            return (exp_term - 1) / (exp_term + np.exp(alpha) - 2)
        case _:
            raise ValueError(
                f"Неизвестная модель: {model_reduced_grade_efficiency}")


def calculate_reduced_total_efficiency(
    particle_diameters: NDArray,
    k: float,
    n: float,
    reduced_grade_efficiency: NDArray,
) -> NDArray | np.floating:
    """Расчёт приведённой полной эффективности E_T'."""
    dy_dd = probability_density(particle_diameters, k, n)
    return np.trapezoid(reduced_grade_efficiency * dy_dd, particle_diameters, axis=0)


def calculate_total_efficiency(
    reduced_total_efficiency: NDArray | np.floating,
    water_flow_ratio: NDArray | np.floating | float,
) -> NDArray | np.floating:
    """Расчёт полной эффективности E_T по приведённой E_T'."""
    return reduced_total_efficiency * (1 - water_flow_ratio) + water_flow_ratio
