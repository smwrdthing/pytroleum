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
from typing import Dict, Tuple, Literal

from geometry import GeometryParameters
from physics import PhysicalParameters, OperatingParameters


# Константы модели для разных типов гидроциклонов
MODEL_PARAMS = {
    'rietema': (4.23, 2.45),
    'bradley': (5.10, 3.12),
    'demco': (5.40, 3.30)
}


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
