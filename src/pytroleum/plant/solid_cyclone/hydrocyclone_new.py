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
- концентрации: доля (0-1)

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Literal

from numpy.typing import NDArray

import numpy as np
import matplotlib.pyplot as plt

# NOTE Говорящие имена параметров, переменных, полей классов и т.д.

# region Enums

# перечисление индексов в массиве диаметров


class HydrocycloneDiameters(IntEnum):
    C, CYCLONE = 0, 0  # индекс 0 — диаметр корпуса циклона Dc
    I, INLET = 1, 1  # индекс 1 — диаметр входного патрубка Di
    O, OVERFLOW = 2, 2  # индекс 2 — диаметр патрубка верхнего слива Do
    U, UNDERFLOW = 3, 3  # индекс 3 — диаметр патрубка нижнего слива Du

    SIZE = auto()  # NOTE к SZIE не помешает пояснительный комментацрий


class HydrocycloneLengths(IntEnum):  # перечисление индексов в массиве длин
    T, TOTAL = 0, 0  # индекс 0 — полная длина циклона L
    V, VORTEX_FINDER = 1, 1  # индекс 1 — длина вихревой трубки

    SIZE = auto()

# region Constants


# Допустимые диапазоны геометрических пропорций (Coelho & Medronho, 2001)
DI_DC_MIN = 0.14  # минимально допустимое отношение Di/Dc
DI_DC_MAX = 0.28  # максимально допустимое отношение Di/Dc

DO_DC_MIN = 0.20  # минимально допустимое отношение Do/Dc
DO_DC_MAX = 0.34  # максимально допустимое отношение Do/Dc

DU_DC_MIN = 0.04  # минимально допустимое отношение Du/Dc
DU_DC_MAX = 0.28  # максимально допустимое отношение Du/Dc

L_VORTEX_DC_MIN = 0.33  # минимально допустимое отношение l/Dc
L_VORTEX_DC_MAX = 0.55  # максимально допустимое отношение l/Dc

L_DC_MIN = 3.30  # минимально допустимое отношение L/Dc (полная длина)
L_DC_MAX = 6.93  # максимально допустимое отношение L/Dc

THETA_MIN = 9.0   # минимально допустимый угол конуса, градусы
THETA_MAX = 20.0  # максимально допустимый угол конуса, градусы

# region Dataclasses


@dataclass
class GeometryParameters:
    """Геометрические параметры гидроциклона."""

    diameters: NDArray = field(
        default_factory=lambda: np.zeros(HydrocycloneDiameters.SIZE))
    lengths: NDArray = field(
        default_factory=lambda: np.zeros(HydrocycloneLengths.SIZE))
    theta: float = 15.0

    @classmethod  # NOTE насколько нам нужен classmethod?
    def from_named(
        cls,
        Dc: float,       # диаметр корпуса, м
        Di: float,       # диаметр входного патрубка, м
        Do: float,       # диаметр патрубка верхнего слива, м
        Du: float,       # диаметр патрубка нижнего слива, м
        L: float,        # полная длина циклона, м
        l_vortex: float,  # длина вихревой трубки, м
        theta: float = 15.0,  # угол конуса, градусы; необязательный параметр с дефолтом
    ) -> 'GeometryParameters':
        # NOTE вместо использования '' можно в начале файла сделать импорт :
        # NOTE >>> from __future__ import annotations

        # NOTE имена параметров
        """Создание объекта из именованных размеров."""
        obj = cls(theta=theta)
        # записывает Dc в ячейку с индексом 0 массива diameters
        obj.diameters[HydrocycloneDiameters.C] = Dc
        # записывает Di в ячейку с индексом 1
        obj.diameters[HydrocycloneDiameters.I] = Di
        # записывает Do в ячейку с индексом 2
        obj.diameters[HydrocycloneDiameters.O] = Do
        # записывает Du в ячейку с индексом 3
        obj.diameters[HydrocycloneDiameters.U] = Du
        # записывает полную длину L в ячейку с индексом 0 массива lengths
        obj.lengths[HydrocycloneLengths.T] = L
        # записывает длину вихревой трубки в ячейку с индексом 1
        obj.lengths[HydrocycloneLengths.V] = l_vortex
        return obj  # возвращает полностью заполненный объект GeometryParameters

    def check_proportions(self) -> list[str]:
        """Проверка соответствия геометрических пропорций допустимому диапазону."""
        violations = []
        # извлекает Dc из массива для дальнейших делений
        Dc = self.diameters[HydrocycloneDiameters.C]
        # вычисляет безразмерное отношение Di/Dc
        Di_Dc = self.diameters[HydrocycloneDiameters.I] / Dc
        # вычисляет безразмерное отношение Do/Dc
        Do_Dc = self.diameters[HydrocycloneDiameters.O] / Dc
        # вычисляет безразмерное отношение Du/Dc
        Du_Dc = self.diameters[HydrocycloneDiameters.U] / Dc
        # вычисляет безразмерное отношение l/Dc (вихревая трубка)
        l_Dc = self.lengths[HydrocycloneLengths.V] / Dc
        # вычисляет безразмерное отношение L/Dc (полная длина)
        L_Dc = self.lengths[HydrocycloneLengths.T] / Dc

        # сравнение с допустимым диапазоном
        if not (DI_DC_MIN <= Di_Dc <= DI_DC_MAX):
            violations.append(
                f"Di/Dc={Di_Dc:.3f} вне диапазона [{DI_DC_MIN}-{DI_DC_MAX}]")

        if not (DO_DC_MIN <= Do_Dc <= DO_DC_MAX):
            violations.append(
                f"Do/Dc={Do_Dc:.3f} вне диапазона [{DO_DC_MIN}-{DO_DC_MAX}]")

        if not (DU_DC_MIN <= Du_Dc <= DU_DC_MAX):
            violations.append(
                f"Du/Dc={Du_Dc:.3f} вне диапазона [{DU_DC_MIN}-{DU_DC_MAX}]")

        if not (L_VORTEX_DC_MIN <= l_Dc <= L_VORTEX_DC_MAX):
            violations.append(
                f"l/Dc={l_Dc:.3f} вне диапазона [{L_VORTEX_DC_MIN}-{L_VORTEX_DC_MAX}]")

        if not (L_DC_MIN <= L_Dc <= L_DC_MAX):
            violations.append(
                f"L/Dc={L_Dc:.3f} вне диапазона [{L_DC_MIN}-{L_DC_MAX}]")

        if not (THETA_MIN <= self.theta <= THETA_MAX):
            violations.append(
                f"θ={self.theta:.1f}° вне диапазона [{THETA_MIN}°-{THETA_MAX}°]")

        return violations

    def get_geometry_ratios(self) -> dict[str, float]:
        """Возвращает словарь с геометрическими пропорциями."""
        Dc = self.diameters[HydrocycloneDiameters.C]
        return {
            # отношение диаметра входа к диаметру корпуса
            'Di/Dc': self.diameters[HydrocycloneDiameters.I] / Dc,
            # отношение диаметра верхнего слива к диаметру корпуса
            'Do/Dc': self.diameters[HydrocycloneDiameters.O] / Dc,
            # отношение диаметра нижнего слива к диаметру корпуса
            'Du/Dc': self.diameters[HydrocycloneDiameters.U] / Dc,
            # отношение длины вихревой трубки к диаметру корпуса
            'l/Dc': self.lengths[HydrocycloneLengths.V] / Dc,
            # отношение полной длины к диаметру корпуса
            'L/Dc': self.lengths[HydrocycloneLengths.T] / Dc,
        }


@dataclass
class PhysicalProperties:
    """Физические свойства среды."""
    # NOTE имена полей
    mu: float    # динамическая вязкость жидкости, Па·с
    rho: float   # плотность жидкости, кг/м³
    rhos: float  # плотность твёрдой фазы (частиц), кг/м³


def _euler_number(
    geometry: GeometryParameters,  # геометрия циклона
    solids_concentration: float,  # объёмная концентрация твёрдых частиц Cv
    Re: float,                     # число Рейнольдса
) -> float:
    """Расчёт числа Эйлера Eu."""
    Dc = geometry.diameters[HydrocycloneDiameters.C]
    Di = geometry.diameters[HydrocycloneDiameters.I]
    Do = geometry.diameters[HydrocycloneDiameters.O]
    Du = geometry.diameters[HydrocycloneDiameters.U]

    L_minus_l = (geometry.lengths[HydrocycloneLengths.T] -
                 geometry.lengths[HydrocycloneLengths.V])

    return (43.5 * Dc**0.57 * (Dc / Di)**2.61 *
            (Dc / (Do**2 + Du**2))**0.42 *
            (Dc / L_minus_l)**0.98 * Re**0.12 * np.exp(-0.51 * solids_concentration))


class BaseHydrocyclone(ABC):
    """Абстрактный базовый класс для моделей гидроциклонов твёрдое-жидкость."""

    def __init__(self, name: str, geometry: GeometryParameters) -> None:
        self.name = name          # строковое имя модели
        self.geometry = geometry  # объект GeometryParameters с размерами циклона

    @property           # объявляет model_params как свойство
    @abstractmethod     # обязывает каждый подкласс переопределить это свойство
    def model_params(self) -> tuple[float, float]:
        """Возвращает параметры модели (alpha, m)."""
        ...

    # NOTE для чего model_params сделан через property?

    def calculate_from_flow_rate(
        self,
        properties: PhysicalProperties,  # физические свойства жидкости и твёрдой фазы
        flow_rate: float,                 # заданный объёмный расход Q, м³/с
        solids_concentration: float,      # объёмная концентрация твёрдых частиц Cv
    ) -> dict[str, float]:
        """Расчёт параметров при заданном объёмном расходе (ΔP = (Q/K)^(1/0.472))"""
        K = self._compute_K(properties, solids_concentration)
        pressure_drop = (flow_rate / K) ** (1 / 0.472)
        # NOTE 1 и 0.472 у всех гидроциклонов?
        return self._compute_results(
            properties, flow_rate, pressure_drop, solids_concentration)

    def calculate_from_pressure_drop(
        self,
        properties: PhysicalProperties,  # физические свойства жидкости и твёрдой фазы
        pressure_drop: float,             # заданный перепад давления ΔP, Па
        solids_concentration: float,      # объёмная концентрация твёрдых частиц Cv
    ) -> dict[str, float]:
        """Расчёт параметров при заданном перепаде давления (Q = K·ΔP^0.472)"""
        K = self._compute_K(properties, solids_concentration)
        flow_rate = K * pressure_drop**0.472
        return self._compute_results(
            properties, flow_rate, pressure_drop, solids_concentration)

    def _compute_K(
        self,
        properties: PhysicalProperties,  # физические свойства
        solids_concentration: float,      # концентрация твёрдых частиц Cv
    ) -> float:
        """Коэффициент K в уравнении Q = K · ΔP^0.472."""
        Dc = self.geometry.diameters[HydrocycloneDiameters.C]
        Di = self.geometry.diameters[HydrocycloneDiameters.I]
        Do = self.geometry.diameters[HydrocycloneDiameters.O]
        Du = self.geometry.diameters[HydrocycloneDiameters.U]
        L_minus_l = (self.geometry.lengths[HydrocycloneLengths.T] -
                     self.geometry.lengths[HydrocycloneLengths.V])

        return (0.184 * Dc**(-0.217) * Di**(1.231) * (Do**2 + Du**2)**0.198 *
                L_minus_l**0.462 * properties.mu**(0.0566) *
                properties.rho**(-0.528) * np.exp(0.241 * solids_concentration))

    def _compute_results(
        self,
        properties: PhysicalProperties,  # физические свойства
        flow_rate: float,  # расход Q, м³/с
        pressure_drop: float,  # перепад давления ΔP, Па
        solids_concentration: float,  # концентрация Cv
    ) -> dict[str, float]:
        """Расчёт всех выходных параметров гидроциклона."""
        alpha, m = self.model_params

        Dc = self.geometry.diameters[HydrocycloneDiameters.C]
        Do = self.geometry.diameters[HydrocycloneDiameters.O]
        Du = self.geometry.diameters[HydrocycloneDiameters.U]
        L_minus_l = (self.geometry.lengths[HydrocycloneLengths.T] -
                     self.geometry.lengths[HydrocycloneLengths.V])

        Re = (4 * properties.rho * flow_rate) / (np.pi * properties.mu * Dc)
        Eu = _euler_number(self.geometry, solids_concentration, Re)

        # Water flow ratio: доля воды, уходящая в нижний слив
        Rw = (1.18 * (Dc / Do)**5.97 *
              (Du / Dc)**3.10 * Eu**(-0.54))

        rhos_minus_rho = properties.rhos - properties.rho

        reduced_cut_size = (1.173 * Dc**0.64 /
                            (Do**0.475 * L_minus_l**0.665) *
                            np.sqrt((properties.mu * properties.rho * flow_rate) /
                                    (rhos_minus_rho * pressure_drop)) *
                            np.log(1 / Rw)**0.395 *
                            np.exp(6.0 * solids_concentration))

        return {
            'Q': flow_rate,                           # расход, м³/с
            'delta_p': pressure_drop,                 # перепад давления, Па
            'Rw': Rw,                                 # water flow ratio
            'Re': Re,                                 # число Рейнольдса
            'Eu': Eu,                                 # число Эйлера
            'reduced_cutoff_diameter': reduced_cut_size,  # приведённый отсечной диаметр
            'alpha': alpha,
            'm': m,
        }


# конкретный подкласс для модели Rietema
class RietemaHydrocyclone(BaseHydrocyclone):
    """Гидроциклон по модели Rietema."""

    @property
    def model_params(self) -> tuple[float, float]:
        return (4.23, 2.45)


# конкретный подкласс для модели Bradley
class BradleyHydrocyclone(BaseHydrocyclone):
    """Гидроциклон по модели Bradley."""

    @property
    def model_params(self) -> tuple[float, float]:
        return (5.10, 3.12)


# конкретный подкласс для модели Demco
class DemcoHydrocyclone(BaseHydrocyclone):
    """Гидроциклон по модели Demco."""

    @property
    def model_params(self) -> tuple[float, float]:
        return (5.40, 3.30)


def calculate_reduced_grade_efficiency(
    particle_diameters: NDArray,  # массив диаметров частиц d, м (или скаляр)
    reduced_cut_size: float,      # приведённый отсечной диаметр d'₅₀, м
    model: Literal['plitt', 'lynch_rao'],
    m: float,
    alpha: float,
) -> NDArray:
    """Расчёт приведённой вероятности уноса G'(d)."""
    ratio = particle_diameters / reduced_cut_size

    match model:
        case 'plitt':
            return 1 - np.exp(-0.693 * ratio**m)
        case 'lynch_rao':
            exp_term = np.exp(alpha * ratio)
            return (exp_term - 1) / (exp_term + np.exp(alpha) - 2)
        case _:
            raise ValueError(f"Неизвестная модель: {model}")


def calculate_cumulative_particle_size_distribution(
    particle_diameters: NDArray,  # массив диаметров частиц d, м
    k: float,
    n: float,
) -> NDArray:
    """Кумулятивное распределение частиц по Розин-Раммлеру."""
    return 1.0 - np.exp(-(particle_diameters / k)**n)


def _rosin_rammler_derivative(
    particle_diameters: NDArray,  # массив диаметров частиц d, м
    k: float,
    n: float,
) -> NDArray:
    """Производная кумулятивного распределения Розин-Раммлера."""
    ratio = particle_diameters / k
    return (n / k) * ratio**(n - 1) * np.exp(-ratio**n)


def calculate_reduced_total_efficiency(
    particle_diameters: NDArray,
    k: float,
    n: float,
    reduced_grade_efficiency: NDArray,
) -> float:
    """Расчёт приведённой полной эффективности E_T'."""
    dy_dd = _rosin_rammler_derivative(particle_diameters, k, n)
    return float(np.trapezoid(reduced_grade_efficiency * dy_dd, particle_diameters))


def calculate_total_efficiency(
    reduced_total_efficiency: float,
    water_flow_ratio: float,
) -> float:
    """Расчёт полной эффективности E_T по приведённой E_T'."""
    return reduced_total_efficiency * (1 - water_flow_ratio) + water_flow_ratio


# region Printing

def print_proportions(geometry: GeometryParameters, name: str) -> None:
    """Вывод геометрических пропорций и проверка их соответствия диапазонам."""

    # NOTE в целом эту функциональность можно внести в базовый класс гидроциклона

    # получает словарь безразмерных пропорций
    ratios = geometry.get_geometry_ratios()
    violations = geometry.check_proportions()  # получает список нарушений

    print(f"\n=== {name} proportions ===")
    # выводит значение и допустимый диапазон
    print(f"Di/Dc = {ratios['Di/Dc']:.3f}  [{DI_DC_MIN}-{DI_DC_MAX}]")
    print(f"Do/Dc = {ratios['Do/Dc']:.3f}  [{DO_DC_MIN}-{DO_DC_MAX}]")
    print(f"Du/Dc = {ratios['Du/Dc']:.3f}  [{DU_DC_MIN}-{DU_DC_MAX}]")
    print(
        f"l/Dc  = {ratios['l/Dc']:.3f}  [{L_VORTEX_DC_MIN}-{L_VORTEX_DC_MAX}]")
    print(f"L/Dc  = {ratios['L/Dc']:.3f}  [{L_DC_MIN}-{L_DC_MAX}]")
    print(f"θ= {geometry.theta:.1f}°  [{THETA_MIN}°-{THETA_MAX}°]")

    if violations:
        print("Пропорции нарушены:")
        for v in violations:
            print(f"  {v}")
    else:
        print("Все пропорции в допустимых диапазонах")
    print()


# region Config builder
def build_standard_configs(Dc: float) -> list[BaseHydrocyclone]:
    """
    Возвращает список стандартных конфигураций гидроциклонов
    с пропорциями в допустимых диапазонах.
    """

    # NOTE возможно есть смысл разделить эту функцию на три (по одной на каждый тип)
    # NOTE для тех случаев, когда мы хотим считать какой-то конкретный гидроциклон
    # NOTE
    # NOTE потом эти отдельные функции можно использовать и в функции, которая собирает
    # NOTE все три типа сразу

    rietema_geometry = GeometryParameters.from_named(
        Dc=Dc, Di=Dc*0.20, Do=Dc*0.25, Du=Dc*0.15,
        L=Dc*4.50, l_vortex=Dc*0.40, theta=15.0,
    )
    bradley_geometry = GeometryParameters.from_named(
        Dc=Dc, Di=Dc*0.16, Do=Dc*0.22, Du=Dc*0.12,
        L=Dc*5.50, l_vortex=Dc*0.45, theta=12.0,
    )
    demco_geometry = GeometryParameters.from_named(
        Dc=Dc, Di=Dc*0.25, Do=Dc*0.30, Du=Dc*0.20,
        L=Dc*5.00, l_vortex=Dc*0.50, theta=18.0,
    )

    hydrocyclones: list[BaseHydrocyclone] = [
        # экземпляр модели Rietema с геометрией и именем
        RietemaHydrocyclone('Rietema', rietema_geometry),
        # экземпляр модели Bradley
        BradleyHydrocyclone('Bradley', bradley_geometry),
        # экземпляр модели Demco
        DemcoHydrocyclone('Demco', demco_geometry),
    ]

    print("\n" + "="*60)
    print(f"ПРОВЕРКА ГЕОМЕТРИЧЕСКИХ ПРОПОРЦИЙ (Dc = {Dc*1000:.1f} мм)")
    print("="*60)
    for h in hydrocyclones:
        print_proportions(h.geometry, h.name)

    # возвращает список для использования в расчётах и построении графиков
    return hydrocyclones


# region Plotting

def _plot_row(
    axes_row,
    hydrocyclone: BaseHydrocyclone,
    results: dict[str, float],
    particle_diameters: NDArray,
    k: float,
    n: float,
    mode: Literal['Q', 'delta_p'],
) -> None:
    """Рисует строку из трёх графиков для одного типа гидроциклона."""

    # NOTE очень частная функция, котрая делает слишком много всего сразу
    # NOTE если мы захотим рисовать другие граифки - придётся здесь всё переписывать
    # NOTE или дописывать другие функции

    reduced_cut_size = results['reduced_cutoff_diameter']
    Rw = results['Rw']
    m = results['m']
    alpha = results['alpha']
    Q_res = results['Q']
    delta_p_res = results['delta_p']

    # нормированный диаметр
    d_norm = particle_diameters / reduced_cut_size

    # 1. Кумулятивное распределение
    ax = axes_row[0]
    y = calculate_cumulative_particle_size_distribution(
        particle_diameters, k, n)
    ax.plot(d_norm, y, 'b-', linewidth=2)
    ax.set_xlabel("$d/d_{50}'$", fontsize=10)
    ax.set_ylabel('y(d)', fontsize=10)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(
        0.95, 0.05,
        f'{hydrocyclone.name}\nk={k*1e6:.0f} мкм\nn={n:.1f}',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )

    # 2. Плотность распределения
    ax = axes_row[1]
    dy_dd = _rosin_rammler_derivative(particle_diameters, k, n)
    dy_dd_scaled = dy_dd * reduced_cut_size
    ax.plot(d_norm, dy_dd_scaled, 'b-', linewidth=2)
    ax.set_xlabel("$d/d_{50}'$", fontsize=10)
    ax.set_ylabel("$dy/d(d/d_{50}')$", fontsize=10)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 0.3])
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5,
               linewidth=2)
    ax.text(
        0.95, 0.95,
        f'{hydrocyclone.name}\nk={k*1e6:.0f} мкм\nn={n:.1f}',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )

    # 3. Вероятность уноса
    ax = axes_row[2]
    reduced_grade_efficiency = calculate_reduced_grade_efficiency(
        particle_diameters, reduced_cut_size, 'plitt', m, alpha)
    G = reduced_grade_efficiency * (1 - Rw) + Rw
    ax.plot(d_norm, reduced_grade_efficiency, 'r-',
            linewidth=2, label="$G'(d/d_{50}')$")
    ax.plot(d_norm, G, 'b--', linewidth=2, label="$G(d/d_{50}')$")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel("$d/d_{50}'$", fontsize=10)
    ax.set_ylabel("$G(d)$, $G'(d)$", fontsize=10)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    if mode == 'Q':
        info_text = (
            f'{hydrocyclone.name}\n'
            f"$d_{{50}}'$={reduced_cut_size*1e6:.1f} мкм\n"
            f'm={m:.2f}\n'
            f'$R_w$={Rw:.3f}\n'
            f'Q={Q_res*1000*60:.1f} л/мин'
        )
    else:
        info_text = (
            f'{hydrocyclone.name}\n'
            f"$d_{{50}}'$={reduced_cut_size*1e6:.1f} мкм\n"
            f'm={m:.2f}\n'
            f'$R_w$={Rw:.3f}\n'
            f'ΔP={delta_p_res/1000:.2f} кПа'
        )

    ax.text(
        0.95, 0.05, info_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )

    reduced_total_efficiency = calculate_reduced_total_efficiency(
        particle_diameters, k, n, reduced_grade_efficiency)
    total_efficiency = calculate_total_efficiency(reduced_total_efficiency, Rw)

    print(f"=== {hydrocyclone.name} ===")
    print(f"Q = {Q_res*1000*60:.2f} л/мин, ΔP = {delta_p_res/1000:.2f} кПа")
    print(f"E_T' = {reduced_total_efficiency*100:.1f}%")
    print(f"E_T  = {total_efficiency*100:.1f}%")
    print()


def plot_hydrocyclone_analysis_Q(Q: float, Cv: float, Dc: float) -> None:
    """Построение графиков анализа гидроциклонов при заданном расходе."""
    _plot_hydrocyclone_analysis(
        'Q', Dc, Cv, flow_rate=Q)


def plot_hydrocyclone_analysis_delta_p(delta_p: float, Cv: float, Dc: float) -> None:
    """Построение графиков анализа гидроциклонов при заданном перепаде давления."""
    _plot_hydrocyclone_analysis(
        'delta_p', Dc, Cv, pressure_drop=delta_p)


def _plot_hydrocyclone_analysis(
    mode: Literal['Q', 'delta_p'],
    Dc: float,
    Cv: float,
    flow_rate: float = 0.0,
    pressure_drop: float = 0.0,
) -> None:
    """Общая логика построения графиков."""
    hydrocyclones = build_standard_configs(Dc)

    # NOTE Тоже очень частная функция, если что-то меняется - нужно переписывать.
    # NOTE Такой код следует делать как пример в examples, в модуле должны лежать
    # NOTE только универсальные для решаемой задачи вещи

    properties = PhysicalProperties(
        mu=0.001,   # вязкость воды при 20°C, Па·с
        rho=1000,   # плотность воды при 20°C, кг/м³
        rhos=2650,  # плотность кварцевого песка, кг/м³
    )

    k = 50e-6  # характерный размер частиц Розин-Раммлера: 50 мкм
    n = 1.5    # параметр Розин-Раммлера
    particle_diameters = np.linspace(0, 1e-3, 500)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # row=0 Rietema, row=1 Bradley, row=2 Demco
    for row, hydrocyclone in enumerate(hydrocyclones):
        if mode == 'Q':
            results = hydrocyclone.calculate_from_flow_rate(
                properties, flow_rate, Cv)  # расчёт при заданном расходе
        else:
            results = hydrocyclone.calculate_from_pressure_drop(
                properties, pressure_drop, Cv)  # расчёт при заданном ΔP
        _plot_row(axes[row], hydrocyclone, results,
                  particle_diameters, k, n, mode)

    plt.tight_layout()
    plt.show()

# NOTE содержимое надо раскидать по модулям


if __name__ == "__main__":
    print("\n" + "="*60)
    print("МОДЕЛЬ РАСЧЁТА ГИДРОЦИКЛОНА")
    print("="*60)

    # print("\n[Режим 1] Фиксированный перепад давления")
    # plot_hydrocyclone_analysis_delta_p(delta_p=100000, Cv=0.05, Dc=50e-3)

    print("\n[Режим 2] Фиксированный расход")
    plot_hydrocyclone_analysis_Q(Q=0.0002, Cv=0.05, Dc=45e-3)
