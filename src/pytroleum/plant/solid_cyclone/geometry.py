"""
Геометрия гидроциклона и фабричные функции для стандартных конфигураций.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pytroleum.plant.solid_cyclone.models import BaseHydrocyclone


# ---------------------------------------------------------------------------
# Допустимые диапазоны пропорций (Coelho & Medronho, 2001)
# ---------------------------------------------------------------------------

DI_DC_MIN = 0.14    # минимально допустимое отношение Di/Dc
DI_DC_MAX = 0.28    # максимально допустимое отношение Di/Dc

DO_DC_MIN = 0.20    # минимально допустимое отношение Do/Dc
DO_DC_MAX = 0.34    # максимально допустимое отношение Do/Dc

DU_DC_MIN = 0.04    # минимально допустимое отношение Du/Dc
DU_DC_MAX = 0.28    # максимально допустимое отношение Du/Dc

l_VORTEX_DC_MIN = 0.33  # минимально допустимое отношение l/Dc
l_VORTEX_DC_MAX = 0.55  # максимально допустимое отношение l/Dc

L_DC_MIN = 3.30  # минимально допустимое отношение L/Dc
L_DC_MAX = 6.93  # максимально допустимое отношение L/Dc

CYCLONE_CONE_ANGLE_MIN = 9.0    # минимально допустимый угол конуса, градусы
CYCLONE_CONE_ANGLE_MAX = 20.0   # максимально допустимый угол конуса, градусы

# Стандартные пропорции конфигураций [Di/Dc, Do/Dc, Du/Dc, L/Dc, l/Dc, angle]
RIETEMA_DEFAULT_PROPORTIONS = [0.20, 0.25, 0.15, 4.50, 0.40, 15.0]
BRADLEY_DEFAULT_PROPORTIONS = [0.16, 0.22, 0.12, 5.50, 0.45, 12.0]
DEMCO_DEFAULT_PROPORTIONS = [0.25, 0.30, 0.20, 5.00, 0.50, 18.0]

# Именованные индексы пропорций
IDX_DI_DC, IDX_DO_DC, IDX_DU_DC, IDX_L_DC, IDX_l_DC, IDX_ANGLE = 0, 1, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# Вспомогательные функции форматирования
# ---------------------------------------------------------------------------

_DIVIDER_LENGTH = 60
_MINOR_DIVIDER = '-' * _DIVIDER_LENGTH
_MAJOR_DIVIDER = '=' * _DIVIDER_LENGTH


def _minor_divider() -> None:
    print(_MINOR_DIVIDER)


def _major_divider() -> None:
    print(_MAJOR_DIVIDER)


# ---------------------------------------------------------------------------
# Индексы массивов геометрических параметров
# ---------------------------------------------------------------------------

class HydrocycloneDiameters(IntEnum):
    C, CYCLONE = 0, 0    # индекс 0 — диаметр корпуса циклона Dc
    I, INLET = 1, 1      # индекс 1 — диаметр входного патрубка Di
    O, OVERFLOW = 2, 2   # индекс 2 — диаметр патрубка верхнего слива Do
    U, UNDERFLOW = 3, 3  # индекс 3 — диаметр патрубка нижнего слива Du

    SIZE = auto()   # количество диаметров = 4


class HydrocycloneLengths(IntEnum):
    T, TOTAL = 0, 0             # индекс 0 — полная длина циклона L
    V, VORTEX_FINDER = 1, 1     # индекс 1 — длина вихревой трубки l
    C, CYLINDRICAL = 2, 2       # индекс 2 — длина цилиндрической части Lc

    SIZE = auto()               # количество длин = 3


# ---------------------------------------------------------------------------
# Датакласс геометрии гидроциклона
# ---------------------------------------------------------------------------

@dataclass
class CycloneDesign:
    """Геометрия гидроциклона: диаметр корпуса + пропорции → вычисляет размеры."""

    hydrocyclone_diameter: float
    proportions: list[float]

    diameters: NDArray = field(init=False)
    lengths: NDArray = field(init=False)

    def __post_init__(self) -> None:
        self._compute_geometry()

    def _compute_geometry(self) -> None:
        Dc = self.hydrocyclone_diameter
        angle_rad = np.radians(self.proportions[IDX_ANGLE])

        self.diameters = np.zeros(HydrocycloneDiameters.SIZE)
        self.diameters[HydrocycloneDiameters.C] = Dc
        self.diameters[HydrocycloneDiameters.I] = Dc * \
            self.proportions[IDX_DI_DC]
        self.diameters[HydrocycloneDiameters.O] = Dc * \
            self.proportions[IDX_DO_DC]
        self.diameters[HydrocycloneDiameters.U] = Dc * \
            self.proportions[IDX_DU_DC]

        self.lengths = np.zeros(HydrocycloneLengths.SIZE)
        self.lengths[HydrocycloneLengths.T] = Dc * self.proportions[IDX_L_DC]
        self.lengths[HydrocycloneLengths.V] = Dc * self.proportions[IDX_l_DC]
        self.lengths[HydrocycloneLengths.C] = (
            self.lengths[HydrocycloneLengths.T] -
            (Dc - self.diameters[HydrocycloneDiameters.U]) /
            (2 * np.tan(angle_rad / 2))
        )

    def check(self) -> list[str]:
        """Проверка соответствия пропорций допустимому диапазону."""
        p = self.proportions
        violations = []

        if not (DI_DC_MIN <= p[IDX_DI_DC] <= DI_DC_MAX):
            violations.append(
                f"Di/Dc={p[IDX_DI_DC]:.3f} out of range [{DI_DC_MIN}–{DI_DC_MAX}]")
        if not (DO_DC_MIN <= p[IDX_DO_DC] <= DO_DC_MAX):
            violations.append(
                f"Do/Dc={p[IDX_DO_DC]:.3f} out of range [{DO_DC_MIN}–{DO_DC_MAX}]")
        if not (DU_DC_MIN <= p[IDX_DU_DC] <= DU_DC_MAX):
            violations.append(
                f"Du/Dc={p[IDX_DU_DC]:.3f} out of range [{DU_DC_MIN}–{DU_DC_MAX}]")
        if not (l_VORTEX_DC_MIN <= p[IDX_l_DC] <= l_VORTEX_DC_MAX):
            violations.append(
                f"l/Dc={p[IDX_l_DC]:.3f} out of range "
                f"[{l_VORTEX_DC_MIN}–{l_VORTEX_DC_MAX}]")
        if not (L_DC_MIN <= p[IDX_L_DC] <= L_DC_MAX):
            violations.append(
                f"L/Dc={p[IDX_L_DC]:.3f} out of range [{L_DC_MIN}–{L_DC_MAX}]")
        if not (CYCLONE_CONE_ANGLE_MIN <= p[IDX_ANGLE] <= CYCLONE_CONE_ANGLE_MAX):
            violations.append(
                f"θ={p[IDX_ANGLE]:.1f}° out of range "
                f"[{CYCLONE_CONE_ANGLE_MIN}°–{CYCLONE_CONE_ANGLE_MAX}°]")

        return violations

    def summary(self) -> None:
        """Вывод геометрических размеров (в мм) с пропорциями и проверкой диапазонов."""
        to_mm = 1000
        p = self.proportions
        d = self.diameters
        le = self.lengths

        _major_divider()
        print("HYDROCYCLONE DESIGN SUMMARY")
        _major_divider()

        print("DIAMETERS:")
        print(f"  Cyclone:   {d[HydrocycloneDiameters.C]*to_mm:.2f} mm")
        print(f"  Inlet:     {d[HydrocycloneDiameters.I]*to_mm:.2f} mm"
              f"  (Di/Dc = {p[IDX_DI_DC]:.3f}, range [{DI_DC_MIN}–{DI_DC_MAX}])")
        print(f"  Overflow:  {d[HydrocycloneDiameters.O]*to_mm:.2f} mm"
              f"  (Do/Dc = {p[IDX_DO_DC]:.3f}, range [{DO_DC_MIN}–{DO_DC_MAX}])")
        print(f"  Underflow: {d[HydrocycloneDiameters.U]*to_mm:.2f} mm"
              f"  (Du/Dc = {p[IDX_DU_DC]:.3f}, range [{DU_DC_MIN}–{DU_DC_MAX}])")

        _minor_divider()

        print("LENGTHS:")
        print(f"  Total:    {le[HydrocycloneLengths.T]*to_mm:.2f} mm"
              f"  (L/Dc = {p[IDX_L_DC]:.3f}, range [{L_DC_MIN}–{L_DC_MAX}])")
        print(f"  Vortex:   {le[HydrocycloneLengths.V]*to_mm:.2f} mm"
              f"  (l/Dc = {p[IDX_l_DC]:.3f}, "
              f"range [{l_VORTEX_DC_MIN}–{l_VORTEX_DC_MAX}])")
        print(f"  Cylinder: {le[HydrocycloneLengths.C]*to_mm:.2f} mm")

        _minor_divider()

        print(f"  Cone angle: {p[IDX_ANGLE]:.1f}°"
              f"  (range [{CYCLONE_CONE_ANGLE_MIN}°–{CYCLONE_CONE_ANGLE_MAX}°])")

        violations = self.check()
        if violations:
            print("Proportion violations:")
            for v in violations:
                print(f"  {v}")
        else:
            print("All proportions within valid ranges")

        _major_divider()
        print()


# ---------------------------------------------------------------------------
# Фабричные функции конфигураций гидроциклонов
# ---------------------------------------------------------------------------

def build_rietema_config(
    hydrocyclone_diameter: float,
    proportions: list[float] = RIETEMA_DEFAULT_PROPORTIONS,
) -> BaseHydrocyclone:
    """Возвращает стандартную конфигурацию гидроциклона Rietema."""
    from pytroleum.plant.solid_cyclone.models import RietemaHydrocyclone
    return RietemaHydrocyclone('Rietema',
                               CycloneDesign(hydrocyclone_diameter, proportions))


def build_bradley_config(
    hydrocyclone_diameter: float,
    proportions: list[float] = BRADLEY_DEFAULT_PROPORTIONS,
) -> BaseHydrocyclone:
    """Возвращает стандартную конфигурацию гидроциклона Bradley."""
    from pytroleum.plant.solid_cyclone.models import BradleyHydrocyclone
    return BradleyHydrocyclone('Bradley',
                               CycloneDesign(hydrocyclone_diameter, proportions))


def build_demco_config(
    hydrocyclone_diameter: float,
    proportions: list[float] = DEMCO_DEFAULT_PROPORTIONS,
) -> BaseHydrocyclone:
    """Возвращает стандартную конфигурацию гидроциклона Demco."""
    from pytroleum.plant.solid_cyclone.models import DemcoHydrocyclone
    return DemcoHydrocyclone('Demco', CycloneDesign(hydrocyclone_diameter, proportions))


def build_standard_configs(hydrocyclone_diameter: float) -> list[BaseHydrocyclone]:
    """Возвращает список стандартных конфигураций [Rietema, Bradley, Demco]."""
    return [
        build_rietema_config(hydrocyclone_diameter),
        build_bradley_config(hydrocyclone_diameter),
        build_demco_config(hydrocyclone_diameter),
    ]


def print_standard_configs_summary(hydrocyclone_diameter: float) -> None:
    """Выводит сводку геометрии для трёх стандартных конфигураций."""
    for hydrocyclone in build_standard_configs(hydrocyclone_diameter):
        hydrocyclone.design.summary()


def build_from_ratios(
    Dc: float,
    ratios: dict[str, float],
    hydrocyclone_cls: type[BaseHydrocyclone],
    name: str = '',
) -> BaseHydrocyclone:
    """Создание экземпляра гидроциклона из диаметра корпуса и словаря пропорций."""
    proportions = [
        ratios['Di/Dc'],
        ratios['Do/Dc'],
        ratios['Du/Dc'],
        ratios['L/Dc'],
        ratios['l/Dc'],
        ratios['angle'],
    ]
    return hydrocyclone_cls(name, CycloneDesign(Dc, proportions))
