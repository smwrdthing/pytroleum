"""
Геометрия гидроциклона.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto

from numpy.typing import NDArray
import numpy as np


# ---------------------------------------------------------------------------
# Допустимые диапазоны пропорций (Coelho & Medronho, 2001)
# ---------------------------------------------------------------------------

DI_DC_MIN = 0.14  # минимально допустимое отношение Di/Dc
DI_DC_MAX = 0.28  # максимально допустимое отношение Di/Dc

DO_DC_MIN = 0.20  # минимально допустимое отношение Do/Dc
DO_DC_MAX = 0.34  # максимально допустимое отношение Do/Dc

DU_DC_MIN = 0.04  # минимально допустимое отношение Du/Dc
DU_DC_MAX = 0.28  # максимально допустимое отношение Du/Dc

l_VORTEX_DC_MIN = 0.33  # минимально допустимое отношение l/Dc
l_VORTEX_DC_MAX = 0.55  # максимально допустимое отношение l/Dc

L_DC_MIN = 3.30  # минимально допустимое отношение L/Dc
L_DC_MAX = 6.93  # максимально допустимое отношение L/Dc

CYCLONE_CONE_ANGLE_MIN = 9.0   # минимально допустимый угол конуса, градусы
CYCLONE_CONE_ANGLE_MAX = 20.0  # максимально допустимый угол конуса, градусы


# ---------------------------------------------------------------------------
# Перечисления индексов
# ---------------------------------------------------------------------------

class HydrocycloneDiameters(IntEnum):
    C, CYCLONE = 0, 0        # индекс 0 — диаметр корпуса циклона Dc
    I, INLET = 1, 1          # индекс 1 — диаметр входного патрубка Di
    O, OVERFLOW = 2, 2       # индекс 2 — диаметр патрубка верхнего слива Do
    U, UNDERFLOW = 3, 3      # индекс 3 — диаметр патрубка нижнего слива Du

    SIZE = auto()            # количество диаметров = 4


class HydrocycloneLengths(IntEnum):
    T, TOTAL = 0, 0          # индекс 0 — полная длина циклона L
    V, VORTEX_FINDER = 1, 1  # индекс 1 — длина вихревой трубки

    SIZE = auto()            # количество длин = 2


# ---------------------------------------------------------------------------
# Датаклассы
# ---------------------------------------------------------------------------

@dataclass
class GeometryParameters:
    """Геометрические параметры гидроциклона."""

    diameters: NDArray = field(
        default_factory=lambda: np.zeros(HydrocycloneDiameters.SIZE))
    lengths: NDArray = field(
        default_factory=lambda: np.zeros(HydrocycloneLengths.SIZE))
    angle: float = 15.0

    @classmethod  # NOTE насколько нам нужен classmethod?
    def from_named(
        cls,
        hydrocyclone_diameter: float,
        feed_inlet_diameter: float,
        overflow_diameter: float,
        underflow_diameter: float,
        hydrocyclone_length: float,
        vortex_finder_length: float,
        angle: float = 15.0,
    ) -> GeometryParameters:
        """Создание объекта из именованных размеров."""
        obj = cls(angle=angle)
        obj.diameters[HydrocycloneDiameters.C] = hydrocyclone_diameter
        obj.diameters[HydrocycloneDiameters.I] = feed_inlet_diameter
        obj.diameters[HydrocycloneDiameters.O] = overflow_diameter
        obj.diameters[HydrocycloneDiameters.U] = underflow_diameter
        obj.lengths[HydrocycloneLengths.T] = hydrocyclone_length
        obj.lengths[HydrocycloneLengths.V] = vortex_finder_length
        return obj

    def check_proportions(self) -> list[str]:
        """Проверка соответствия геометрических пропорций допустимому диапазону."""
        violations = []
        hydrocyclone_diameter = self.diameters[HydrocycloneDiameters.C]

        Di_Dc = self.diameters[HydrocycloneDiameters.I] / hydrocyclone_diameter
        Do_Dc = self.diameters[HydrocycloneDiameters.O] / hydrocyclone_diameter
        Du_Dc = self.diameters[HydrocycloneDiameters.U] / hydrocyclone_diameter
        l_Dc = self.lengths[HydrocycloneLengths.V] / hydrocyclone_diameter
        L_Dc = self.lengths[HydrocycloneLengths.T] / hydrocyclone_diameter

        if not (DI_DC_MIN <= Di_Dc <= DI_DC_MAX):
            violations.append(
                f"Di/Dc={Di_Dc:.3f} вне диапазона [{DI_DC_MIN}-{DI_DC_MAX}]")
        if not (DO_DC_MIN <= Do_Dc <= DO_DC_MAX):
            violations.append(
                f"Do/Dc={Do_Dc:.3f} вне диапазона [{DO_DC_MIN}-{DO_DC_MAX}]")
        if not (DU_DC_MIN <= Du_Dc <= DU_DC_MAX):
            violations.append(
                f"Du/Dc={Du_Dc:.3f} вне диапазона [{DU_DC_MIN}-{DU_DC_MAX}]")
        if not (l_VORTEX_DC_MIN <= l_Dc <= l_VORTEX_DC_MAX):
            violations.append(
                f"l/Dc={l_Dc:.3f} вне диапазона [{l_VORTEX_DC_MIN}-{l_VORTEX_DC_MAX}]")
        if not (L_DC_MIN <= L_Dc <= L_DC_MAX):
            violations.append(
                f"L/Dc={L_Dc:.3f} вне диапазона [{L_DC_MIN}-{L_DC_MAX}]")
        if not (CYCLONE_CONE_ANGLE_MIN <= self.angle <= CYCLONE_CONE_ANGLE_MAX):
            violations.append(
                f"θ={self.angle:.1f}° вне диапазона "
                f"[{CYCLONE_CONE_ANGLE_MIN}°-{CYCLONE_CONE_ANGLE_MAX}°]")

        return violations

    def get_geometry_ratios(self) -> dict[str, float]:
        """Возвращает словарь с геометрическими пропорциями."""
        hydrocyclone_diameter = self.diameters[HydrocycloneDiameters.C]
        return {
            'Di/Dc': self.diameters[HydrocycloneDiameters.I] / hydrocyclone_diameter,
            'Do/Dc': self.diameters[HydrocycloneDiameters.O] / hydrocyclone_diameter,
            'Du/Dc': self.diameters[HydrocycloneDiameters.U] / hydrocyclone_diameter,
            'l/Dc': self.lengths[HydrocycloneLengths.V] / hydrocyclone_diameter,
            'L/Dc': self.lengths[HydrocycloneLengths.T] / hydrocyclone_diameter,
        }
