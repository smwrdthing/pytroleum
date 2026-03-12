"""
Геометрия гидроциклона и фабричные функции для стандартных конфигураций.
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

# Стандартные пропорции конфигураций [Di/Dc, Do/Dc, Du/Dc, L/Dc, l/Dc, angle]
RIETEMA_DEFAULT_PROPORTIONS = [0.20, 0.25, 0.15, 4.50, 0.40, 15.0]
BRADLEY_DEFAULT_PROPORTIONS = [0.16, 0.22, 0.12, 5.50, 0.45, 12.0]
DEMCO_DEFAULT_PROPORTIONS = [0.25, 0.30, 0.20, 5.00, 0.50, 18.0]

# именованные индексы пропорций
IDX_DI_DC, IDX_DO_DC, IDX_DU_DC, IDX_L_DC, IDX_l_DC, IDX_ANGLE = 0, 1, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# Индексы массивов геометрических параметров
# ---------------------------------------------------------------------------

class HydrocycloneDiameters(IntEnum):
    C, CYCLONE = 0, 0        # индекс 0 — диаметр корпуса циклона Dc
    I, INLET = 1, 1          # индекс 1 — диаметр входного патрубка Di
    O, OVERFLOW = 2, 2       # индекс 2 — диаметр патрубка верхнего слива Do
    U, UNDERFLOW = 3, 3      # индекс 3 — диаметр патрубка нижнего слива Du

    SIZE = auto()            # количество диаметров = 4


class HydrocycloneLengths(IntEnum):
    T, TOTAL = 0, 0                  # индекс 0 — полная длина циклона L
    V, VORTEX_FINDER = 1, 1          # индекс 1 — длина вихревой трубки l
    C, CYLINDRICAL = 2, 2            # индекс 2 — длина цилиндрической части Lc

    SIZE = auto()                    # количество длин = 3


# ---------------------------------------------------------------------------
# Датакласс геометрии
# ---------------------------------------------------------------------------

@dataclass
class GeometryParameters:
    """Геометрические параметры гидроциклона."""

    angle: float
    diameters: NDArray = field(
        default_factory=lambda: np.zeros(HydrocycloneDiameters.SIZE))
    lengths: NDArray = field(
        default_factory=lambda: np.zeros(HydrocycloneLengths.SIZE))

    def check_proportions(self) -> list[str]:
        """Проверка соответствия геометрических пропорций допустимому диапазону."""
        violations = []
        Dc = self.diameters[HydrocycloneDiameters.C]

        Di_Dc = self.diameters[HydrocycloneDiameters.I] / Dc
        Do_Dc = self.diameters[HydrocycloneDiameters.O] / Dc
        Du_Dc = self.diameters[HydrocycloneDiameters.U] / Dc
        l_Dc = self.lengths[HydrocycloneLengths.V] / Dc
        L_Dc = self.lengths[HydrocycloneLengths.T] / Dc

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


# ---------------------------------------------------------------------------
# Фабричная функция создания GeometryParameters
# ---------------------------------------------------------------------------

def build_geometry(
    hydrocyclone_diameter: float,
    feed_inlet_diameter: float,
    overflow_diameter: float,
    underflow_diameter: float,
    hydrocyclone_length: float,
    vortex_finder_length: float,
    angle: float,
) -> GeometryParameters:
    """
    Создание GeometryParameters из именованных абсолютных размеров.
    """
    angle_rad = np.radians(angle)
    cylindrical_length = hydrocyclone_length - (
        (hydrocyclone_diameter - underflow_diameter) / (2 * np.tan(angle_rad / 2))
    )

    geometry = GeometryParameters(angle=angle)
    geometry.diameters[HydrocycloneDiameters.C] = hydrocyclone_diameter
    geometry.diameters[HydrocycloneDiameters.I] = feed_inlet_diameter
    geometry.diameters[HydrocycloneDiameters.O] = overflow_diameter
    geometry.diameters[HydrocycloneDiameters.U] = underflow_diameter
    geometry.lengths[HydrocycloneLengths.T] = hydrocyclone_length
    geometry.lengths[HydrocycloneLengths.V] = vortex_finder_length
    geometry.lengths[HydrocycloneLengths.C] = cylindrical_length
    return geometry


# ---------------------------------------------------------------------------
# Фабричные функции конфигураций гидроциклонов
# ---------------------------------------------------------------------------

def build_rietema_config(
    hydrocyclone_diameter: float,
    proportions: list[float] = RIETEMA_DEFAULT_PROPORTIONS,
):
    """Возвращает стандартную конфигурацию гидроциклона Rietema."""
    from pytroleum.plant.solid_cyclone.models import RietemaHydrocyclone

    geometry = build_geometry(
        hydrocyclone_diameter=hydrocyclone_diameter,
        feed_inlet_diameter=hydrocyclone_diameter * proportions[IDX_DI_DC],
        overflow_diameter=hydrocyclone_diameter * proportions[IDX_DO_DC],
        underflow_diameter=hydrocyclone_diameter * proportions[IDX_DU_DC],
        hydrocyclone_length=hydrocyclone_diameter * proportions[IDX_L_DC],
        vortex_finder_length=hydrocyclone_diameter * proportions[IDX_l_DC],
        angle=proportions[IDX_ANGLE],
    )
    return RietemaHydrocyclone('Rietema', geometry)


def build_bradley_config(
    hydrocyclone_diameter: float,
    proportions: list[float] = BRADLEY_DEFAULT_PROPORTIONS,
):
    """Возвращает стандартную конфигурацию гидроциклона Bradley."""
    from pytroleum.plant.solid_cyclone.models import BradleyHydrocyclone

    geometry = build_geometry(
        hydrocyclone_diameter=hydrocyclone_diameter,
        feed_inlet_diameter=hydrocyclone_diameter * proportions[IDX_DI_DC],
        overflow_diameter=hydrocyclone_diameter * proportions[IDX_DO_DC],
        underflow_diameter=hydrocyclone_diameter * proportions[IDX_DU_DC],
        hydrocyclone_length=hydrocyclone_diameter * proportions[IDX_L_DC],
        vortex_finder_length=hydrocyclone_diameter * proportions[IDX_l_DC],
        angle=proportions[IDX_ANGLE],
    )
    return BradleyHydrocyclone('Bradley', geometry)


def build_demco_config(
    hydrocyclone_diameter: float,
    proportions: list[float] = DEMCO_DEFAULT_PROPORTIONS,
):
    """Возвращает стандартную конфигурацию гидроциклона Demco."""
    from pytroleum.plant.solid_cyclone.models import DemcoHydrocyclone

    geometry = build_geometry(
        hydrocyclone_diameter=hydrocyclone_diameter,
        feed_inlet_diameter=hydrocyclone_diameter * proportions[IDX_DI_DC],
        overflow_diameter=hydrocyclone_diameter * proportions[IDX_DO_DC],
        underflow_diameter=hydrocyclone_diameter * proportions[IDX_DU_DC],
        hydrocyclone_length=hydrocyclone_diameter * proportions[IDX_L_DC],
        vortex_finder_length=hydrocyclone_diameter * proportions[IDX_l_DC],
        angle=proportions[IDX_ANGLE],
    )
    return DemcoHydrocyclone('Demco', geometry)


def build_standard_configs(hydrocyclone_diameter: float) -> list:
    """
    Возвращает список стандартных конфигураций [Rietema, Bradley, Demco]
    с пропорциями в допустимых диапазонах.
    """
    from pytroleum.plant.solid_cyclone.models import BaseHydrocyclone

    hydrocyclones: list[BaseHydrocyclone] = [
        build_rietema_config(hydrocyclone_diameter),
        build_bradley_config(hydrocyclone_diameter),
        build_demco_config(hydrocyclone_diameter),
    ]

    print("\n" + "=" * 60)
    print(f"ПРОВЕРКА ГЕОМЕТРИЧЕСКИХ ПРОПОРЦИЙ "
          f"(Dc = {hydrocyclone_diameter * 1000:.1f} мм)")
    print("=" * 60)
    for h in hydrocyclones:
        h.print_proportions()
    return hydrocyclones
