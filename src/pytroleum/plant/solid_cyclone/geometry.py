"""
Hydrocyclone geometry and factory functions for standard configurations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pytroleum.plant.solid_cyclone.models import (
        BaseHydrocyclone,
        RietemaHydrocyclone,
        BradleyHydrocyclone,
        DemcoHydrocyclone,
    )

from pytroleum.plant.solid_cyclone.utils import _minor_divider, _major_divider

# ---------------------------------------------------------------------------
# Valid proportion ranges (Coelho & Medronho, 2001)
# ---------------------------------------------------------------------------

DI_DC_MIN = 0.14    # minimum allowable Di/Dc ratio
DI_DC_MAX = 0.28    # maximum allowable Di/Dc ratio

DO_DC_MIN = 0.20    # minimum allowable Do/Dc ratio
DO_DC_MAX = 0.34    # maximum allowable Do/Dc ratio

DU_DC_MIN = 0.04    # minimum allowable Du/Dc ratio
DU_DC_MAX = 0.28    # maximum allowable Du/Dc ratio

l_VORTEX_DC_MIN = 0.33  # minimum allowable l/Dc ratio
l_VORTEX_DC_MAX = 0.55  # maximum allowable l/Dc ratio

L_DC_MIN = 3.30  # minimum allowable L/Dc ratio
L_DC_MAX = 6.93  # maximum allowable L/Dc ratio

CYCLONE_CONE_ANGLE_MIN = 9.0    # minimum allowable cone angle, degrees
CYCLONE_CONE_ANGLE_MAX = 20.0   # maximum allowable cone angle, degrees

_TO_MM = 1000

# Standard configuration proportions [Di/Dc, Do/Dc, Du/Dc, L/Dc, l/Dc, angle]
RIETEMA_DEFAULT_PROPORTIONS = [0.20, 0.25, 0.15, 4.50, 0.40, 15.0]
BRADLEY_DEFAULT_PROPORTIONS = [0.16, 0.22, 0.12, 5.50, 0.45, 12.0]
DEMCO_DEFAULT_PROPORTIONS = [0.25, 0.30, 0.20, 5.00, 0.50, 18.0]

# Named proportion indices
IDX_DI_DC, IDX_DO_DC, IDX_DU_DC, IDX_L_DC, IDX_l_DC, IDX_ANGLE = 0, 1, 2, 3, 4, 5
# NOTE эти константы дублируют то, что уже сделано через Enum для диаметров
# NOTE
# NOTE можно завести массив с пропорциями диаметров в формате
# NOTE diameter_proportion = [1, DI_DC, ...]
# NOTE и пользоваться индексами из HydrocycloneDiameters
# NOTE
# NOTE Такая же идея с пропорциями длин
# NOTE
# NOTE Тогда инициализация размеров будет максимально простой:
# NOTE self.diameter = Dc*diameter_proportions <- умножение float на массив
# NOTE
# NOTE Во всех геометриях коническая секция одна, так что угол можно хранить
# NOTE просто в атрибуте

# ---------------------------------------------------------------------------
# Geometry parameter array indices
# ---------------------------------------------------------------------------


class HydrocycloneDiameters(IntEnum):
    C, CYCLONE = 0, 0    # index 0 — cyclone body diameter Dc
    I, INLET = 1, 1      # index 1 — inlet nozzle diameter Di
    O, OVERFLOW = 2, 2   # index 2 — overflow nozzle diameter Do
    U, UNDERFLOW = 3, 3  # index 3 — underflow nozzle diameter Du

    SIZE = auto()   # number of diameters = 4


class HydrocycloneLengths(IntEnum):
    T, TOTAL = 0, 0             # index 0 — total cyclone length L
    V, VORTEX_FINDER = 1, 1     # index 1 — vortex finder length l
    C, CYLINDRICAL = 2, 2       # index 2 — cylindrical section length Lc

    SIZE = auto()               # number of lengths = 3


# ---------------------------------------------------------------------------
# Hydrocyclone geometry dataclass
# ---------------------------------------------------------------------------

@dataclass
class CycloneDesign:
    """Hydrocyclone geometry: body diameter + proportions → computes dimensions."""

    hydrocyclone_diameter: float
    proportions: list[float]
    # NOTE массив с пропорциями уже есть, но в таком виде им не получится пользоваться как
    # NOTE описано в заметке выше (перемешаны пропорции диаметров и длин)

    diameters: NDArray = field(init=False)
    lengths: NDArray = field(init=False)

    def __post_init__(self) -> None:
        self._compute_geometry()

    def _compute_geometry(self) -> None:
        Dc = self.hydrocyclone_diameter
        angle_rad = np.radians(self.proportions[IDX_ANGLE])

        # NOTE см. заметку выше про массив с пропорциями,
        # NOTE вместо 8 строк кода можно уложиться в 1
        self.diameters = np.zeros(HydrocycloneDiameters.SIZE)
        self.diameters[HydrocycloneDiameters.C] = Dc
        self.diameters[HydrocycloneDiameters.I] = Dc * \
            self.proportions[IDX_DI_DC]
        self.diameters[HydrocycloneDiameters.O] = Dc * \
            self.proportions[IDX_DO_DC]
        self.diameters[HydrocycloneDiameters.U] = Dc * \
            self.proportions[IDX_DU_DC]

        # NOTE и здесь тоже
        self.lengths = np.zeros(HydrocycloneLengths.SIZE)
        self.lengths[HydrocycloneLengths.T] = Dc * self.proportions[IDX_L_DC]
        self.lengths[HydrocycloneLengths.V] = Dc * self.proportions[IDX_l_DC]
        self.lengths[HydrocycloneLengths.C] = (
            self.lengths[HydrocycloneLengths.T] -
            (Dc - self.diameters[HydrocycloneDiameters.U]) /
            (2 * np.tan(angle_rad / 2))
        )

    def check(self) -> list[str]:
        """Check proportions against valid ranges."""
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
        """Print geometric dimensions (in mm) with proportions and range checks."""
        p = self.proportions
        d = self.diameters
        le = self.lengths

        _major_divider()
        print("HYDROCYCLONE DESIGN SUMMARY")
        _major_divider()

        print("DIAMETERS:")
        print(f"  Cyclone: {d[HydrocycloneDiameters.C]*_TO_MM:.2f} mm")
        print(f"  Inlet: {d[HydrocycloneDiameters.I]*_TO_MM:.2f} mm"
              f"  (Di/Dc = {p[IDX_DI_DC]:.3f}, range [{DI_DC_MIN}–{DI_DC_MAX}])")
        print(f"  Overflow: {d[HydrocycloneDiameters.O]*_TO_MM:.2f} mm"
              f"  (Do/Dc = {p[IDX_DO_DC]:.3f}, range [{DO_DC_MIN}–{DO_DC_MAX}])")
        print(f"  Underflow: {d[HydrocycloneDiameters.U]*_TO_MM:.2f} mm"
              f"  (Du/Dc = {p[IDX_DU_DC]:.3f}, range [{DU_DC_MIN}–{DU_DC_MAX}])")

        _minor_divider()

        print("LENGTHS:")
        print(f"  Total: {le[HydrocycloneLengths.T]*_TO_MM:.2f} mm"
              f"  (L/Dc = {p[IDX_L_DC]:.3f}, range [{L_DC_MIN}–{L_DC_MAX}])")
        print(f"  Vortex: {le[HydrocycloneLengths.V]*_TO_MM:.2f} mm"
              f"  (l/Dc = {p[IDX_l_DC]:.3f}, "
              f"range [{l_VORTEX_DC_MIN}–{l_VORTEX_DC_MAX}])")
        print(f"  Cylinder: {le[HydrocycloneLengths.C]*_TO_MM:.2f} mm")

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
# Hydrocyclone configuration factory functions
# ---------------------------------------------------------------------------

def build_rietema_config(
    hydrocyclone_diameter: float,
    proportions: list[float] = RIETEMA_DEFAULT_PROPORTIONS,
) -> RietemaHydrocyclone:
    """Return standard Rietema hydrocyclone configuration."""
    from pytroleum.plant.solid_cyclone.models import RietemaHydrocyclone
    return RietemaHydrocyclone('Rietema',
                               CycloneDesign(hydrocyclone_diameter, proportions))


def build_bradley_config(
    hydrocyclone_diameter: float,
    proportions: list[float] = BRADLEY_DEFAULT_PROPORTIONS,
) -> BradleyHydrocyclone:
    """Return standard Bradley hydrocyclone configuration."""
    from pytroleum.plant.solid_cyclone.models import BradleyHydrocyclone
    return BradleyHydrocyclone('Bradley',
                               CycloneDesign(hydrocyclone_diameter, proportions))


def build_demco_config(
    hydrocyclone_diameter: float,
    proportions: list[float] = DEMCO_DEFAULT_PROPORTIONS,
) -> DemcoHydrocyclone:
    """Return standard Demco hydrocyclone configuration."""
    from pytroleum.plant.solid_cyclone.models import DemcoHydrocyclone
    return DemcoHydrocyclone('Demco', CycloneDesign(hydrocyclone_diameter, proportions))


def build_standard_configs(hydrocyclone_diameter: float) -> list[BaseHydrocyclone]:
    """Return list of standard configurations [Rietema, Bradley, Demco]."""
    return [
        build_rietema_config(hydrocyclone_diameter),
        build_bradley_config(hydrocyclone_diameter),
        build_demco_config(hydrocyclone_diameter),
    ]


def print_standard_configs_summary(hydrocyclone_diameter: float) -> None:
    """Print geometry summary for the three standard configurations."""
    for hydrocyclone in build_standard_configs(hydrocyclone_diameter):
        hydrocyclone.design.summary()


def build_from_ratios(
    Dc: float,
    ratios: dict[str, float],
    hydrocyclone_cls: type[BaseHydrocyclone],
    name: str = '',
) -> BaseHydrocyclone:
    # NOTE Эта функция неуместна, если у нас уже есть три функции выше + она работает
    # NOTE с классом, как передаваемым параметром, зачем?
    """Create a hydrocyclone instance from body diameter and a proportions dict."""
    proportions = [
        ratios['Di/Dc'],
        ratios['Do/Dc'],
        ratios['Du/Dc'],
        ratios['L/Dc'],
        ratios['l/Dc'],
        ratios['angle'],
    ]
    return hydrocyclone_cls(name, CycloneDesign(Dc, proportions))
