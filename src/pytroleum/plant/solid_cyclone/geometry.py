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
    from pytroleum.plant.solid_cyclone.inputs import OperationConditions

from pytroleum.plant.solid_cyclone.utils import _minor_divider, _major_divider

_TO_MM = 1000

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
# Standard configuration proportions.
# ---------------------------------------------------------------------------

# diameter_proportions[i] corresponds to
# HydrocycloneDiameters(i): [Dc/Dc, Di/Dc, Do/Dc, Du/Dc]

# length_proportions[i] corresponds to
# HydrocycloneLengths(i): [L/Dc, l/Dc]


RIETEMA_DIAMETER_PROPORTIONS = np.array([1.0, 0.20, 0.25, 0.15])
RIETEMA_LENGTH_PROPORTIONS = np.array([4.50, 0.40])
RIETEMA_CONE_ANGLE = 15.0

BRADLEY_DIAMETER_PROPORTIONS = np.array([1.0, 0.16, 0.22, 0.12])
BRADLEY_LENGTH_PROPORTIONS = np.array([5.50, 0.45])
BRADLEY_CONE_ANGLE = 12.0

DEMCO_DIAMETER_PROPORTIONS = np.array([1.0, 0.25, 0.30, 0.20])
DEMCO_LENGTH_PROPORTIONS = np.array([5.00, 0.50])
DEMCO_CONE_ANGLE = 18.0


# ---------------------------------------------------------------------------
# Hydrocyclone geometry dataclass
# ---------------------------------------------------------------------------

@dataclass
class CycloneDesign:
    """Hydrocyclone geometry: body diameter + proportions → computes dimensions."""

    hydrocyclone_diameter: float
    diameter_proportions: NDArray   # [Dc/Dc, Di/Dc, Do/Dc, Du/Dc]
    length_proportions: NDArray     # [L/Dc, l/Dc]
    cone_angle: float                   # degrees

    diameters: NDArray = field(init=False)
    lengths: NDArray = field(init=False)

    def __post_init__(self) -> None:
        self._compute_geometry()

    def _compute_geometry(self) -> None:
        Dc = self.hydrocyclone_diameter
        angle_rad = np.radians(self.cone_angle)

        self.diameters = Dc * self.diameter_proportions

        self.lengths = np.zeros(HydrocycloneLengths.SIZE)
        self.lengths[:HydrocycloneLengths.C] = Dc * self.length_proportions
        self.lengths[HydrocycloneLengths.C] = (
            self.lengths[HydrocycloneLengths.T] -
            (Dc - self.diameters[HydrocycloneDiameters.U]) /
            (2 * np.tan(angle_rad / 2))
        )

    def check(self) -> list[str]:
        """Check proportions against valid ranges."""
        dp = self.diameter_proportions
        lp = self.length_proportions
        violations = []

        if not (DI_DC_MIN <= dp[HydrocycloneDiameters.I] <= DI_DC_MAX):
            violations.append(
                f"Di/Dc={dp[HydrocycloneDiameters.I]:.3f} out of range "
                f"[{DI_DC_MIN}–{DI_DC_MAX}]")
        if not (DO_DC_MIN <= dp[HydrocycloneDiameters.O] <= DO_DC_MAX):
            violations.append(
                f"Do/Dc={dp[HydrocycloneDiameters.O]:.3f} out of range "
                f"[{DO_DC_MIN}–{DO_DC_MAX}]")
        if not (DU_DC_MIN <= dp[HydrocycloneDiameters.U] <= DU_DC_MAX):
            violations.append(
                f"Du/Dc={dp[HydrocycloneDiameters.U]:.3f} out of range "
                f"[{DU_DC_MIN}–{DU_DC_MAX}]")
        if not (l_VORTEX_DC_MIN <= lp[HydrocycloneLengths.V] <= l_VORTEX_DC_MAX):
            violations.append(
                f"l/Dc={lp[HydrocycloneLengths.V]:.3f} out of range "
                f"[{l_VORTEX_DC_MIN}–{l_VORTEX_DC_MAX}]")
        if not (L_DC_MIN <= lp[HydrocycloneLengths.T] <= L_DC_MAX):
            violations.append(
                f"L/Dc={lp[HydrocycloneLengths.T]:.3f} out of range "
                f"[{L_DC_MIN}–{L_DC_MAX}]")
        if not (CYCLONE_CONE_ANGLE_MIN <= self.cone_angle <= CYCLONE_CONE_ANGLE_MAX):
            violations.append(
                f"θ={self.cone_angle:.1f}° out of range "
                f"[{CYCLONE_CONE_ANGLE_MIN}°–{CYCLONE_CONE_ANGLE_MAX}°]")

        return violations

    def summary(self) -> None:
        """Print geometric dimensions (in mm) with proportions and range checks."""
        dp = self.diameter_proportions
        lp = self.length_proportions
        d = self.diameters
        le = self.lengths

        _major_divider()
        print("HYDROCYCLONE DESIGN SUMMARY")
        _major_divider()

        print("DIAMETERS:")
        print(f" Cyclone: {d[HydrocycloneDiameters.C]*_TO_MM:.2f} mm")
        print(f" Inlet: {d[HydrocycloneDiameters.I]*_TO_MM:.2f} mm  "
              f" (Di/Dc = {dp[HydrocycloneDiameters.I]:.3f}, "
              f" range [{DI_DC_MIN}–{DI_DC_MAX}])")
        print(f" Overflow: {d[HydrocycloneDiameters.O]*_TO_MM:.2f} mm  "
              f" (Do/Dc = {dp[HydrocycloneDiameters.O]:.3f}, "
              f" range [{DO_DC_MIN}–{DO_DC_MAX}])")
        print(f" Underflow: {d[HydrocycloneDiameters.U]*_TO_MM:.2f} mm  "
              f" (Du/Dc = {dp[HydrocycloneDiameters.U]:.3f}, "
              f" range [{DU_DC_MIN}–{DU_DC_MAX}])")

        _minor_divider()

        print("LENGTHS:")
        print(f" Total: {le[HydrocycloneLengths.T]*_TO_MM:.2f} mm  "
              f" (L/Dc = {lp[HydrocycloneLengths.T]:.3f}, "
              f" range [{L_DC_MIN}–{L_DC_MAX}])")
        print(f" Vortex: {le[HydrocycloneLengths.V]*_TO_MM:.2f} mm"
              f" (l/Dc = {lp[HydrocycloneLengths.V]:.3f}, "
              f" range [{l_VORTEX_DC_MIN}–{l_VORTEX_DC_MAX}])")
        print(f" Cylinder: {le[HydrocycloneLengths.C]*_TO_MM:.2f} mm")

        _minor_divider()

        print(f" Cone angle: {self.cone_angle:.1f}°"
              f" (range [{CYCLONE_CONE_ANGLE_MIN}°–{CYCLONE_CONE_ANGLE_MAX}°])")

        violations = self.check()
        if violations:
            print(" Proportion violations:")
            for v in violations:
                print(f" {v}")
        else:
            print(" All proportions within valid ranges")

        _major_divider()
        print()


# ---------------------------------------------------------------------------
# Hydrocyclone configuration factory functions
# ---------------------------------------------------------------------------

def build_rietema_config(
    hydrocyclone_diameter: float,
    conditions: OperationConditions,
    diameter_proportions: NDArray = RIETEMA_DIAMETER_PROPORTIONS,
    length_proportions: NDArray = RIETEMA_LENGTH_PROPORTIONS,
    cone_angle: float = RIETEMA_CONE_ANGLE,
) -> RietemaHydrocyclone:
    """Return standard Rietema hydrocyclone configuration."""
    from pytroleum.plant.solid_cyclone.models import RietemaHydrocyclone
    return RietemaHydrocyclone(
        'Rietema',
        CycloneDesign(hydrocyclone_diameter, diameter_proportions,
                      length_proportions, cone_angle),
        conditions,
    )


def build_bradley_config(
    hydrocyclone_diameter: float,
    conditions: OperationConditions,
    diameter_proportions: NDArray = BRADLEY_DIAMETER_PROPORTIONS,
    length_proportions: NDArray = BRADLEY_LENGTH_PROPORTIONS,
    cone_angle: float = BRADLEY_CONE_ANGLE,
) -> BradleyHydrocyclone:
    """Return standard Bradley hydrocyclone configuration."""
    from pytroleum.plant.solid_cyclone.models import BradleyHydrocyclone
    return BradleyHydrocyclone(
        'Bradley',
        CycloneDesign(hydrocyclone_diameter, diameter_proportions,
                      length_proportions, cone_angle),
        conditions,
    )


def build_demco_config(
    hydrocyclone_diameter: float,
    conditions: OperationConditions,
    diameter_proportions: NDArray = DEMCO_DIAMETER_PROPORTIONS,
    length_proportions: NDArray = DEMCO_LENGTH_PROPORTIONS,
    cone_angle: float = DEMCO_CONE_ANGLE,
) -> DemcoHydrocyclone:
    """Return standard Demco hydrocyclone configuration."""
    from pytroleum.plant.solid_cyclone.models import DemcoHydrocyclone
    return DemcoHydrocyclone(
        'Demco',
        CycloneDesign(hydrocyclone_diameter, diameter_proportions,
                      length_proportions, cone_angle),
        conditions,
    )


def build_standard_configs(
    hydrocyclone_diameter: float,
    conditions: OperationConditions,
) -> list[BaseHydrocyclone]:
    """Return list of standard configurations [Rietema, Bradley, Demco]."""
    return [
        build_rietema_config(hydrocyclone_diameter, conditions),
        build_bradley_config(hydrocyclone_diameter, conditions),
        build_demco_config(hydrocyclone_diameter, conditions),
    ]


def print_standard_configs_summary(
    hydrocyclone_diameter: float,
    conditions: OperationConditions,
) -> None:
    """Print geometry summary for the three standard configurations."""
    for hydrocyclone in build_standard_configs(hydrocyclone_diameter, conditions):
        hydrocyclone.design.summary()
