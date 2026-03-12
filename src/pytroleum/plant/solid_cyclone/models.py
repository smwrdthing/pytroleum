"""
Модель расчёта гидроциклона.
"""
from __future__ import annotations

from abc import ABC

import numpy as np

from pytroleum.plant.solid_cyclone.properties import PhysicalProperties
from pytroleum.plant.solid_cyclone.geometry import (
    GeometryParameters,
    HydrocycloneDiameters,
    HydrocycloneLengths,
    DI_DC_MIN, DI_DC_MAX,
    DO_DC_MIN, DO_DC_MAX,
    DU_DC_MIN, DU_DC_MAX,
    l_VORTEX_DC_MIN, l_VORTEX_DC_MAX,
    L_DC_MIN, L_DC_MAX,
    CYCLONE_CONE_ANGLE_MIN, CYCLONE_CONE_ANGLE_MAX,
)


# ---------------------------------------------------------------------------
# Константы моделей (параметры функции приведённой вероятности уноса)
# ---------------------------------------------------------------------------

RIETEMA_ALPHA = 4.23
RIETEMA_M = 2.45

BRADLEY_ALPHA = 5.10
BRADLEY_M = 3.12

DEMCO_ALPHA = 5.40
DEMCO_M = 3.30


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _euler_number(
    geometry: GeometryParameters,
    feed_volumetric_concentration: float,
    Re: float,
) -> float:
    """Расчёт числа Эйлера Eu."""
    hydrocyclone_diameter = geometry.diameters[HydrocycloneDiameters.C]
    feed_inlet_diameter = geometry.diameters[HydrocycloneDiameters.I]
    overflow_diameter = geometry.diameters[HydrocycloneDiameters.O]
    underflow_diameter = geometry.diameters[HydrocycloneDiameters.U]
    L_minus_l = (geometry.lengths[HydrocycloneLengths.T] -
                 geometry.lengths[HydrocycloneLengths.V])

    return (43.5 * hydrocyclone_diameter**0.57 *
            (hydrocyclone_diameter / feed_inlet_diameter)**2.61 *
            (hydrocyclone_diameter /
             (overflow_diameter**2 + underflow_diameter**2))**0.42 *
            (hydrocyclone_diameter / L_minus_l)**0.98 *
            Re**0.12 *
            np.exp(-0.51 * feed_volumetric_concentration))


# ---------------------------------------------------------------------------
# Базовый класс и конкретные модели
# ---------------------------------------------------------------------------

class BaseHydrocyclone(ABC):
    """Абстрактный базовый класс для моделей гидроциклонов (твёрдые частицы-жидкость)."""

    def __init__(self, name: str, geometry: GeometryParameters) -> None:
        self.name = name
        self.geometry = geometry
        self.alpha: float
        self.m: float

    def get_geometry_ratios(self) -> dict[str, float]:
        """Возвращает словарь безразмерных пропорций геометрии гидроциклона."""
        Dc = self.geometry.diameters[HydrocycloneDiameters.C]
        return {
            'Di/Dc': self.geometry.diameters[HydrocycloneDiameters.I] / Dc,
            'Do/Dc': self.geometry.diameters[HydrocycloneDiameters.O] / Dc,
            'Du/Dc': self.geometry.diameters[HydrocycloneDiameters.U] / Dc,
            'l/Dc': self.geometry.lengths[HydrocycloneLengths.V] / Dc,
            'L/Dc': self.geometry.lengths[HydrocycloneLengths.T] / Dc,
        }
        # NOTE сделать из этой функции summary, пусть summary печатает финальные размеры
        # NOTE если пропорции очень нужно выводить - можно это сделать в скобках рядом с
        # NOTE самим размером
        # NOTE
        # NOTE Размеры выводить в легко-воспринимаемых единицах (здесь в мм)

    def print_proportions(self) -> None:
        """Вывод геометрических пропорций и проверка их соответствия диапазонам."""
        ratios = self.get_geometry_ratios()
        violations = self.geometry.check_proportions()

        print(f"\n=== {self.name} proportions ===")
        print(f"Di/Dc = {ratios['Di/Dc']:.3f}  [{DI_DC_MIN}-{DI_DC_MAX}]")
        print(f"Do/Dc = {ratios['Do/Dc']:.3f}  [{DO_DC_MIN}-{DO_DC_MAX}]")
        print(f"Du/Dc = {ratios['Du/Dc']:.3f}  [{DU_DC_MIN}-{DU_DC_MAX}]")
        print(
            f"l/Dc = {ratios['l/Dc']:.3f}  [{l_VORTEX_DC_MIN}-{l_VORTEX_DC_MAX}]")
        print(f"L/Dc = {ratios['L/Dc']:.3f}  [{L_DC_MIN}-{L_DC_MAX}]")
        print(f"θ     = {self.geometry.angle:.1f}°  "
              f"[{CYCLONE_CONE_ANGLE_MIN}°-{CYCLONE_CONE_ANGLE_MAX}°]")

        if violations:
            print("Пропорции нарушены:")
            for v in violations:
                print(f"  {v}")
        else:
            print("Все пропорции в допустимых диапазонах")
        print()

    def calculate_from_flow_rate(
        self,
        properties: PhysicalProperties,
        feed_volumetric_flow_rate: float,
        feed_volumetric_concentration: float,
    ) -> dict[str, float]:
        """Расчёт параметров при заданном объёмном расходе (ΔP = (Q/K)^(1/0.472))."""
        K = self._compute_K(properties, feed_volumetric_concentration)
        pressure_drop = (feed_volumetric_flow_rate / K) ** (1 / 0.472)
        return self._compute_results(
            properties, feed_volumetric_flow_rate,
            pressure_drop, feed_volumetric_concentration)

    def calculate_from_pressure_drop(
        self,
        properties: PhysicalProperties,
        pressure_drop: float,
        feed_volumetric_concentration: float,
    ) -> dict[str, float]:
        """Расчёт параметров при заданном перепаде давления (Q = K·ΔP^0.472)."""
        K = self._compute_K(properties, feed_volumetric_concentration)
        feed_volumetric_flow_rate = K * pressure_drop**0.472
        return self._compute_results(
            properties, feed_volumetric_flow_rate,
            pressure_drop, feed_volumetric_concentration)

    def _compute_K(
        self,
        properties: PhysicalProperties,
        feed_volumetric_concentration: float,
    ) -> float:
        """Коэффициент K в уравнении Q = K · ΔP^0.472."""
        hydrocyclone_diameter = self.geometry.diameters[HydrocycloneDiameters.C]
        feed_inlet_diameter = self.geometry.diameters[HydrocycloneDiameters.I]
        overflow_diameter = self.geometry.diameters[HydrocycloneDiameters.O]
        under_flow_diameter = self.geometry.diameters[HydrocycloneDiameters.U]
        L_minus_l = (self.geometry.lengths[HydrocycloneLengths.T] -
                     self.geometry.lengths[HydrocycloneLengths.V])

        return (0.184 * hydrocyclone_diameter**(-0.217) * feed_inlet_diameter**(1.231) *
                (overflow_diameter**2 + under_flow_diameter**2)**0.198 *
                L_minus_l**0.462 *
                properties.liquid_eos.viscosity()**(0.0566) *
                properties.liquid_eos.rhomass()**(-0.528) *
                np.exp(0.241 * feed_volumetric_concentration))

    def _compute_results(
        self,
        properties: PhysicalProperties,
        feed_volumetric_flow_rate: float,
        pressure_drop: float,
        feed_volumetric_concentration: float,
    ) -> dict[str, float]:
        """Расчёт всех выходных параметров гидроциклона."""
        hydrocyclone_diameter = self.geometry.diameters[HydrocycloneDiameters.C]
        overflow_diameter = self.geometry.diameters[HydrocycloneDiameters.O]
        under_flow_diameter = self.geometry.diameters[HydrocycloneDiameters.U]
        L_minus_l = (self.geometry.lengths[HydrocycloneLengths.T] -
                     self.geometry.lengths[HydrocycloneLengths.V])

        Re = (4 * properties.liquid_eos.rhomass() * feed_volumetric_flow_rate /
              (np.pi * properties.liquid_eos.viscosity() * hydrocyclone_diameter))
        Eu = _euler_number(self.geometry, feed_volumetric_concentration, Re)

        water_flow_ratio = (1.18 *
                            (hydrocyclone_diameter / overflow_diameter)**5.97 *
                            (under_flow_diameter / hydrocyclone_diameter)**3.10 *
                            Eu**(-0.54))

        rhos_minus_rho = properties.solid_density - properties.liquid_eos.rhomass()

        reduced_cut_size = (1.173 * hydrocyclone_diameter**0.64 /
                            (overflow_diameter**0.475 * L_minus_l**0.665) *
                            np.sqrt((properties.liquid_eos.viscosity() *
                                     properties.liquid_eos.rhomass() *
                                     feed_volumetric_flow_rate) /
                                    (rhos_minus_rho * pressure_drop)) *
                            np.log(1 / water_flow_ratio)**0.395 *
                            np.exp(6.0 * feed_volumetric_concentration))

        return {
            'feed_volumetric_flow_rate': feed_volumetric_flow_rate,
            'pressure_drop': pressure_drop,
            'water_flow_ratio': water_flow_ratio,
            'Re': Re,
            'Eu': Eu,
            'reduced_cut_size': reduced_cut_size,
            'alpha': self.alpha,
            'm': self.m,
        }


class RietemaHydrocyclone(BaseHydrocyclone):
    """Гидроциклон по модели Rietema."""

    def __init__(self, name: str, geometry: GeometryParameters) -> None:
        super().__init__(name, geometry)
        self.alpha = RIETEMA_ALPHA
        self.m = RIETEMA_M


class BradleyHydrocyclone(BaseHydrocyclone):
    """Гидроциклон по модели Bradley."""

    def __init__(self, name: str, geometry: GeometryParameters) -> None:
        super().__init__(name, geometry)
        self.alpha = BRADLEY_ALPHA
        self.m = BRADLEY_M


class DemcoHydrocyclone(BaseHydrocyclone):
    """Гидроциклон по модели Demco."""

    def __init__(self, name: str, geometry: GeometryParameters) -> None:
        super().__init__(name, geometry)
        self.alpha = DEMCO_ALPHA
        self.m = DEMCO_M
