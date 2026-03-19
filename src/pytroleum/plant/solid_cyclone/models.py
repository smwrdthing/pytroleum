"""
Модель расчёта гидроциклона.
"""
from __future__ import annotations
from abc import ABC
import numpy as np

from pytroleum.plant.solid_cyclone.inputs import PhysicalProperties
from pytroleum.plant.solid_cyclone.geometry import (
    CycloneDesign,
    HydrocycloneDiameters,
    HydrocycloneLengths,
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
# Базовый класс и конкретные модели
# ---------------------------------------------------------------------------


class BaseHydrocyclone(ABC):
    """Абстрактный базовый класс для моделей гидроциклонов (твёрдые частицы-жидкость)."""

    def __init__(self, name: str, design: CycloneDesign) -> None:
        self.name = name
        self.design = design
        self.alpha: float
        self.m: float

    def calculate_from_flow_rate(
        self,
        properties: PhysicalProperties,
        feed_volumetric_flow_rate: float,
        feed_volumetric_concentration: float,
    ) -> dict[str, float]:
        """Расчёт параметров при заданном объёмном расходе (ΔP = (Q/K)^(1/0.472))."""
        K = self.compute_K(properties, feed_volumetric_concentration)
        pressure_drop = (feed_volumetric_flow_rate / K) ** (1 / 0.472)
        return self.compute_results(
            properties, feed_volumetric_flow_rate,
            pressure_drop, feed_volumetric_concentration)

    def calculate_from_pressure_drop(
        self,
        properties: PhysicalProperties,
        pressure_drop: float,
        feed_volumetric_concentration: float,
    ) -> dict[str, float]:
        """Расчёт параметров при заданном перепаде давления (Q = K·ΔP^0.472)."""
        K = self.compute_K(properties, feed_volumetric_concentration)
        feed_volumetric_flow_rate = K * pressure_drop**0.472
        return self.compute_results(
            properties, feed_volumetric_flow_rate,
            pressure_drop, feed_volumetric_concentration)

    def compute_K(
        self,
        properties: PhysicalProperties,
        feed_volumetric_concentration: float,
    ) -> float:
        """Коэффициент K в уравнении Q = K · ΔP^0.472."""
        hydrocyclone_diameter = self.design.diameters[HydrocycloneDiameters.C]
        feed_inlet_diameter = self.design.diameters[HydrocycloneDiameters.I]
        overflow_diameter = self.design.diameters[HydrocycloneDiameters.O]
        under_flow_diameter = self.design.diameters[HydrocycloneDiameters.U]
        L_minus_l = (self.design.lengths[HydrocycloneLengths.T] -
                     self.design.lengths[HydrocycloneLengths.V])

        return (0.184 * hydrocyclone_diameter**(-0.217) * feed_inlet_diameter**(1.231) *
                (overflow_diameter**2 + under_flow_diameter**2)**0.198 *
                L_minus_l**0.462 *
                properties.liquid_eos.viscosity()**(0.0566) *
                properties.liquid_eos.rhomass()**(-0.528) *
                np.exp(0.241 * feed_volumetric_concentration))

    def compute_reynolds_number(
        self,
        properties: PhysicalProperties,
        feed_volumetric_flow_rate: float,
    ) -> float:
        """Расчёт числа Рейнольдса Re."""
        hydrocyclone_diameter = self.design.diameters[HydrocycloneDiameters.C]
        return (4 * properties.liquid_eos.rhomass() * feed_volumetric_flow_rate /
                (np.pi * properties.liquid_eos.viscosity() * hydrocyclone_diameter))

    def compute_euler_number(
        self,
        feed_volumetric_concentration: float,
        Re: float,
    ) -> float:
        """Расчёт числа Эйлера Eu."""
        hydrocyclone_diameter = self.design.diameters[HydrocycloneDiameters.C]
        feed_inlet_diameter = self.design.diameters[HydrocycloneDiameters.I]
        overflow_diameter = self.design.diameters[HydrocycloneDiameters.O]
        underflow_diameter = self.design.diameters[HydrocycloneDiameters.U]
        L_minus_l = (self.design.lengths[HydrocycloneLengths.T] -
                     self.design.lengths[HydrocycloneLengths.V])

        return (43.5 * hydrocyclone_diameter**0.57 *
                (hydrocyclone_diameter / feed_inlet_diameter)**2.61 *
                (hydrocyclone_diameter /
                 (overflow_diameter**2 + underflow_diameter**2))**0.42 *
                (hydrocyclone_diameter / L_minus_l)**0.98 *
                Re**0.12 *
                np.exp(-0.51 * feed_volumetric_concentration))

    def compute_water_flow_ratio(self, Eu: float) -> float:
        """Расчёт соотношения потоков жидкости Rw."""
        hydrocyclone_diameter = self.design.diameters[HydrocycloneDiameters.C]
        overflow_diameter = self.design.diameters[HydrocycloneDiameters.O]
        under_flow_diameter = self.design.diameters[HydrocycloneDiameters.U]

        return (1.18 *
                (hydrocyclone_diameter / overflow_diameter)**5.97 *
                (under_flow_diameter / hydrocyclone_diameter)**3.10 *
                Eu**(-0.54))

    def compute_reduced_cut_size(
        self,
        properties: PhysicalProperties,
        feed_volumetric_flow_rate: float,
        pressure_drop: float,
        feed_volumetric_concentration: float,
        water_flow_ratio: float,
    ) -> float:
        """Расчёт приведённого отсечного размера d₅₀'."""
        hydrocyclone_diameter = self.design.diameters[HydrocycloneDiameters.C]
        overflow_diameter = self.design.diameters[HydrocycloneDiameters.O]
        L_minus_l = (self.design.lengths[HydrocycloneLengths.T] -
                     self.design.lengths[HydrocycloneLengths.V])

        rhos_minus_rho = properties.solid_density - properties.liquid_eos.rhomass()

        return (1.173 * hydrocyclone_diameter**0.64 /
                (overflow_diameter**0.475 * L_minus_l**0.665) *
                np.sqrt((properties.liquid_eos.viscosity() *
                         properties.liquid_eos.rhomass() *
                         feed_volumetric_flow_rate) /
                        (rhos_minus_rho * pressure_drop)) *
                np.log(1 / water_flow_ratio)**0.395 *
                np.exp(6.0 * feed_volumetric_concentration))

    def compute_results(
        self,
        properties: PhysicalProperties,
        feed_volumetric_flow_rate: float,
        pressure_drop: float,
        feed_volumetric_concentration: float,
    ) -> dict[str, float]:
        """Сборка всех выходных параметров гидроциклона."""
        Re = self.compute_reynolds_number(properties,
                                          feed_volumetric_flow_rate)
        Eu = self.compute_euler_number(feed_volumetric_concentration,
                                       Re)
        water_flow_ratio = self.compute_water_flow_ratio(Eu)
        reduced_cut_size = self.compute_reduced_cut_size(properties,
                                                         feed_volumetric_flow_rate,
                                                         pressure_drop,
                                                         feed_volumetric_concentration,
                                                         water_flow_ratio)

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

    def __init__(self, name: str, design: CycloneDesign) -> None:
        super().__init__(name, design)
        self.alpha = RIETEMA_ALPHA
        self.m = RIETEMA_M


class BradleyHydrocyclone(BaseHydrocyclone):
    """Гидроциклон по модели Bradley."""

    def __init__(self, name: str, design: CycloneDesign) -> None:
        super().__init__(name, design)
        self.alpha = BRADLEY_ALPHA
        self.m = BRADLEY_M


class DemcoHydrocyclone(BaseHydrocyclone):
    """Гидроциклон по модели Demco."""

    def __init__(self, name: str, design: CycloneDesign) -> None:
        super().__init__(name, design)
        self.alpha = DEMCO_ALPHA
        self.m = DEMCO_M
