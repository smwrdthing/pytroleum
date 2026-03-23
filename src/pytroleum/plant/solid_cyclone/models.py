"""
Hydrocyclone calculation model.
"""
from __future__ import annotations
from abc import ABC
import numpy as np

from pytroleum.plant.solid_cyclone.inputs import PhysicalProperties, OperationConditions
from pytroleum.plant.solid_cyclone.geometry import (
    CycloneDesign,
    HydrocycloneDiameters,
    HydrocycloneLengths,
)

# ---------------------------------------------------------------------------
# Model constants (parameters for the reduced grade efficiency function)
# ---------------------------------------------------------------------------

RIETEMA_ALPHA = 4.23
RIETEMA_M = 2.45

BRADLEY_ALPHA = 5.10
BRADLEY_M = 3.12

DEMCO_ALPHA = 5.40
DEMCO_M = 3.30

# ---------------------------------------------------------------------------
# Base class and concrete models
# ---------------------------------------------------------------------------


class BaseHydrocyclone(ABC):
    """Abstract base class for hydrocyclone models (solid-liquid separation)."""

    def __init__(self, name: str, design: CycloneDesign) -> None:
        self.name = name
        self.design = design
        self.alpha: float
        self.m: float
        self.feed_volumetric_flow_rate: float = 0.0
        self.pressure_drop: float = 0.0
        self.water_flow_ratio: float = 0.0
        self.Re: float = 0.0
        self.Eu: float = 0.0
        self.reduced_cut_size: float = 0.0

    def calculate_from_flow_rate(
        self,
        properties: PhysicalProperties,
        feed_volumetric_flow_rate: float,
        feed_volumetric_concentration: float,
    ) -> None:
        """Calculate parameters for a given volumetric flow rate (ΔP = (Q/K)^(1/0.472)).
        """
        K = self.compute_K(properties, feed_volumetric_concentration)
        pressure_drop = (feed_volumetric_flow_rate / K) ** (1 / 0.472)
        self.compute_results(properties, OperationConditions(
            feed_volumetric_concentration=feed_volumetric_concentration,
            mode='Q',
            feed_volumetric_flow_rate=feed_volumetric_flow_rate,
            pressure_drop=pressure_drop,
        ))

    def calculate_from_pressure_drop(
        self,
        properties: PhysicalProperties,
        pressure_drop: float,
        feed_volumetric_concentration: float,
    ) -> None:
        """Calculate parameters for a given pressure drop (Q = K·ΔP^0.472)."""
        K = self.compute_K(properties, feed_volumetric_concentration)
        feed_volumetric_flow_rate = K * pressure_drop**0.472
        self.compute_results(properties, OperationConditions(
            feed_volumetric_concentration=feed_volumetric_concentration,
            mode='delta_p',
            feed_volumetric_flow_rate=feed_volumetric_flow_rate,
            pressure_drop=pressure_drop,
        ))

    def compute_K(
        self,
        properties: PhysicalProperties,
        feed_volumetric_concentration: float,
    ) -> float:
        """Coefficient K in the equation Q = K · ΔP^0.472."""
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
        """Calculate Reynolds number Re."""
        hydrocyclone_diameter = self.design.diameters[HydrocycloneDiameters.C]
        return (4 * properties.liquid_eos.rhomass() * feed_volumetric_flow_rate /
                (np.pi * properties.liquid_eos.viscosity() * hydrocyclone_diameter))

    def compute_euler_number(
        self,
        feed_volumetric_concentration: float,
        Re: float,
    ) -> float:
        """Calculate Euler number Eu."""
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
        """Calculate liquid flow ratio Rw."""
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
        conditions: OperationConditions,
        water_flow_ratio: float,
    ) -> float:
        """Calculate reduced cut size d₅₀'."""
        hydrocyclone_diameter = self.design.diameters[HydrocycloneDiameters.C]
        overflow_diameter = self.design.diameters[HydrocycloneDiameters.O]
        L_minus_l = (self.design.lengths[HydrocycloneLengths.T] -
                     self.design.lengths[HydrocycloneLengths.V])

        rhos_minus_rho = properties.solid_density - properties.liquid_eos.rhomass()

        return (1.173 * hydrocyclone_diameter**0.64 /
                (overflow_diameter**0.475 * L_minus_l**0.665) *
                np.sqrt((properties.liquid_eos.viscosity() *
                         properties.liquid_eos.rhomass() *
                         conditions.feed_volumetric_flow_rate) /
                        (rhos_minus_rho * conditions.pressure_drop)) *
                np.log(1 / water_flow_ratio)**0.395 *
                np.exp(6.0 * conditions.feed_volumetric_concentration))

    def compute_results(
        self,
        properties: PhysicalProperties,
        conditions: OperationConditions,
    ) -> None:
        """Compute all hydrocyclone output parameters and store them as attributes."""
        self.feed_volumetric_flow_rate = conditions.feed_volumetric_flow_rate
        self.pressure_drop = conditions.pressure_drop
        self.Re = self.compute_reynolds_number(
            properties, conditions.feed_volumetric_flow_rate)
        self.Eu = self.compute_euler_number(
            conditions.feed_volumetric_concentration, self.Re)
        self.water_flow_ratio = self.compute_water_flow_ratio(self.Eu)
        self.reduced_cut_size = self.compute_reduced_cut_size(
            properties, conditions, self.water_flow_ratio)


class RietemaHydrocyclone(BaseHydrocyclone):
    """Hydrocyclone using the Rietema model."""

    def __init__(self, name: str, design: CycloneDesign) -> None:
        super().__init__(name, design)
        self.alpha = RIETEMA_ALPHA
        self.m = RIETEMA_M


class BradleyHydrocyclone(BaseHydrocyclone):
    """Hydrocyclone using the Bradley model."""

    def __init__(self, name: str, design: CycloneDesign) -> None:
        super().__init__(name, design)
        self.alpha = BRADLEY_ALPHA
        self.m = BRADLEY_M


class DemcoHydrocyclone(BaseHydrocyclone):
    """Hydrocyclone using the Demco model."""

    def __init__(self, name: str, design: CycloneDesign) -> None:
        super().__init__(name, design)
        self.alpha = DEMCO_ALPHA
        self.m = DEMCO_M
