from dataclasses import dataclass
from typing import Literal

from numpy.typing import NDArray

from pytroleum.tdyna import eos
import CoolProp.constants as CoolConst


DEFAULT_TEMPERATURE = 20 + 273.15   # K
DEFAULT_PRESSURE = 101_325          # Pa

DEFAULT_LIQUID_EOS = eos.AbstractState("HEOS", "water")
DEFAULT_LIQUID_EOS.update(CoolConst.PT_INPUTS,
                          DEFAULT_PRESSURE,
                          DEFAULT_TEMPERATURE)

type OptionalPhase = eos.AbstractState | eos.AbstractStateImitator


# region PhysicalProperties
@dataclass
class PhysicalProperties:
    solid_density: float
    liquid_eos: OptionalPhase = DEFAULT_LIQUID_EOS


# region OperatingConditions
@dataclass
class OperationConditions:
    feed_volumetric_concentration: float
    mode: Literal['Q', 'delta_p']
    feed_volumetric_flow_rate: float = 0.0
    pressure_drop: float = 0.0


# region SizeDistribution
@dataclass
class SizeDistribution:
    particle_diameters: NDArray
    k: float
    n: float
