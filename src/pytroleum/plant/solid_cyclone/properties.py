"""
Физические свойства рабочей среды гидроциклона.
"""
from dataclasses import dataclass

from pytroleum.tdyna import eos
import CoolProp.constants as CoolConst

DEFAULT_TEMPERATURE = 20 + 273.15   # К
DEFAULT_PRESSURE = 101_325          # Па

DEFAULT_LIQUID_EOS = eos.AbstractState("HEOS", "water")
DEFAULT_LIQUID_EOS.update(CoolConst.PT_INPUTS,
                          DEFAULT_PRESSURE,
                          DEFAULT_TEMPERATURE)

type OptionalPhase = eos.AbstractState | eos.AbstractStateImitator | None


@dataclass
class PhysicalProperties:
    """Физические свойства среды."""
    solid_density: float
    liquid_eos: OptionalPhase = DEFAULT_LIQUID_EOS

    liquid_viscosity: float = DEFAULT_LIQUID_EOS.viscosity()
    liquid_density: float = DEFAULT_LIQUID_EOS.rhomass()
