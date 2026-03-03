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
    solid_density: float                       # плотность твёрдой фазы (частиц), кг/м³
    liquid_eos: OptionalPhase = DEFAULT_LIQUID_EOS

    @property  # NOTE property избыточны
    def liquid_viscosity(self) -> float:
        if self.liquid_eos is None:
            # NOTE мы никогда не попадём в этот условный блок
            # NOTE (есть значение по умолчанию, оно не None) => он не нужен
            return DEFAULT_LIQUID_EOS.viscosity()
        return self.liquid_eos.viscosity()

    @property
    def liquid_density(self) -> float:
        if self.liquid_eos is None:
            # NOTE мы никогда не попадём в этот условный блок
            # NOTE (есть значение по умолчанию, оно не None) => он не нужен
            return DEFAULT_LIQUID_EOS.rhomass()
        return self.liquid_eos.rhomass()
