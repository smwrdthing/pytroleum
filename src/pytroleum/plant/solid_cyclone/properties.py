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

    # NOTE здесь либо убрать атрибуты свойств и брать их напрямую из интерфейсов
    # NOTE уравнений состояния, либо нужно хорошо задокументировать эти атрибуты
    # NOTE
    # NOTE Лучше первое. Так нам, конечно, нужно знать что из себя представляют
    # NOTE интерфейсы уравнений состояния, но
    # NOTE 1) мы для этого их и разрабатывали
    # NOTE 2) атрибуты в этом классе дублируют то, что уже лежит в DEFAULT_LIQUID_EOS
    # NOTE 3) то, что лежит в liquid_viscosity и liquid_density не обновляется
    # NOTE    автоматически вместе с DEFAULT_LIQUID_EOS
    # NOTE
    # NOTE Если состояние обновится (считаем для разных температур, например), то придётся
    # NOTE вручную обновлять поля здесь
    # NOTE
    # NOTE Есть ещё опция сделать это методами или через property, но я думаю это
    # NOTE избыточно
