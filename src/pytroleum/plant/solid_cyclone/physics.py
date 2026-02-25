from dataclasses import dataclass
from typing import Literal

# NOTE старые замечания из hydrocyclone.py, интегрировать tdyna для свойств
# NOTE жидкостей. Из свойств твёрдой фазы нужна только плотность, её можно передавать
# NOTE на месте там, где она нужна


@dataclass
class PhysicalParameters:
    """Физические параметры среды"""
    mu: float   # Динамическая вязкость жидкости, Па·с
    rho: float  # Плотность жидкости, кг/м³
    rhos: float  # Плотность твёрдой фазы, кг/м³


@dataclass
class OperatingParameters:
    """Рабочие параметры гидроциклона"""
    Q: float    # Входной расход, м³/с
    Cv: float   # Концентрация твёрдых частиц на входе


# Тип гидроциклона
HydrocycloneType = Literal['rietema', 'bradley', 'demco']
