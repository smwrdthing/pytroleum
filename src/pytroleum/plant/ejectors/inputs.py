from dataclasses import dataclass


@dataclass
class BaseMediumData:
    """Базовый класс с общими полями"""
    mass_flow: float  # кг/c
    temperature: float  # K
    inlet_pressure: float  # Па
    enthalpy: float  # Дж/кг
    entropy: float  # Дж/(кг·К)
    specific_volume: float  # м3/кг
    density: float  # кг/м3
    dynamic_viscosity: float  # Па·с
    inlet_diameter: float  # м
    molecular_mass: float  # кг/моль
    heat_capacity: float  # Дж/(моль·К)

    # NOTE вместо этого класса хорошо впишется интерфейс уравнения состояния из CoolProp


@dataclass
class ActiveMediumData(BaseMediumData):
    """Входные данные эжектирующей среды"""
    # NOTE я думаю мы можем использовать один "шаблон" (один класс) для работы со
    # NOTE свойствами как для активной, так и для пассивной среды
    pass


@dataclass
class PassiveMediumData(BaseMediumData):
    """Входные данные эжектируемой среды"""
    pass


@dataclass
class CommonParams:
    """Общие входные параметры"""
    num_stages: int
    outlet_pressure: float  # Па
    outlet_diameter: float  # м

    # NOTE Это больше выглядит как какой-нибудь класс исходных данных,
    # NOTE что-то типа Requirements
    # NOTE
    # NOTE
    # NOTE В целом потом можно написать что-то вроде
    # NOTE def design(
    # NOTE     req:Requirements,
    # NOTE     active_phase_eos: EquationOfState,
    # NOTE     passive_phase_eos: EquationOfState) -> Ejector
    # NOTE а может даже сделать уравнения состояния частью req
