from dataclasses import dataclass

THERMAL_EQUIVALENT_OF_WORK = 0.982   # Дж/Дж

# Все исходные данные взяты с ТЗ или же с Aspen Hysys


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


@dataclass
class EjectingData(BaseMediumData):
    """Входные данные эжектирующей среды"""
    pass


@dataclass
class EjectedData(BaseMediumData):
    """Входные данные эжектируемой среды"""
    pass


@dataclass
class CommonParams:
    """Общие входные параметры"""
    num_stages: int
    outlet_pressure: float  # Па
    outlet_diameter: float  # м

# После схождения результатов с экселем,
# можно будет применить, что написано в tdyna
