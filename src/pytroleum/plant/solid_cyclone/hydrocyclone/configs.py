"""
Функции для конфигураций гидроциклонов.
"""
from .geometry import GeometryParameters
from .models import (
    BaseHydrocyclone,
    RietemaHydrocyclone,
    BradleyHydrocyclone,
    DemcoHydrocyclone,
)


def build_rietema_config(hydrocyclone_diameter: float) -> RietemaHydrocyclone:
    """Возвращает стандартную конфигурацию гидроциклона Rietema."""
    geometry = GeometryParameters.from_named(
        hydrocyclone_diameter=hydrocyclone_diameter,
        feed_inlet_diameter=hydrocyclone_diameter * 0.20,
        overflow_diameter=hydrocyclone_diameter * 0.25,
        underflow_diameter=hydrocyclone_diameter * 0.15,
        hydrocyclone_length=hydrocyclone_diameter * 4.50,
        vortex_finder_length=hydrocyclone_diameter * 0.40,
        angle=15.0,
    )
    return RietemaHydrocyclone('Rietema', geometry)


def build_bradley_config(hydrocyclone_diameter: float) -> BradleyHydrocyclone:
    """Возвращает стандартную конфигурацию гидроциклона Bradley."""
    geometry = GeometryParameters.from_named(
        hydrocyclone_diameter=hydrocyclone_diameter,
        feed_inlet_diameter=hydrocyclone_diameter * 0.16,
        overflow_diameter=hydrocyclone_diameter * 0.22,
        underflow_diameter=hydrocyclone_diameter * 0.12,
        hydrocyclone_length=hydrocyclone_diameter * 5.50,
        vortex_finder_length=hydrocyclone_diameter * 0.45,
        angle=12.0,
    )
    return BradleyHydrocyclone('Bradley', geometry)


def build_demco_config(hydrocyclone_diameter: float) -> DemcoHydrocyclone:
    """Возвращает стандартную конфигурацию гидроциклона Demco."""
    geometry = GeometryParameters.from_named(
        hydrocyclone_diameter=hydrocyclone_diameter,
        feed_inlet_diameter=hydrocyclone_diameter * 0.25,
        overflow_diameter=hydrocyclone_diameter * 0.30,
        underflow_diameter=hydrocyclone_diameter * 0.20,
        hydrocyclone_length=hydrocyclone_diameter * 5.00,
        vortex_finder_length=hydrocyclone_diameter * 0.50,
        angle=18.0,
    )
    return DemcoHydrocyclone('Demco', geometry)


def build_standard_configs(hydrocyclone_diameter: float) -> list[BaseHydrocyclone]:
    """
    Возвращает список стандартных конфигураций [Rietema, Bradley, Demco]
    с пропорциями в допустимых диапазонах.
    """
    hydrocyclones: list[BaseHydrocyclone] = [
        build_rietema_config(hydrocyclone_diameter),
        build_bradley_config(hydrocyclone_diameter),
        build_demco_config(hydrocyclone_diameter),
    ]

    print("\n" + "=" * 60)
    print(f"ПРОВЕРКА ГЕОМЕТРИЧЕСКИХ ПРОПОРЦИЙ "
          f"(Dc = {hydrocyclone_diameter * 1000:.1f} мм)")
    print("=" * 60)
    for h in hydrocyclones:
        h.print_proportions()
    return hydrocyclones
