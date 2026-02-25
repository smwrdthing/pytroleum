from dataclasses import dataclass

# NOTE гвоорящие имена полей датакласса


@dataclass
class GeometryParameters:
    """Геометрические параметры гидроциклона"""
    Dc: float  # Диаметр циклона, м
    Di: float  # Диаметр входа, м
    Do: float  # Диаметр вихреуловителя, м
    Du: float  # Диаметр нижнего слива, м
    L: float   # Длина гидроциклона, м
    l_vortex: float  # Выбег вихреуловителя, м
