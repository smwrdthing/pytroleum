import numpy as np
from pytroleum.plant.tps.inputs import (OperationConditions,
                                        CoalescerPacking,
                                        GeometryCyclone,
                                        VAPOR)
from pytroleum.plant.tps.separator import compute_settling_velocity, Separator


class Coalescer:
    def __init__(self, packing: CoalescerPacking) -> None:
        self.packing = packing

    def droplet_settling_time(self, plate_spacing: float, drop_diameter: float,
                              continuous_phase_density: float,
                              continuous_phase_viscosity: float,
                              dispersed_phase_density: float) -> float:
        """Время осаждения/всплытия капли в зазоре между пластинами, с.

        t_к = h / (|v_ст| * cos(α))

        где h — расстояние между пластинами, v_ст — скорость Стокса,
        α — угол наклона пластин.
        """
        velocity = compute_settling_velocity(
            drop_diameter, continuous_phase_density,
            continuous_phase_viscosity, dispersed_phase_density,
        )
        return (plate_spacing / (abs(velocity) *
                                 np.cos(np.radians(self.packing.angle))))


def required_length_for(phase_velocity: float, settling_time: float) -> float:
    """Длина канала коалесцера, м.

    L_кан = u_ф * t_к

    где u_ф — скорость фазы в канале, t_к — время осаждения/всплытия капли.
    """
    return phase_velocity * settling_time


class Cyclone:

    def __init__(self, conditions: OperationConditions,
                 geometry: GeometryCyclone):
        self.conditions = conditions
        self.geometry = geometry

    def vapor_velocity(self, number_of_cyclones: int) -> float:
        """Скорость газа в спиральном канале, м/с.

        uг_сп = Qг_ру / (n * F_кан)

        где Qг_ру — расход газа при р.у., n — число циклонов,
        F_кан — площадь сечения спирального канала одного циклона.
        """
        return self.conditions.vol_flow_rate[VAPOR] / (
            number_of_cyclones * self.geometry.area_spiral_channel)
