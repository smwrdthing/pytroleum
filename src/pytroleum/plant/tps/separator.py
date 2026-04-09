from pytroleum.plant.tps.utils import (_major_header, _minor_divider,
                                       _TO_MM, PERCENT, SECONDS_PER_MINUTE)
from pytroleum.plant.tps.inputs import SeparatorParameters, OperationConditions
import numpy as np


class Separator:
    """Расчет пропускной способности сепаратора по жидкости"""

    def __init__(self, sepparam: SeparatorParameters,
                 conditions: OperationConditions):
        self.sepparam = sepparam
        self.conditions = conditions

    def volume_separator(self):
        """Номинальный объем сепаратора"""
        return (np.pi*self.sepparam.inner_diameter**2/4 *
                self.sepparam.length_cylindrical_part+2*self.sepparam.volume_ell_head)

    def residence_time(self):
        """Время прибывания"""
        return (self.volume_separator()*self.sepparam.fill_coefficient /
                self.conditions.flow_liquid)

    def capacity(self):
        """Максимальная производительность аппарата по жидкости с
        учетом коэффициента заполнения"""
        return (self.volume_separator()*self.sepparam.fill_coefficient /
                self.residence_time())

    """Первая секция"""

    def volume_first_section(self):
        """Объем первой секции"""
        return np.pi*self.sepparam.inner_diameter**2/4*self.sepparam.length_first_section

    def residence_time_first_section(self):
        """"Время прибывания в первой секции"""
        return (self.volume_first_section()*self.sepparam.fill_coefficient /
                self.conditions.flow_liquid)

    def capacity_first_section(self):
        """Пропускная способность первой секции"""
        return (self.volume_first_section()*self.sepparam.fill_coefficient /
                self.residence_time_first_section())

    """Секция после перегородки """

    def residence_time_after_wall(self):
        """Время прибывания после перегородки"""
        return self.residence_time()-self.residence_time_first_section()

    def volume_after_wall(self):
        """Объем секции после перегородки"""
        return (np.pi * self.sepparam.inner_diameter**2/4 *
                self.sepparam.length_section_after_wall+self.sepparam.volume_ell_head)

    def capacity_after_wall(self):
        return (self.volume_after_wall()*self.sepparam.fill_coefficient /
                self.residence_time_after_wall())


if __name__ == "__main__":
    from pytroleum.plant.tps.utils import SECONDS_PER_DAY

    sepparam = SeparatorParameters(
        inner_diameter=2.0,
        length_cylindrical_part=9.5,
        fill_coefficient=0.858,
        volume_ell_head=1.294,
        length_first_section=8.2,
        length_section_after_wall=1.3,
    )
    conditions = OperationConditions(
        pressure_work=4e6,
        temperature_work=353,
        flow_gas_norm=300000 / SECONDS_PER_DAY,
        flow_liquid=500 / SECONDS_PER_DAY,
    )

    sep = Separator(sepparam=sepparam, conditions=conditions)

    _major_header("РАСЧЁТ ПРОПУСКНОЙ СПОСОБНОСТИ СЕПАРАТОРА ПО ЖИДКОСТИ")

    _minor_divider()
    print(f"Внутренний диаметр:                      "
          f"{sepparam.inner_diameter * _TO_MM:.0f} мм")
    print(f"Длина цилиндрической части:              "
          f"{sepparam.length_cylindrical_part:.1f} м")
    print(f"Объём эллиптического днища:              "
          f"{sepparam.volume_ell_head:.3f} м³")
    print(f"Коэффициент заполнения:                  "
          f"{sepparam.fill_coefficient * PERCENT:.1f} %")
    print(f"Объёмный расход жидкости:                "
          f"{conditions.flow_liquid * SECONDS_PER_DAY:.1f} м³/сут")

    _minor_divider()
    print(f"Номинальный объём сепаратора:            "
          f"{sep.volume_separator():.3f} м³")
    print(f"Время пребывания жидкости:               "
          f"{sep.residence_time() / SECONDS_PER_MINUTE:.2f} мин")
    print(f"Пропускная способность сепаратора:       "
          f"{sep.capacity() * SECONDS_PER_DAY:.2f} м³/сут")

    _minor_divider()
    print(f"Объём первой секции (Н+В):               "
          f"{sep.volume_first_section():.3f} м³")
    print(f"Время пребывания в первой секции:        "
          f"{sep.residence_time_first_section() / SECONDS_PER_MINUTE:.3f} мин")
    print(f"Пропускная способность первой секции:    "
          f"{sep.capacity_first_section() * SECONDS_PER_DAY:.2f} м³/сут")

    _minor_divider()
    print(f"Объём сборника нефти:                    "
          f"{sep.volume_after_wall():.3f} м³")
    print(f"Время пребывания нефти в сборнике:       "
          f"{sep.residence_time_after_wall() / SECONDS_PER_MINUTE:.3f} мин")
    print(f"Пропускная способность сборника нефти:   "
          f"{sep.capacity_after_wall() * SECONDS_PER_DAY:.3f} м³/сут")
