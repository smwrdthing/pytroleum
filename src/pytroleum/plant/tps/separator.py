from pytroleum.plant.tps.utils import (_major_header, _minor_divider,
                                       _TO_MM, PERCENT, SECONDS_PER_MINUTE)
from pytroleum.plant.tps.inputs import SeparatorParameters, OperationConditions
import numpy as np


class Separator:
    """Расчёт сепаратора"""

    def __init__(self, sepparam: SeparatorParameters,
                 conditions: OperationConditions):
        self.sepparam = sepparam
        self.conditions = conditions

    # --- Сепаратор ---

    def volume_separator(self) -> float:
        """Номинальный объём сепаратора, м³"""
        return (np.pi * self.sepparam.inner_diameter ** 2 / 4 *
                self.sepparam.length_cylindrical_part +
                2 * self.sepparam.volume_ell_head)

    def residence_time(self) -> float:
        """Время пребывания жидкости в сепараторе, с"""
        return (self.volume_separator() * self.sepparam.fill_coeff /
                self.conditions.flow_liquid)

    def capacity(self):
        """Максимальная производительность аппарата по жидкости с
        учетом коэффициента заполнения"""
        return (self.volume_separator()*self.sepparam.fill_coeff /
                self.residence_time())

    # --- Первая секция (Н+В) ---

    def volume_first_section(self) -> float:
        """Объём первой секции, м³"""
        return (np.pi * self.sepparam.inner_diameter ** 2 / 4 *
                self.sepparam.length_first_section)

    def residence_time_first_section(self) -> float:
        """Время пребывания жидкости в первой секции, с"""
        return (self.volume_first_section() * self.sepparam.fill_coeff_first_section /
                self.conditions.flow_liquid)

    def capacity_first_section(self) -> float:
        """Пропускная способность первой секции, м³/с"""
        return (self.volume_first_section() * self.sepparam.fill_coeff_first_section /
                self.residence_time_first_section())

    # --- Сборник нефти после перегородки ---

    def volume_after_wall(self) -> float:
        """Объём сборника нефти после перегородки, м³"""
        return (np.pi * self.sepparam.inner_diameter ** 2 / 4 *
                self.sepparam.length_section_after_wall +
                self.sepparam.volume_ell_head)

    def residence_time_after_wall(self) -> float:
        """Время пребывания нефти в сборнике после перегородки, с"""
        return self.residence_time() - self.residence_time_first_section()

    def capacity_after_wall(self) -> float:
        """Пропускная способность сборника нефти после перегородки, м³/с"""
        return (self.volume_after_wall() * self.sepparam.fill_coeff_after_wall /
                self.residence_time_after_wall())

    # ---Скорость движения жидкой фазы и газовой фазы в сечении сепаратора ---


if __name__ == "__main__":
    from pytroleum.plant.tps.utils import SECONDS_PER_DAY

    sepparam = SeparatorParameters(
        inner_diameter=2.0,
        length_cylindrical_part=9.5,
        fill_coeff=0.858,
        fill_coeff_after_wall=0.50,
        fill_coeff_first_section=0.2,
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

    print(f"Внутренний диаметр сепаратора: "
          f"{sepparam.inner_diameter * _TO_MM:.0f} мм")
    print(f"Длина цилиндрической части сепаратора: "
          f"{sepparam.length_cylindrical_part:.1f} м")
    print(f"Объём эллиптического днища: "
          f"{sepparam.volume_ell_head:.3f} м³")
    print(f"Объёмный расход жидкости: "
          f"{conditions.flow_liquid * SECONDS_PER_DAY:.1f} м³/сут")
    print(f"Коэффициент заполнения: "
          f"{sepparam.fill_coeff * PERCENT:.1f} %")
    print(f"Номинальный объём сепаратора: "
          f"{sep.volume_separator():.3f} м³")
    print(f"Время пребывания жидкости: "
          f"{sep.residence_time() / SECONDS_PER_MINUTE:.2f} мин")
    print(f"Пропускная способность: "
          f"{sep.capacity() * SECONDS_PER_DAY:.2f} м³/сут")

    _minor_divider()
    print("ПЕРВАЯ СЕКЦИЯ (НЕФТЬ + ВОДА)")
    _minor_divider()
    print(f"Длина первой секции: "
          f"{sepparam.length_first_section:.1f} м")
    print(f"Коэффициент заполнения: "
          f"{sepparam.fill_coeff_first_section * PERCENT:.1f} %")
    print(f"Объём первой секции: "
          f"{sep.volume_first_section():.3f} м³")
    print(f"Время пребывания (Н+В): "
          f"{sep.residence_time_first_section() / SECONDS_PER_MINUTE:.3f} мин")
    print(f"Пропускная способность: "
          f"{sep.capacity_first_section() * SECONDS_PER_DAY:.2f} м³/сут")

    _minor_divider()
    print("СБОРНИК НЕФТИ (ПОСЛЕ ПЕРЕГОРОДКИ)")
    _minor_divider()
    print(f"Длина секции после перегородки: "
          f"{sepparam.length_section_after_wall:.1f} м")
    print(f"Коэффициент заполнения: "
          f"{sepparam.fill_coeff_after_wall * PERCENT:.1f} %")
    print(f"Объём секции после перегородки: "
          f"{sep.volume_after_wall():.3f} м³")
    print(f"Время пребывания: "
          f"{sep.residence_time_after_wall() / SECONDS_PER_MINUTE:.3f} мин")
    print(f"Пропускная способность: "
          f"{sep.capacity_after_wall() * SECONDS_PER_DAY:.3f} м³/сут")

    _minor_divider()
