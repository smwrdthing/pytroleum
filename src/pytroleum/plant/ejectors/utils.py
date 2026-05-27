# Физические константы
ATMOSPHERIC_PRESSURE = 101325    # Па

# Коэффициенты перевода единиц измерения
PA_TO_MPA = 1e6
KELVIN_TO_CELSIUS = 273.15

KCAL_TO_J = 4186.8                      # ккал → Дж
KCAL_PER_KMOL_TO_J_PER_MOL = 4.1868    # ккал/(кмоль·°С) → Дж/(моль·К)
KGS_S_M2_TO_PA_S = 9.80665             # кгс·с/м² → Па·с
KG_PER_KMOL_TO_KG_PER_MOL = 1e-3       # кг/кмоль → кг/моль

KG_PER_MOL_TO_G_PER_MOL = 1e3          # кг/моль → г/моль
J_TO_KJ = 1e3                           # Дж → кДж
M_TO_MM = 1e3                           # м → мм
M2_TO_MM2 = 1e6                         # м² → мм²
# ============================================================
# Функции форматирования консольного вывода результатов расчёта

_LABEL_WIDTH = 60
_DIVIDER_LENGTH = 75
_MINOR_DIVIDER = '-' * _DIVIDER_LENGTH
_MAJOR_DIVIDER = '=' * _DIVIDER_LENGTH


def _minor_divider() -> None:
    print(_MINOR_DIVIDER)


def _major_divider() -> None:
    print(_MAJOR_DIVIDER)


def _major_header(title: str) -> None:
    print(_MAJOR_DIVIDER)
    print(title.center(_DIVIDER_LENGTH))
    print(_MAJOR_DIVIDER)


def _minor_header(title: str) -> None:
    print(_MINOR_DIVIDER)
    print(title.center(_DIVIDER_LENGTH))
    print(_MINOR_DIVIDER)


def print_row(label: str, value: str, unit: str = '') -> None:
    """Вывод строки таблицы результатов с выравниванием по столбцам."""
    print(f"{label:<{_LABEL_WIDTH}} {value} {unit}".rstrip())
