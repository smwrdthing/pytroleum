# Коэффициенты перевода единиц измерения
SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60
KG_PER_TON = 1000
KG_S_TO_T_H = 3.6
PA_TO_MPA = 1e6
PERCENT = 100
_TO_MM = 1000
_TO_M = 1e-3
_TO_MICRON = 1_000_000
KELVIN_TO_CELSIUS = 273


# ============================================================
# Функции форматирования консольного вывода результатов расчёта


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
