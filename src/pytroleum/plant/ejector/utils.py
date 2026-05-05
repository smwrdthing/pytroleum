# Коэффициенты перевода единиц измерения
PA_TO_MPA = 1e6
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


_LABEL_WIDTH = 60


def print_row(label: str, value: str, unit: str = '') -> None:
    """Вывод строки таблицы результатов с выравниванием по столбцам."""
    print(f"{label:<{_LABEL_WIDTH}} {value} {unit}".rstrip())
