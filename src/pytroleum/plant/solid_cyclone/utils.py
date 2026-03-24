"""Console output utilities."""

_DIVIDER_LENGTH = 75
_MINOR_DIVIDER = '-' * _DIVIDER_LENGTH
_MAJOR_DIVIDER = '=' * _DIVIDER_LENGTH


def _minor_divider() -> None:
    print(_MINOR_DIVIDER)


def _major_divider() -> None:
    print(_MAJOR_DIVIDER)
