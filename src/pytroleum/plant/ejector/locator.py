from __future__ import annotations
from enum import IntEnum, auto


import numpy as np


# Integers are assigned manually and not via auto() because we also want to have
# shortcuts, this messes up auto() mechanism. For convenient container creation we
# still use auto() to ensure appropriate SIZE in each enumeration


class Stream(IntEnum):

    """Enumeration to hold access indices that specify certain stream in the ejector"""

    P = PRIMPARY = 0
    S = SECONDARY = 1
    M = MIXED = 2
    SIZE = auto()


class Place(IntEnum):

    """Enumeration to hold access indices that specify certain location in the ejector"""

    I = INLET = 0
    O = OUTLET = 1
    T = THROAT = 2
    N = NOZZLE = 3
    M = MIXING = 4
    S = SHOCK = 5
    C = CONST = 6
    SIZE = auto()


# To use in situations when specific index is irrelevant
ANY = -1
# To use in situations when everything must be affected
ALL = slice(None)
