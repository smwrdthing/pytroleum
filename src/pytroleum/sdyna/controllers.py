import abc
import numpy as np


class PropIntDiff:

    def __init__(self, P, I, D, F, setpoint, saturation, ratelim) -> None:
        self.P, self.I, self.D = P, I, D
        self.F = F
        self.setpoint = setpoint
        self.saturation = saturation
        self.ratelim = ratelim


class StartStop:

    def __init__(self, upperlim, bottomlim) -> None:
        self.upperlim = upperlim
        self.bottomlim = bottomlim
