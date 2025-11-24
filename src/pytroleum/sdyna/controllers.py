import abc
import numpy as np
from typing import Iterable


class PropIntDiff:

    def __init__(
            self, P: float, I: float, D: float, F: float,  # noqa
            setpoint: float, saturation: Iterable[float] = (-np.inf, np.inf),
            ratelim: float = np.inf) -> None:

        # noqa is needed because flake8 does not like I as variable name :
        # ambiguous variable name 'I' (E741)
        # but application is clear, so we retain this

        # Assigning paramterers
        self.P, self.I, self.D = P, I, D
        self.F = F
        self.setpoint = setpoint
        self.saturation = saturation
        self.ratelim = ratelim

        # Entries for active operation attributes
        self.signal: float = 0
        self.history_signal: float = 0

        self.gain: float = 0
        self.history_gain: float = 0

        self.integral: float = 0
        self.history_integral: float = 0

        self.diff: float = 0
        self.history_diff: float = 0

        self.error: float = 0
        self.history_error: float = 0

        self.saturated: bool = False

    def check_saturation(self):

        upperlim, lowerlim = max(self.saturation), min(self.saturation)

        self.saturated = False
        if self.signal > upperlim:
            self.saturated = True
            self.signal = upperlim
        elif self.signal < lowerlim:
            self.saturated = True
            self.signal = lowerlim

    def check_rate(self, dt: float):

        rate = (self.signal-self.history_signal)/dt
        if abs(rate) > self.ratelim:
            self.signal = self.history_signal+np.sign(rate)*self.ratelim*dt

    def control(self, dt: float, probe: float, polarity: float = 1,
                norm_by: float = 1):

        # Previous step values become history
        self.history_gain = self.gain
        self.history_integral = self.integral
        self.history_diff = self.diff
        self.history_error = self.error
        self.history_signal = self.signal

        # Computing new error
        self.error = polarity*(self.setpoint-probe)/norm_by

        # Computing new signal
        self.gain = self.P*self.error
        self.integral = self.integral + self.I*self.error*dt
        self.diff = self.D*(
            self.error-self.history_error + self.F*self.history_diff)/(dt+self.F)
        self.signal = self.gain+self.integral+self.error

        # Performing saturation and rate limitataions checks and amends
        self.check_saturation()
        self.check_rate(dt)

        # At this point signal for the next step is generated and can be employed


class StartStop:

    def __init__(self, upperlim: float, lowerlim: float,
                 signal_max: float, signal_min: float) -> None:

        # Setting parameters
        self.signal_min = signal_min
        self.signal_max = signal_max

        self.upperlim = upperlim
        self.lowerlim = lowerlim

        # Entry to operational value of signal
        self.signal: float = 0

    def control(self, probe, invert=False):

        # Inverted on-off - on mode decreases probes, off increases
        # example : pump controlling liquid level
        if invert:
            # If probe overshoots upper limit we impose max signal
            if probe > self.upperlim:
                self.signal = self.signal_max
            # If probe is less than lower limit we minimize signal
            if probe < self.lowerlim:
                self.signal = self.signal_min

        # Non-inverted on-off - on mode increases probes, off decreases
        # example : compressor controlling pressure in vessel
        else:
            # If probe overshoots upper limit we impose minimal signal
            if probe > self.upperlim:
                self.signal = self.signal_min
            # If probe is less than lower limit we give max signal
            if probe < self.lowerlim:
                self.signal = self.signal_max
            # signal value in-between does not change, this is expected behavior

        # This should cover start/stop logic entirely
