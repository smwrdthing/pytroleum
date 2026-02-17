import numpy as np
from typing import Iterable


class PropIntDiff:

    def __init__(
            self,
            gain_coeff: float,
            integral_coeff: float,
            derivative_coeff: float,
            filter: float,
            setpoint: float, saturation: Iterable[float] = (-np.inf, np.inf),
            ratelim: float = np.inf,
            polarity: float = 1, norm_by: float = 1) -> None:

        self.gain_coeff = gain_coeff
        self.integral_coeff = integral_coeff
        self.derivative_coeff = derivative_coeff
        self.filter = filter
        self.setpoint = setpoint
        self.saturation = saturation
        self.ratelim = ratelim

        self._signal: float = 0
        self._history_signal: float = 0

        self._gain: float = 0
        self._history_gain: float = 0

        self._integral: float = 0
        self._history_integral: float = 0

        self._diff: float = 0
        self._history_diff: float = 0

        self._error: float = 0
        self._history_error: float = 0

        self._saturated: bool = False

        self.polarity = polarity
        self.norm_by = norm_by

    def check_saturation(self):
        """Checks if signal gets saturated and enforces saturation boundaries"""
        upperlim, lowerlim = max(self.saturation), min(self.saturation)

        self._saturated = False
        if self._signal > upperlim:
            self._saturated = True
            self._signal = upperlim
        elif self._signal < lowerlim:
            self._saturated = True
            self._signal = lowerlim

    def check_rate(self, time_step: float):
        """Checks if rate of change of signal exceeds limit, amends value to
        comply with limit if it does."""
        rate = (self._signal-self._history_signal)/time_step
        if abs(rate) > self.ratelim:
            self._signal = (
                self._history_signal+np.sign(rate)*self.ratelim*time_step)

    def control(self, time_step: float, probe: float):
        """Generates contorl signal of PID controller from given time step and
        probe value"""
        # Previous step values become history
        self._history_gain = self._gain
        self._history_integral = self._integral
        self._history_diff = self._diff
        self._history_error = self._error
        self._history_signal = self._signal

        # Computing new error
        self._error = self.polarity*(self.setpoint-probe)/self.norm_by

        # Computing new signal
        self._gain = self.gain_coeff*self._error
        self._integral = self._integral + self.integral_coeff * (
            self._error*time_step*(not self._saturated))  # anti-windup measure
        self._diff = self._history_diff + (
            self.derivative_coeff*(self._error-self._history_error) -
            self._history_diff*time_step)/self.filter
        self._signal = self._gain+self._integral+self._diff

        # Performing saturation and rate limitataions checks and amends
        self.check_saturation()
        self.check_rate(time_step)


class StartStop:

    def __init__(self, upperlim: float, lowerlim: float,
                 signal_max: float, signal_min: float) -> None:
        self.signal_min = signal_min
        self.signal_max = signal_max

        self.upperlim = upperlim
        self.lowerlim = lowerlim

        self._signal: float = 0.0

    def control(self, probe, invert=False):
        """Generates control signal (either max or min value).
        Logic can be inverted with flag"""
        # Inverted on-off - on mode decreases probes, off increases
        # example : pump controlling liquid level
        if invert:
            if probe > self.upperlim:
                self._signal = self.signal_max
            if probe < self.lowerlim:
                self._signal = self.signal_min
        # Non-inverted on-off - on mode increases probes, off decreases
        # example : compressor controlling pressure in vessel
        else:
            if probe > self.upperlim:
                self._signal = self.signal_min
            if probe < self.lowerlim:
                self._signal = self.signal_max
