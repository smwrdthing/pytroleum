import abc
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

        self.polarity = polarity
        self.norm_by = norm_by

    def check_saturation(self):
        """Checks if signal gets saturated and enforces saturation boundaries"""
        upperlim, lowerlim = max(self.saturation), min(self.saturation)

        self.saturated = False
        if self.signal > upperlim:
            self.saturated = True
            self.signal = upperlim
        elif self.signal < lowerlim:
            self.saturated = True
            self.signal = lowerlim

    def check_rate(self, time_step: float):
        """Checks if rate of change of signal exceeds limit, amends value to
        comply with limit if it does."""
        rate = (self.signal-self.history_signal)/time_step
        if abs(rate) > self.ratelim:
            self.signal = (
                self.history_signal+np.sign(rate)*self.ratelim*time_step)

    def control(self, time_step: float, probe: float):
        """Generates contorl signal of PID controller from given time step and
        probe value"""
        # Previous step values become history
        self.history_gain = self.gain
        self.history_integral = self.integral
        self.history_diff = self.diff
        self.history_error = self.error
        self.history_signal = self.signal

        # Computing new error
        self.error = self.polarity*(self.setpoint-probe)/self.norm_by

        # Computing new signal
        self.gain = self.gain_coeff*self.error
        self.integral = self.integral + self.integral_coeff * (
            self.error*time_step*(not self.saturated))  # anti-windup measure
        self.diff = self.history_diff + (
            self.derivative_coeff*(self.error-self.history_error) -
            self.history_diff*time_step)/self.filter
        self.signal = self.gain+self.integral+self.diff

        # Performing saturation and rate limitataions checks and amends
        self.check_saturation()
        self.check_rate(time_step)

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
        """Generates control signal (either max or min value).
        Logic can be inverted with flag"""
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


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Testing area for PID
    PID = PropIntDiff(
        gain_coeff=0.01, integral_coeff=0.01, derivative_coeff=0.95,
        filter=100, setpoint=1,
        # saturation=[-0.001, 0.001]
    )
    probe = 0.
    dt = 0.01

    PID.error = PID.setpoint-probe
    PID.control(dt, probe)

    probes = [probe]
    gains = [PID.gain]
    integrals = [PID.integral]
    diffs = [PID.diff]
    signals = [PID.signal]
    errors = [PID.error]
    time = [0.]
    counter = 0
    T = 20

    while time[-1] < T:
        PID.control(0.01, probe)

        probe += PID.signal
        probes.append(probe)

        gains.append(PID.gain)
        integrals.append(PID.integral)
        diffs.append(PID.diff)
        signals.append(PID.signal)
        errors.append(PID.error)

        time.append(time[-1]+dt)

    fig, ax = plt.subplots()
    ax.set_title('Probe')
    ax.set_xlabel('time [-]')
    ax.set_ylabel('probe [-]')
    ax.plot(time, probes)
    ax.grid(True)

    fig, ax = plt.subplots()
    ax.set_title('Signal')
    ax.set_xlabel('time [-]')
    ax.set_ylabel('signal [-]')
    ax.plot(time, signals)
    ax.plot(time, gains, 'C1')
    ax.plot(time, integrals, 'C2')
    ax.plot(time, diffs, 'C3')
    ax.grid(True)
    ax.legend([
        "overall",
        "gain",
        "integral",
        "diff"
    ])

    fig, ax = plt.subplots()
    ax.set_title('Error')
    ax.set_xlabel('time [-]')
    ax.set_ylabel('error [-]')
    ax.plot(time, errors)
    ax.grid(True)

    # Looks like it works
