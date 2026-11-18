"""One clock for the policy's observation.

Every input is given a content time, when it was true, on the node's clock, from its header stamp:

- A stamp near the receive time is on this clock already (the third-eye RealSense, the teleop's TCP) and
  is used as is: receive minus stamp is the input's age.
- A stamp far from it is on another clock: Baxter stamps its cameras, joint states and torques on its own
  clock, which the bridge passes through unchanged (13.9 years off in the 2026-11-05 bags). The clock gap is
  estimated as the smallest receive minus stamp seen over a window, the one-way lower bound, and the stamp
  moved by it. The residual age is then the transport jitter above that minimum, not the full delay; the
  minimum itself (about 10 ms through the bridge) cannot be seen from one side.

The observation time is the oldest camera's content time, as UMI aligns every stream to its slowest sensor,
and the state is the TCP and joints sampled nearest that time, so image and state describe one instant. The
plan's due times and anchor follow from it, which holds wherever the policy runs: the observation is stamped
here, on the robot PC, whether the network is local or a server on lamb.
"""

from collections import deque

FOREIGN = 1.0  # s: a stamp this far from the receive time is on another clock
WINDOW = 300  # samples the clock gap's minimum is kept over (3 s of joint states)


def seconds(stamp):
    """A builtin_interfaces Time as float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


class Stamps:
    """Content time of each named input on this clock, from its stamp and receive time."""

    def __init__(self, window=WINDOW):
        self._gaps = {}  # name -> recent receive - stamp of a foreign clock
        self.window = window

    def time(self, name, stamp, receive):
        gap = receive - stamp
        if abs(gap) < FOREIGN:
            return stamp
        gaps = self._gaps.setdefault(name, deque(maxlen=self.window))
        gaps.append(gap)
        return stamp + min(gaps)

    def foreign(self, name):
        """Whether `name` arrives on another clock, once seen."""
        return name in self._gaps


class Sampled:
    """A stream's recent values by content time: the value nearest a time, for the state at the
    observation's instant."""

    def __init__(self, maxlen=200):
        self._samples = deque(maxlen=maxlen)  # (time, value), time increasing

    def add(self, t, value):
        self._samples.append((t, value))

    def at(self, t):
        """The value nearest `t`, or None without samples."""
        if not self._samples:
            return None
        return min(self._samples, key=lambda s: abs(s[0] - t))[1]

    def __len__(self):
        return len(self._samples)
