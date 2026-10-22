"""Correctors: small models that learn, from a person's corrections of a frozen policy, the nudge to
add to its plans (README.md). Every method takes the same inputs, targets and weights (data.py), is
evaluated the same way (evaluate.py) and is served by the same nudge adapter (inference.nudge).

A method is a class with
  fit(z, y, w, groups) -> self   rows' inputs (N, d), targets (N, 7), weights (N,), episodes (N,)
  predict(z) -> (N, 7)
  summary() -> dict              what it chose or how it trained, for the results and adapter.json
  export(outputs) -> (dict, dict)  its model description and arrays for the adapter
"""

from .mlp import MLP
from .ridge import Ridge

METHODS = {cls.name: cls for cls in (Ridge, MLP)}
