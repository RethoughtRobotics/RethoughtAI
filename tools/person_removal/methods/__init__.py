"""Removal methods. Each module has the same two things:

  DEFAULTS  its settings: steps, clip_len (frames per model call), overlap (frames shared by consecutive
            clips, cross-faded), dilation and temporal (mask clean-up, see masks.region), feather (see
            masks.alpha)
  load(models), returning inpaint(frames, masks, steps):
            frames (T, H, W, 3) uint8 RGB, masks (T, H, W) bool, returns (T, H, W, 3) uint8; `models` is the
            directory fetch.py fills with the method's code and weights

A new method is one more module here and one line in METHODS.
"""

import importlib

METHODS = ('telea', 'minimax', 'effecterase')


def get(name: str):
    """The method's module."""
    if name not in METHODS:
        raise ValueError(f'unknown method {name!r}; one of {", ".join(METHODS)}')
    return importlib.import_module(f'methods.{name}')
