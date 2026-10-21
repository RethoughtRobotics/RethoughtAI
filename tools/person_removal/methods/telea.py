"""OpenCV's Telea inpainting, frame by frame: fills the mask from its border inwards.

No model and no GPU, seconds per video. Fine for thin or small masks; over a large region such as a torso it
leaves a pale smear that flickers from frame to frame. Use it as the baseline the video models are compared to.
"""

import cv2
import numpy as np

RADIUS = 6  # px, neighbourhood Telea fills each pixel from

# One frame per call and no feathering: Telea changes only the masked pixels, so the result is exactly
# cv2.inpaint on each frame with its mask dilated by 3 px.
DEFAULTS = {'steps': 0, 'clip_len': 1, 'overlap': 0, 'dilation': 3, 'temporal': 0, 'feather': 0}


def load(models=None):
    def inpaint(frames: np.ndarray, masks: np.ndarray, steps: int) -> np.ndarray:
        out = []
        for rgb, m in zip(frames, masks):
            bgr = cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), m.astype(np.uint8) * 255, RADIUS, cv2.INPAINT_TELEA)
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return np.stack(out)

    return inpaint
