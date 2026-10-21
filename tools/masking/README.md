# Mask baselines

Earlier person-masking tools, kept for comparison:

- `evaluate.py` runs SAM3, DeepLab or Mask R-CNN on sampled frames of a dataset and writes an HTML review of
  their masks side by side.
- `mask_dataset.py` writes a dataset variant with the person's pixels filled with a constant colour
  (DeepLab or Mask R-CNN masks).

To remove people from a dataset's videos (SAM3 masks, then Telea, MiniMax-Remover or EffectErase), use
[`../person_removal`](../person_removal/README.md).
