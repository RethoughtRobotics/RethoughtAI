"""The offline corrector (the demonstration plays the human) and the path error it gates on.

A frame is corrected when the base's planned executed waypoints (the first n_action_steps of its
chunk) are on average more than `gate` cm from where the demonstration went; its target is z_demo.
"""

import torch


def path_error(chunk, demo, pad, steps):
    """Mean distance (m) between the first `steps` positions of each chunk and the demonstration's,
    over the steps inside the episode (`pad` False). chunk, demo: (N, H, >=3); pad: (N, H)."""
    d = torch.linalg.vector_norm(chunk[:, :steps, :3].float() - demo[:, :steps, :3].float(), dim=-1)
    valid = (~pad[:, :steps]).float()
    return (d * valid).sum(-1) / valid.sum(-1).clamp(min=1)


def in_episodes(cache, episodes):
    """Mask of the cache rows in `episodes`."""
    return torch.isin(cache['episode'], torch.as_tensor(list(episodes), dtype=cache['episode'].dtype))


def corrected(cache, episodes, gate=3.0, steps=16):
    """Mask of the frames of `episodes` the corrector takes over: the base off by more than `gate` cm
    (every frame if gate is None)."""
    mask = in_episodes(cache, episodes)
    if gate is None:
        return mask
    return mask & (path_error(cache['chunk_pi'], cache['chunk_demo'], cache['pad'], steps) > gate / 100)


def summary(cache, episodes, gate=3.0, steps=16):
    """The cost column: corrected frames in all and per episode, out of the episodes' frames."""
    mask = corrected(cache, episodes, gate, steps)
    per = {int(e): int(mask[cache['episode'] == e].sum()) for e in episodes}
    return {
        'gate_cm': gate,
        'episodes': [int(e) for e in episodes],
        'corrected_frames': int(mask.sum()),
        'frames': int(in_episodes(cache, episodes).sum()),
        'per_episode': per,
    }
