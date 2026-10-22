"""A corrector's rows: at every frame of a correction dataset, what the frozen base would have done and
what the person did, in the base's own action space.

The base is called as the robot's runtime calls it (inference.policies.LeRobotPolicy): the frame's
cameras and state, the frame before as its history (n_obs_steps), noise seeded per frame (PLAN_SEED)
so a plan is repeatable. Rows, one per frame:

  cond        the base's observation encoder output for the frame, the corrector's input z_t
  chunk_pi    the base's plan from the frame on: n_action_steps x 7, relative to the TCP at the frame
  chunk_demo  the person's goals over the same steps, made relative to the same TCP (labels.chunk_actions;
              past the episode's end the last goal repeats, the arm holding, as training pads it)
  episode, frame

The correction a corrector learns is chunk_demo - chunk_pi. Datasets are the exporter's raw layout
(15-D state, 8-D absolute goal and trigger); this module turns them into the 17-D state and 7-D
relative chunks of a relative checkpoint (training/TRAINING.md), the layout of the VR-trained bases.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

PLAN_SEED = 0
CAMERA_PREFIX = 'observation.images.'


def relative(goals, state):
    """(K, 8) absolute goals [x y z qx qy qz qw trigger] and the (15,) raw state at the frame: (K, 7)
    targets relative to the frame's TCP (training/TRAINING.md tcp_relative, labels.chunk_actions)."""
    R_tcp = Rotation.from_quat(state[3:7])
    dp = goals[:, :3] - state[:3]
    dr = (Rotation.from_quat(goals[:, 3:7]) * R_tcp.inv()).as_rotvec()
    return np.column_stack([dp, dr, goals[:, 7]]).astype(np.float32)


def image(frame):
    """A dataset camera frame (3, H, W) float in [0, 1]: (H, W, 3) uint8 RGB, as the runtime receives it."""
    return (frame.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()


class Encoded:
    """Records the observation encoder's output of each call (the DiT's conditioning vector)."""

    def __init__(self, encoder):
        self.last = None
        original = encoder.encode

        def encode(batch):
            self.last = original(batch)
            return self.last

        encoder.encode = encode


@torch.no_grad()
def build(model, root, repo_id, device='cuda', task=None, log=print):
    """Rows (tensors) and meta of every frame of the LeRobot dataset at `root`."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from adapter.labels import model_state
    from inference.policies import LeRobotPolicy

    ds = LeRobotDataset(repo_id, root=root, video_backend='pyav')
    task = task or ds.meta.tasks.index[0]
    lp = LeRobotPolicy(str(model), device, task=task)
    if lp.absolute or lp.policy.config.output_features['action'].shape[0] != 7:
        raise ValueError(f'{model}: rows are built for relative (17-D state, 7-D action) checkpoints')
    encoded = Encoded(lp.policy.observation_encoder)
    steps = lp.policy.config.n_action_steps
    cameras = [k for k in lp.features if k.startswith(CAMERA_PREFIX)]

    states = np.stack(ds.hf_dataset['observation.state']).astype(np.float64)
    goals = np.stack(ds.hf_dataset['action']).astype(np.float64)
    episodes = np.asarray(ds.hf_dataset['episode_index'])
    frames = np.asarray(ds.hf_dataset['frame_index'])

    rows = {'cond': [], 'chunk_pi': [], 'chunk_demo': [], 'episode': [], 'frame': []}
    previous = None
    for i in range(len(ds)):
        item = ds[i]
        s = states[i]
        observation = {k: image(item[k]) for k in cameras}
        observation['observation.state'] = model_state(s[:3], s[3:7], s[7], s[8:15])
        if frames[i] == 0:
            previous = observation  # LeRobot starts an episode by repeating its first observation
        torch.manual_seed(PLAN_SEED)
        plan = lp(observation, history=[previous])[:steps]
        previous = observation

        last = np.flatnonzero(episodes == episodes[i]).max()
        k = np.minimum(np.arange(i, i + steps), last)
        rows['cond'].append(encoded.last[0].float().cpu())
        rows['chunk_pi'].append(torch.as_tensor(plan, dtype=torch.float32))
        rows['chunk_demo'].append(torch.as_tensor(relative(goals[k], s)))
        rows['episode'].append(int(episodes[i]))
        rows['frame'].append(int(frames[i]))
        if i % 100 == 0:
            log(f'frame {i}/{len(ds)}')

    tensors = {
        'cond': torch.stack(rows['cond']).half(),
        'chunk_pi': torch.stack(rows['chunk_pi']),
        'chunk_demo': torch.stack(rows['chunk_demo']),
        'episode': torch.tensor(rows['episode'], dtype=torch.int32),
        'frame': torch.tensor(rows['frame'], dtype=torch.int32),
    }
    meta = {
        'model': str(model),
        'type': lp.policy.config.type,
        'repo_id': repo_id,
        'root': str(root),
        'task': task,
        'fps': ds.fps,
        'frames': len(ds),
        'episodes': sorted({int(e) for e in episodes}),
        'n_action_steps': steps,
        'plan_seed': PLAN_SEED,
    }
    return tensors, meta


# The first corrector: one nudge for the whole plan, linear in the base's features.


def nudge(chunk_pi, chunk_demo):
    """The whole-plan nudge per row: the person's plan minus the base's, averaged over the plan's
    steps, (N, 7). Rotation vectors are subtracted directly, a first-order approximation for the
    small angles of a correction."""
    return (chunk_demo - chunk_pi).mean(axis=1)


def fit_ridge(z, y, lam):
    """W (d, k) and b (k,) minimising |z W + b - y|^2 + lam |W|^2 over standardised features; returns
    (W, b, mean, std) with mean and std of z for standardising new rows. Solved in the dual
    (n x n), since the features (d) outnumber the rows.

    A feature that does not vary over the rows (e.g. the task text's embedding) carries no information:
    it gets std 1 and weight exactly 0, so a live value that differs from the rows' by rounding (they
    are cached as float16) cannot be amplified."""
    mean, spread = z.mean(0), z.std(0)
    varies = spread > CONSTANT
    std = np.where(varies, spread, 1.0)
    x = np.where(varies, (z - mean) / std, 0.0)
    b = y.mean(0)
    K = x @ x.T
    alpha = np.linalg.solve(K + lam * np.eye(len(x)), y - b)
    return x.T @ alpha, b, mean, std


def predict(model, z):
    W, b, mean, std = model
    return ((z - mean) / std) @ W + b


def plan_error(chunk_pi, chunk_demo, offset):
    """Per row: the mean distance (cm) between the person's plan and the base's plan shifted by
    `offset` (N, 7) over every step, and the mean gripper difference."""
    shifted = chunk_pi + offset[:, None, :]
    position = np.linalg.norm((shifted - chunk_demo)[:, :, :3], axis=2).mean(1) * 100
    gripper = np.abs((shifted - chunk_demo)[:, :, 6]).mean(1)
    return position, gripper


LAMBDAS = np.logspace(0, 6, 13)
CONSTANT = 1e-6  # a feature's spread over the rows below this: it does not vary


def leave_one_out(tensors):
    """Each correction held out in turn: ridge trained on the others (lambda chosen by leaving one of
    those out in turn), and the corrected plan's error on the held-out one for the base alone, a
    constant nudge (the training mean), the linear nudge and the true nudge (the best any one nudge
    per row can do). Returns one dict per held-out correction."""
    z = tensors['cond'].float().numpy().astype(np.float64)
    pi = tensors['chunk_pi'].numpy().astype(np.float64)
    demo = tensors['chunk_demo'].numpy().astype(np.float64)
    episode = tensors['episode'].numpy()
    y = nudge(pi, demo)
    results = []
    for held in np.unique(episode):
        train, test = episode != held, episode == held
        lam = _choose_lambda(z[train], y[train], episode[train])
        model = fit_ridge(z[train], y[train], lam)
        offsets = {
            'base': np.zeros_like(y[test]),
            'constant': np.broadcast_to(y[train].mean(0), y[test].shape),
            'linear': predict(model, z[test]),
            'true nudge': y[test],
        }
        row = {'held_out': int(held), 'frames': int(test.sum()), 'lambda': float(lam)}
        for name, offset in offsets.items():
            position, gripper = plan_error(pi[test], demo[test], offset)
            row[name] = {'position_cm': float(position.mean()), 'gripper': float(gripper.mean())}
        results.append(row)
    return results


def _choose_lambda(z, y, episode):
    """The lambda with the lowest held-out nudge error, each training correction held out in turn."""
    errors = []
    for lam in LAMBDAS:
        e = []
        for held in np.unique(episode):
            tr, te = episode != held, episode == held
            e.append(np.linalg.norm(predict(fit_ridge(z[tr], y[tr], lam), z[te]) - y[te], axis=1).mean())
        errors.append(np.mean(e))
    return LAMBDAS[int(np.argmin(errors))]


def save_adapter(tensors, meta, path, model, results):
    """Fit ridge on every correction (lambda chosen by holding each out in turn) and write it as a
    nudge adapter (inference.nudge) for `model`, with `results` (leave_one_out's) beside it."""
    import json
    from pathlib import Path

    from inference import nudge as nudge_adapter

    z = tensors['cond'].float().numpy().astype(np.float64)
    y = nudge(tensors['chunk_pi'].numpy().astype(np.float64), tensors['chunk_demo'].numpy().astype(np.float64))
    episode = tensors['episode'].numpy()
    lam = _choose_lambda(z, y, episode)
    W, b, mean, std = fit_ridge(z, y, lam)
    path = nudge_adapter.save(
        path,
        W,
        b,
        mean,
        std,
        model,
        corrections={
            'repo_id': meta.get('repo_id'),
            'root': meta.get('root'),
            'task': meta.get('task'),
            'episodes': [int(e) for e in np.unique(episode)],
            'frames': int(len(episode)),
        },
        fit={
            'method': 'ridge',
            'lambda': float(lam),
            'target': "whole-plan nudge: the person's plan minus the base's, averaged over the plan",
            'n_action_steps': meta.get('n_action_steps'),
            'plan_seed': meta.get('plan_seed'),
        },
    )
    (Path(path) / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    return path
