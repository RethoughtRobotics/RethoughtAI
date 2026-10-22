"""A corrector's rows: at every frame of a correction dataset, what the frozen base would have done and
what the person did, in the base's own action space.

The base is called as the robot's runtime calls it (inference.policies.LeRobotPolicy): the frame's
cameras and state, the frame before as its history (n_obs_steps). Its plan is the one it made on the
robot where the dataset has it (adapt lerobot --plans: policy.plan), and the rows are then only the
frames a plan was made from (every few frames); otherwise it plans again here at every frame, noise
seeded per frame (PLAN_SEED) so a plan is repeatable, which is a new sample, not the plan that ran.
Rows:

  cond        the base's observation encoder output for the frame, the corrector's input z_t
  chunk_pi    the base's plan from the frame on: n_action_steps x 7, relative to the TCP at the frame
  chunk_demo  the person's goals over the same steps, made relative to the same TCP (labels.chunk_actions;
              past the episode's end the last goal repeats, the arm holding, as training pads it)
  state       the base's 17-D state at the frame (labels.model_state)
  intervention  1 where the person drove (the dataset's `intervention`, adapt lerobot --intervention);
              a dataset without it is all corrections, 1 everywhere
  step_intervention  the same at each step of the plan (n_action_steps; past the episode's end its
              last frame): which of the plan's steps the person drove
  episode, frame

The correction a corrector learns is chunk_demo - chunk_pi (ttt.correctors, its README.md). Datasets are the exporter's raw layout
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
    if 'intervention' in ds.features:
        intervention = np.stack(ds.hf_dataset['intervention']).reshape(-1).astype(np.float32)
    else:
        intervention = np.ones(len(ds), np.float32)
    logged = 'policy.plan' in ds.features
    if logged:
        ran = np.stack(ds.hf_dataset['policy.plan']).reshape(len(ds), -1, 8).astype(np.float64)
        planned = np.stack(ds.hf_dataset['policy.planned']).reshape(-1) > 0.5
        if ran.shape[1] < steps:
            raise ValueError(f'the logged plans have {ran.shape[1]} steps, the base plans {steps}')
    else:
        planned = np.ones(len(ds), bool)

    rows = {k: [] for k in ('cond', 'chunk_pi', 'chunk_demo', 'state', 'intervention', 'step_intervention')}
    rows.update(episode=[], frame=[])
    previous = None
    for i in range(len(ds)):
        item = ds[i]
        s = states[i]
        observation = {k: image(item[k]) for k in cameras}
        observation['observation.state'] = model_state(s[:3], s[3:7], s[7], s[8:15])
        if frames[i] == 0:
            previous = observation  # LeRobot starts an episode by repeating its first observation
        history, previous = previous, observation
        if not planned[i]:
            continue
        if logged:  # the plan that ran; only the encoder is called
            lp.encode(observation, history=[history])
            plan = relative(ran[i, :steps], s)
        else:
            torch.manual_seed(PLAN_SEED)
            plan = lp(observation, history=[history])[:steps]

        last = np.flatnonzero(episodes == episodes[i]).max()
        k = np.minimum(np.arange(i, i + steps), last)
        rows['cond'].append(encoded.last[0].float().cpu())
        rows['chunk_pi'].append(torch.as_tensor(plan, dtype=torch.float32))
        rows['chunk_demo'].append(torch.as_tensor(relative(goals[k], s)))
        rows['step_intervention'].append(torch.as_tensor(intervention[k]))
        rows['intervention'].append(float(intervention[i]))
        rows['state'].append(torch.as_tensor(observation['observation.state'], dtype=torch.float32))
        rows['episode'].append(int(episodes[i]))
        rows['frame'].append(int(frames[i]))
        if i % 100 == 0:
            log(f'frame {i}/{len(ds)}')

    tensors = {
        'cond': torch.stack(rows['cond']).half(),
        'chunk_pi': torch.stack(rows['chunk_pi']),
        'chunk_demo': torch.stack(rows['chunk_demo']),
        'state': torch.stack(rows['state']),
        'intervention': torch.tensor(rows['intervention'], dtype=torch.float32),
        'step_intervention': torch.stack(rows['step_intervention']),
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
        'plan_seed': None if logged else PLAN_SEED,
        'plans': 'logged: the plans the base made on the robot' if logged else 'resampled here, seeded',
    }
    return tensors, meta
