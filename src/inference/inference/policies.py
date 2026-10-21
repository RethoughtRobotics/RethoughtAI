"""Policies for the runtime: callables from an observation to an action chunk (adapter.labels)."""

import json
from pathlib import Path

import numpy as np

# Policies whose predict_action_chunk reads the last n_obs_steps observations from queues that only
# select_action fills: LeRobotPolicy fills them itself from the observations it is given.
OBS_HISTORY = {'multi_task_dit'}


def raw_state(tcp_p, tcp_q, gripper, joints):
    """The dataset's 15-D state as exported (lerobot_export): TCP position, quaternion [x y z w],
    gripper, 7 joints. The quaternion takes the sign with qy >= 0, as every recorded frame has it (the
    gripper points down: qy 0.87 .. 1 in BlueCupToContainer)."""
    q = np.asarray(tcp_q, dtype=np.float64)
    return np.concatenate([tcp_p, -q if q[1] < 0 else q, [gripper], joints]).astype(np.float32)


def letterbox(image, height, width):
    """(H, W, 3) uint8 image scaled to fit (height, width), keeping its aspect, centred on black: how
    BlueCupToContainer's 224 x 224 frames were made from the cameras' (mean error 0.006 against them).
    An image already at (height, width), the checkpoints trained at camera resolution, is unchanged."""
    h, w = image.shape[:2]
    if (h, w) == (height, width):
        return image
    scale = min(height / h, width / w)
    nh, nw = round(h * scale), round(w * scale)
    out = np.zeros((height, width, 3), np.uint8)
    top, left = (height - nh) // 2, (width - nw) // 2
    out[top : top + nh, left : left + nw] = stretch(image, nh, nw)
    return out


def stretch(image, height, width):
    """(H, W, 3) uint8 image resized to (height, width), aspect not kept, as a policy with an
    `image_resize_shape` resizes its cameras' frames."""
    if image.shape[:2] == (height, width):
        return image
    import torch
    import torch.nn.functional as F

    resized = F.interpolate(
        torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float(),
        (height, width),
        mode='bilinear',
        antialias=True,
    )
    return resized[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


class ReplayPolicy:
    """A recorded episode played back as a policy: at step t, the chunk labelled at step t of the
    recording, ignoring the observation. Everything around it (anchoring on the live TCP, chunk
    execution, the teleop) is what a trained policy runs through, so a replay that reproduces the
    demonstration tests all of it."""

    def __init__(self, episode, arm, fps, chunk_size):
        from adapter.labels import load_episode

        self.episode = load_episode(episode, arm, fps)
        self.chunk_size = chunk_size

    @property
    def start(self):
        """The demonstration's first TCP pose (position, quaternion): its actions are relative to the
        TCP, so the replay reproduces it only from where it began."""
        return self.episode.tcp_p[0], self.episode.tcp_q[0]

    def __len__(self):
        return len(self.episode)

    def __call__(self, observation):
        from adapter.labels import chunk_actions

        return chunk_actions(self.episode, observation['step'], self.chunk_size)


def model_dir(model, revision=None):
    """Local checkpoint directory: `model` itself or its Hub download."""
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision))


def model_config(model, revision=None):
    """The checkpoint's config.json, without loading the weights."""
    if Path(model).is_dir():
        return json.loads((Path(model) / 'config.json').read_text())
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(model, 'config.json', revision=revision)).read_text())


def train_config(model, revision=None):
    """The checkpoint's train_config.json, or None if it has none."""
    if Path(model).is_dir():
        path = Path(model) / 'train_config.json'
    else:
        from huggingface_hub import hf_hub_download

        try:
            path = Path(hf_hub_download(model, 'train_config.json', revision=revision))
        except Exception:
            return None
    return json.loads(path.read_text()) if path.exists() else None


def trained_lead(model, revision=None):
    """The training dataset's name if its labels already lead the TCP (an export's `_lead<N>ms`
    variant, e.g. BlueCupRedContainer_kin_lead640ms), else None: a run-time lead would add to it."""
    import re

    train = train_config(model, revision)
    name = train and train['dataset']['repo_id']
    return name if name and re.search(r'_lead\d+ms', name) else None


def trained_tasks(model, revision=None):
    """The task prompts of the dataset (at the revision) the checkpoint was trained on, or () if
    unknown. Language-conditioned policies (SmolVLA, MultiTask DiT) only know these exact strings."""
    train = train_config(model, revision)
    if not train:
        return ()
    import pandas as pd
    from huggingface_hub import hf_hub_download

    dataset = train['dataset']
    try:
        path = hf_hub_download(
            dataset['repo_id'], 'meta/tasks.parquet', repo_type='dataset', revision=dataset.get('revision')
        )
    except Exception:  # offline and not cached, or an older dataset format
        return ()
    return tuple(pd.read_parquet(path).index)


def chunk_length(config):
    """Actions in a chunk the policy returns (config.json as a dict): `chunk_size` (ACT, SmolVLA), or
    `n_action_steps` for a policy that plans `horizon` steps and returns the first ones (VITA)."""
    return config.get('chunk_size') or config['n_action_steps']


def action_layout(config):
    """Validate the state/action contract before loading weights or starting a runtime."""
    layout = (
        config['input_features']['observation.state']['shape'][0],
        config['output_features']['action']['shape'][0],
    )
    if layout == (17, 7):
        return 'relative'
    if layout == (15, 8):
        return 'absolute'
    if layout == (28, 14):
        return 'bimanual_absolute'
    raise ValueError(f'state/action dims {layout}, expected (17, 7), (15, 8) or (28, 14)')


def load_config(path):
    """The checkpoint's PreTrainedConfig, without the fields this LeRobot doesn't have (a checkpoint
    saved by a newer LeRobot, e.g. `dtype`). A dropped field that isn't null is warned about."""
    import dataclasses
    import tempfile
    import warnings

    from lerobot.configs.policies import PreTrainedConfig

    import inference.vita  # noqa: F401  registers the "vita" policy type

    config = json.loads((path / 'config.json').read_text())
    known = {f.name for f in dataclasses.fields(PreTrainedConfig.get_choice_class(config['type']))} | {'type'}
    extra = {k: config.pop(k) for k in list(config) if k not in known}
    if not extra:
        return PreTrainedConfig.from_pretrained(path)
    if ignored := {k: v for k, v in extra.items() if v is not None}:
        warnings.warn(f'{path}: this LeRobot ignores {ignored}')
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'config.json').write_text(json.dumps(config))
        return PreTrainedConfig.from_pretrained(tmp)


class LeRobotPolicy:
    """Any LeRobot policy (ACT, SmolVLA, ...) through LeRobot's own loading and processors.

    Observation: `observation.images.<key>` (H, W, 3) uint8 RGB and `observation.state` (`state()`).
    Returns the unnormalized (chunk_size, 7) chunk, relative to the TCP (labels.chunk_actions). Without
    LeRobot processor files, normalizes with the checkpoint's normalization_stats.json.

    Layouts, told apart by the checkpoint's feature shapes:
      17-D state, 7-D actions   trained with training/TRAINING.md's transform (ACT, SmolVLA, VITA, MultiTask DiT)
            15-D state, 8-D actions   trained on the dataset as exported (GR00T, kinesthetic VITA): absolute goals
                                [x y z qx qy qz qw trigger] out, made relative to the TCP observed
                28-D state, 14-D actions  raw bimanual pose-only export: absolute left/right poses stay
                                  absolute for BimanualPolicyNode; no gripper or relative conversion

    With `rtc` (RTCConfig fields): `prev`, the unexecuted actions of the current chunk, is inpainted
    over the first `delay` steps (Real-Time Chunking).
    With `adapter` (a lora.save directory), VITA runs with that LoRA adapter at `alpha` (0: the base
    exactly), on the flow net or the modules its adapter.json names as root; with a weights adapter
    (lora.save_weights: ttt update's full, flow, flow_last), its weights blended with the base's by
    `alpha` (0 the base, 1 the tuned weights exactly); with a head adapter (lora.save_head: ttt update's
    residual, latent_offset), the head's output scaled by `alpha`.
    With `explore` (inference.explore.Explorer's sigma, seed, noise and record), VITA plans from a seeded
    offset of its latent, its flow optionally stochastic, every plan optionally recorded (plan S7).
    """

    def __init__(
        self, model, device, task='', rtc=None, revision=None, overrides=None, adapter=None, alpha=1.0, explore=None
    ):
        import torch
        from lerobot.policies import get_policy_class, make_pre_post_processors
        from lerobot.processor import NormalizerProcessorStep

        path = model_dir(model, revision)
        cfg = load_config(path)
        layout = action_layout(json.loads((path / 'config.json').read_text()))
        self.bimanual = layout == 'bimanual_absolute'
        self.absolute = layout in ('absolute', 'bimanual_absolute')
        if self.absolute and rtc:
            raise ValueError(
                f'{cfg.type} predicts absolute goals; RTC continues TCP-relative chunks: '
                'leave `rtc` out of its inference settings'
            )
        cfg.device, cfg.pretrained_path = device, path
        if cfg.type == 'groot' and device == 'cpu':  # its bf16 autocast mixes dtypes on the CPU
            cfg.use_bf16, cfg.model_params_fp32 = False, True
        for key, value in (overrides or {}).items():  # e.g. num_steps: fewer flow steps, less latency
            setattr(cfg, key, value)
        # Cameras of different shapes, which a policy resizes to one itself (MultiTask DiT's
        # image_resize_shape, by a newer LeRobot): this one requires them equal, so they get that shape
        # here, stretched as the policy would. Equal shapes keep their letterbox (BlueCupToContainer).
        cameras = {tuple(f.shape) for k, f in cfg.input_features.items() if k.startswith('observation.images.')}
        self.resize = getattr(cfg, 'image_resize_shape', None) if len(cameras) > 1 else None
        if self.resize:
            for key, feature in cfg.input_features.items():
                if key.startswith('observation.images.'):
                    feature.shape = (feature.shape[0], *self.resize)
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        if rtc:
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            from lerobot.rollout.inference.rtc import supports_rtc_inference

            if not supports_rtc_inference(self.policy):
                raise ValueError(f'{cfg.type} does not support RTC: leave `rtc` out of its inference settings')
            from lerobot.configs import RTCAttentionSchedule

            rtc = dict(rtc)
            if 'prefix_attention_schedule' in rtc:
                rtc['prefix_attention_schedule'] = RTCAttentionSchedule[rtc['prefix_attention_schedule'].upper()]
            self.policy.config.rtc_config = RTCConfig(**rtc)
            self.policy.init_rtc_processor()
        self.policy.to(device).eval()
        if adapter:
            from inference import lora

            if cfg.type != 'vita':
                raise ValueError(f'adapters are for VITA checkpoints, not {cfg.type}')
            lora.load_policy(self.policy, adapter, alpha)
        self.explorer = None
        if explore:
            from inference.explore import Explorer

            if cfg.type != 'vita':
                raise ValueError(f'exploration is in a VITA latent, not {cfg.type}')
            self.explorer = Explorer(self.policy.vita, **explore)

        if (path / 'policy_preprocessor.json').exists():
            overrides = {'device_processor': {'device': device}}
            if (path / 'tokenizer').is_dir():  # saved with the checkpoint, named relative to it
                overrides['tokenizer_processor'] = {'tokenizer_name': str(path / 'tokenizer')}
            self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=path, preprocessor_overrides=overrides)
        else:
            stats = json.loads((path / 'normalization_stats.json').read_text())
            stats = {
                k: {s: np.asarray(v, dtype=np.float32) for s, v in st.items() if v is not None}
                for k, st in stats.items()
            }
            self.pre, self.post = make_pre_post_processors(cfg, dataset_stats=stats)
        # for RTC; GR00T normalizes inside its own packing step
        self.normalizer = next((s for s in self.pre.steps if isinstance(s, NormalizerProcessorStep)), None)
        self.torch, self.device, self.task, self.rtc = torch, torch.device(device), task, rtc
        self.type, self.chunk_size = cfg.type, chunk_length(vars(cfg))
        self.n_obs_steps = cfg.n_obs_steps if cfg.type in OBS_HISTORY else 1
        self.features = list(cfg.input_features)
        self.image_shapes = {
            k: tuple(f.shape[1:]) for k, f in cfg.input_features.items() if k.startswith('observation.images.')
        }

    def reset(self):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def state(self, tcp_p, tcp_q, gripper, joints):
        """observation.state for this checkpoint's layout."""
        if getattr(self, 'bimanual', False):
            # Dataset order is [left pose, left joints, right pose, right joints].
            # Signs are made continuous at TCP reception, not forced to qy >= 0.
            return np.concatenate([np.r_[tcp_p[a], tcp_q[a], joints[a]] for a in ('left', 'right')]).astype(np.float32)
        if self.absolute:
            return raw_state(tcp_p, tcp_q, gripper, joints)
        from adapter.labels import model_state

        return model_state(tcp_p, tcp_q, gripper, joints)

    def __call__(self, observation, delay=0, prev=None, tcp=None, history=()):
        """`tcp` (position, quaternion), the TCP observed: absolute goals are made relative to it.
        `history`: for a policy that sees past steps (n_obs_steps), the observations one step apart
        before this one, oldest first; fewer repeat the oldest given, as LeRobot starts an episode."""
        from lerobot.policies.utils import populate_queues, prepare_observation_for_inference
        from lerobot.processor import TransitionKey, create_transition

        if self.absolute and not getattr(self, 'bimanual', False) and tcp is None:
            raise ValueError(f'{self.type} predicts absolute goals: pass the TCP observed')
        torch = self.torch

        def preprocess(obs):
            return self.pre(
                prepare_observation_for_inference(
                    {
                        k: (stretch if self.resize else letterbox)(obs[k], *self.image_shapes[k])
                        if k in self.image_shapes
                        else obs[k]
                        for k in self.features
                    },
                    self.device,
                    self.task,
                )
            )

        # RTC guidance needs gradients, which inference_mode forbids
        with torch.no_grad() if self.rtc else torch.inference_mode():
            if self.type in OBS_HISTORY:  # its queues hold exactly these observations
                self.policy.reset()
                for obs in [*history[-(self.n_obs_steps - 1) :], observation]:
                    # cameras stacked, as the queue holds them; no `action` (None here), as select_action
                    batch = self.policy._prepare_batch({k: v for k, v in preprocess(obs).items() if k != 'action'})
                    self.policy._queues = populate_queues(self.policy._queues, batch)
            else:
                batch = preprocess(observation)
            kwargs = {}
            if self.rtc:
                if prev is not None:  # normalized, padded to the horizon, as LeRobot's RTC engine does
                    prev = torch.as_tensor(prev, dtype=torch.float32, device=self.device)
                    prev = self.normalizer(create_transition(action=prev))[TransitionKey.ACTION]
                    horizon = self.policy.config.rtc_config.execution_horizon
                    pad = torch.zeros((horizon, prev.shape[-1]), device=self.device)
                    pad[: min(horizon, len(prev))] = prev[:horizon]
                    prev = pad
                kwargs = {'inference_delay': delay, 'prev_chunk_left_over': prev}
            actions = self.policy.predict_action_chunk(batch, **kwargs)
            actions = self.post(actions)[0].float().cpu().numpy().astype(np.float64)
        if self.absolute and not getattr(self, 'bimanual', False):
            from scipy.spatial.transform import Rotation

            from adapter.labels import relative_actions

            actions = relative_actions(actions[:, :3], Rotation.from_quat(actions[:, 3:7]), actions[:, 7], *tcp)
        return actions
