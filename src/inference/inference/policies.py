"""Policies for the runtime: callables from an observation to an action chunk (adapter.labels)."""

import json
from pathlib import Path

import numpy as np

from adapter.labels import chunk_actions, load_episode


class ReplayPolicy:
    """A recorded episode played back as a policy: at step t, the chunk labelled at step t of the
    recording, ignoring the observation. Everything around it (anchoring on the live TCP, chunk
    execution, the teleop) is what a trained policy runs through, so a replay that reproduces the
    demonstration tests all of it."""

    def __init__(self, episode, arm, fps, chunk_size):
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
        return chunk_actions(self.episode, observation["step"], self.chunk_size)


def model_dir(model, revision=None):
    """Local checkpoint directory: `model` itself or its Hub download."""
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision))


def model_config(model, revision=None):
    """The checkpoint's config.json, without loading the weights."""
    if Path(model).is_dir():
        return json.loads((Path(model) / "config.json").read_text())
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(model, "config.json", revision=revision)).read_text())


class LeRobotPolicy:
    """Any LeRobot policy (ACT, SmolVLA, ...) through LeRobot's own loading and processors.

    Observation: `observation.images.<key>` (H, W, 3) uint8 RGB and `observation.state`
    (labels.model_state). Returns the unnormalized (chunk_size, 7) chunk. Without LeRobot processor
    files, normalizes with the checkpoint's normalization_stats.json.

    With `rtc` (RTCConfig fields): `prev`, the unexecuted actions of the current chunk, is inpainted
    over the first `delay` steps (Real-Time Chunking).
    """

    def __init__(self, model, device, task="", rtc=None, revision=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors
        from lerobot.processor import NormalizerProcessorStep

        path = model_dir(model, revision)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device, cfg.pretrained_path = device, path
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        if rtc:
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            from lerobot.rollout.inference.rtc import supports_rtc_inference

            if not supports_rtc_inference(self.policy):
                raise ValueError(f"{cfg.type} does not support RTC: leave `rtc` out of its inference settings")
            from lerobot.configs import RTCAttentionSchedule

            rtc = dict(rtc)
            if "prefix_attention_schedule" in rtc:
                rtc["prefix_attention_schedule"] = RTCAttentionSchedule[rtc["prefix_attention_schedule"].upper()]
            self.policy.config.rtc_config = RTCConfig(**rtc)
            self.policy.init_rtc_processor()
        self.policy.to(device).eval()

        if (path / "policy_preprocessor.json").exists():
            self.pre, self.post = make_pre_post_processors(
                cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})
        else:
            stats = json.loads((path / "normalization_stats.json").read_text())
            stats = {k: {s: np.asarray(v, dtype=np.float32) for s, v in st.items() if v is not None}
                     for k, st in stats.items()}
            self.pre, self.post = make_pre_post_processors(cfg, dataset_stats=stats)
        self.normalizer = next(s for s in self.pre.steps if isinstance(s, NormalizerProcessorStep))
        self.torch, self.device, self.task, self.rtc = torch, torch.device(device), task, rtc
        self.type, self.chunk_size = cfg.type, cfg.chunk_size
        self.features = list(cfg.input_features)

    def reset(self):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def __call__(self, observation, delay=0, prev=None):
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.processor import TransitionKey, create_transition

        torch = self.torch
        # RTC guidance needs gradients, which inference_mode forbids
        with torch.no_grad() if self.rtc else torch.inference_mode():
            batch = prepare_observation_for_inference(
                {k: observation[k] for k in self.features}, self.device, self.task)
            batch = self.pre(batch)
            kwargs = {}
            if self.rtc:
                if prev is not None:  # normalized, padded to the horizon, as LeRobot's RTC engine does
                    prev = torch.as_tensor(prev, dtype=torch.float32, device=self.device)
                    prev = self.normalizer(create_transition(action=prev))[TransitionKey.ACTION]
                    horizon = self.policy.config.rtc_config.execution_horizon
                    pad = torch.zeros((horizon, prev.shape[-1]), device=self.device)
                    pad[:min(horizon, len(prev))] = prev[:horizon]
                    prev = pad
                kwargs = {"inference_delay": delay, "prev_chunk_left_over": prev}
            actions = self.policy.predict_action_chunk(batch, **kwargs)
            return self.post(actions)[0].cpu().numpy().astype(np.float64)
