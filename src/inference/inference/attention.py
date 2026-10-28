"""Where a policy looks, for one observation: scores over its image regions, for recap's ML tab.

Two methods, one output format:
- attention (SmolVLA): how much the action expert's queries attend to each image token, per flow
  step, layer and head. What the model reads, not proof of what it uses.
- occlusion (any policy, e.g. VITA): each cell of a grid over each camera is grayed out in turn and
  the action chunk re-predicted; the score is how far the chunk moves. Causal, but one inference
  per cell, so for fast policies.

`ClsAttention` is a third, live one (MultiTask DiT): where its CLIP's CLS token attends in the last
layer, kept from every inference the policy node runs anyway and published beside its chunks.

`Maps.scores` has named `dims`, the last always "token"; `segments` say which tokens are which: a
camera's are a grid of cells covering `extent` (top, left, bottom, right) of its frame, as fractions
of the frame (beyond 0..1 where the grid covers padding); language and state tokens have no grid.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from adapter.lerobot_export import JOINTS

OCCLUSION_GRID = (6, 8)  # cells per camera (rows, columns)
BATCH = 8  # frames per GPU call (halved while the GPU runs out of memory)
TARGETS = ('position', 'rotation', 'gripper')  # what occlusion measures the chunk's change in


@dataclass
class Segment:
    name: str
    start: int
    length: int
    grid: tuple[int, int] | None = None
    extent: tuple[float, float, float, float] | None = None


@dataclass
class Maps:
    method: str
    dims: list[str]
    scores: np.ndarray
    segments: list[Segment]
    labels: dict = field(default_factory=dict)  # dim -> a name per index, where indices aren't enough
    means: list[str] | None = None  # dims that may be averaged (default: all but the tokens)

    def averaged(self, dims: list[str]) -> 'Maps':
        """These scores with ``dims`` averaged out (those that may be: see ``means``)."""
        means = self.dims[:-1] if self.means is None else self.means
        drop = [d for d in dims if d in means and d in self.dims]
        if not drop:
            return self
        scores = self.scores.mean(axis=tuple(self.dims.index(d) for d in drop))
        keep = [d for d in self.dims if d not in drop]
        return Maps(
            self.method,
            keep,
            scores,
            self.segments,
            {d: v for d, v in self.labels.items() if d in keep},
            [d for d in means if d in keep],
        )

    def meta(self) -> dict:
        means = self.dims[:-1] if self.means is None else self.means
        return {
            'method': self.method,
            'dims': self.dims,
            'shape': list(self.scores.shape),
            'segments': [asdict(s) for s in self.segments],
            'labels': self.labels,
            'means': means,
        }


def joint_names(arm: str) -> list[str]:
    return [f'{arm}_{j}' for j in JOINTS]


def serve(policy, cfg: dict, images: list[dict], requests, replies, batch: int = BATCH) -> None:
    """Answer requests from recap, one JSON line each, until the input closes.

    First line out: what to send (the policy's camera topics, TCP, joints and gripper topics).
    Request: {"inputs": [npz path per frame], "out": npy path, "mean": [dims to average first]}; each
    npz holds each camera key's frame (H, W, 3) uint8, `tcp_p`, `tcp_q`, `joints` (named in the first
    line) and `gripper` (its state's position, 0 closed .. 100 open). Frames run `batch` at a time on
    the GPU. Reply: {"meta": Maps.meta()} with the frames' scores stacked in `out`, or {"error": ...}.
    """
    import json

    cameras = {c['key']: c['topic'] for c in images if c['key'] in policy.features}
    names = joint_names(cfg['arm'])
    replies.write(
        json.dumps({
            'ready': True,
            'type': policy.type,
            'cameras': cameras,
            'tcp': cfg['tcp'],
            'joint_states': cfg['joint_states'],
            'joints': names,
            'gripper': cfg['gripper'],
            'batch': batch,
        })
        + '\n'
    )
    replies.flush()
    for line in requests:
        try:
            request = json.loads(line)
            paths = request['inputs'] if isinstance(request['inputs'], list) else [request['inputs']]
            frames = []
            for path in paths:
                raw = np.load(path)
                tcp = (raw['tcp_p'], raw['tcp_q'])
                observation = {k: raw[k] for k in cameras}
                # as the policy node: gripper 0 open .. 1 closed
                observation['observation.state'] = policy.state(
                    *tcp, 1.0 - float(raw['gripper']) / 100.0, raw['joints']
                )
                frames.append((observation, tcp))
            maps = [m.averaged(request.get('mean', [])) for m in capture_many(policy, frames, batch)]
            np.save(request['out'], np.stack([m.scores for m in maps]))
            reply = {'meta': maps[0].meta()}
        except Exception as e:  # one bad frame must not end the session
            reply = {'error': f'{type(e).__name__}: {e}'}
        replies.write(json.dumps(reply) + '\n')
        replies.flush()


def capture(policy, observation: dict, tcp) -> Maps:
    """Maps for one observation (as LeRobotPolicy takes it), by the method that suits the policy."""
    return capture_many(policy, [(observation, tcp)])[0]


def capture_many(policy, frames: list[tuple[dict, tuple]], batch: int = BATCH) -> list[Maps]:
    """Maps for (observation, TCP) pairs; SmolVLA runs `batch` frames per GPU call."""
    if policy.type == 'smolvla':
        out = []
        for i in range(0, len(frames), batch):
            out += _halving(
                lambda n, part=frames[i : i + batch]: [
                    m for j in range(0, len(part), n) for m in smolvla_attention(policy, part[j : j + n])
                ],
                batch,
            )
        return out
    return [occlusion(policy, obs, tcp, batch=4 * batch) for obs, tcp in frames]


def _halving(run, n: int):
    """``run(n)``, with ``n`` halved while the GPU runs out of memory."""
    import torch

    while True:
        try:
            return run(n)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if n == 1:
                raise
            n //= 2


def _prepare(policy, observations: list[dict]) -> dict:
    """One batch of observations, each preprocessed as LeRobotPolicy does for one."""
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    from inference.policies import letterbox

    parts = [
        policy.pre(
            prepare_observation_for_inference(
                {
                    k: letterbox(o[k], *policy.image_shapes[k]) if k in policy.image_shapes else o[k]
                    for k in policy.features
                },
                policy.device,
                policy.task,
            )
        )
        for o in observations
    ]

    def merge(values):
        if torch.is_tensor(values[0]):
            return torch.cat(values)
        if isinstance(values[0], (list, tuple)):
            return [x for v in values for x in v]
        return values[0]

    return {k: merge([p[k] for p in parts]) for k in parts[0]}


# -- geometry: where a model's grid lies on the camera frame ----------------------------------


def _letterbox(h, w, height, width):
    """(top, left, height, width) of an (h, w) frame inside its letterbox to (height, width)."""
    if (h, w) == (height, width):
        return 0.0, 0.0, float(h), float(w)
    s = min(height / h, width / w)
    nh, nw = round(h * s), round(w * s)
    return (height - nh) // 2, (width - nw) // 2, nh, nw


def _to_frame(box_in_input, frame_hw, input_hw):
    """A (top, left, bottom, right) box in model-input pixels as fractions of the original frame."""
    top, left, nh, nw = _letterbox(*frame_hw, *input_hw)
    y0, x0, y1, x1 = box_in_input
    return ((y0 - top) / nh, (x0 - left) / nw, (y1 - top) / nh, (x1 - left) / nw)


# -- SmolVLA: the action expert's attention to the image tokens ---------------------------------


def smolvla_attention(policy, frames: list[tuple[dict, tuple]]) -> list[Maps]:
    """The action expert's attention to the prefix, for a batch of frames in one pass. Every frame gets
    the same seeded flow noise, so its map does not depend on the batch it ran in."""
    import torch

    model = policy.policy.model
    vwe = model.vlm_with_expert
    config = policy.policy.config
    chunk = config.chunk_size
    probs, sizes = [], {}

    original = vwe.eager_attention_forward

    def recording(attention_mask, batch_size, head_dim, q, k, v):
        """SmolVLA's eager attention (smolvlm_with_expert.eager_attention_forward), keeping the
        probabilities it computes for the expert's action queries instead of computing them twice."""
        if q.shape[1] != chunk:  # the prefix pass
            return original(attention_mask, batch_size, head_dim, q, k, v)
        groups = vwe.num_attention_heads // vwe.num_key_value_heads
        k = k.repeat_interleave(groups, dim=2)  # (B, L, H, D): each KV head shared by `groups` heads
        v = v.repeat_interleave(groups, dim=2)
        w = torch.matmul(q.float().transpose(1, 2), k.float().transpose(1, 2).transpose(2, 3)) * head_dim**-0.5
        w = torch.where(attention_mask[:, None], w, torch.finfo(w.dtype).min).softmax(-1).to(v.dtype)
        probs.append(w.mean(2))  # (B, heads, keys): mean over the action queries; stays on the GPU
        out = torch.matmul(w, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return out.reshape(batch_size, -1, k.shape[2] * head_dim)

    def size(name):
        def hook(module, args, out):  # tokens a module gives (returns None: the output stays)
            sizes.setdefault(name, out.shape[1])

        return hook

    hooks = [
        vwe.vlm.model.connector.register_forward_hook(size('image')),
        vwe.vlm.model.text_model.embed_tokens.register_forward_hook(size('language')),
    ]
    vwe.eager_attention_forward = recording
    try:
        with torch.inference_mode():
            batch = _prepare(policy, [obs for obs, _ in frames])
            generator = torch.Generator(policy.device).manual_seed(0)
            noise = torch.randn((1, chunk, config.max_action_dim), generator=generator, device=policy.device)
            policy.policy.reset()
            policy.policy.predict_action_chunk(batch, noise=noise.expand(len(frames), -1, -1).contiguous())
            cameras = [k for k in config.image_features if k in frames[0][0]]
            per_image, lang = sizes['image'], sizes['language']
            prefix = per_image * len(cameras) + lang + 1  # images, language, state: first in every layer's keys
            stacked = torch.stack([p[..., :prefix] for p in probs]).cpu().numpy()  # (calls, B, heads, prefix)
    finally:
        vwe.eager_attention_forward = original
        for h in hooks:
            h.remove()
        policy.policy.reset()

    observation = frames[0][0]
    layers = len(vwe.lm_expert.layers)
    steps = len(stacked) // layers

    side = int(round(per_image**0.5))
    canvas_h, canvas_w = config.resize_imgs_with_padding[1], config.resize_imgs_with_padding[0]
    segments = []
    for i, key in enumerate(cameras):
        h, w = observation[key].shape[:2]
        H, W = policy.image_shapes.get(key, (h, w))
        # resize_with_pad: scaled to fit the canvas, padded on the left and top
        ratio = max(W / canvas_w, H / canvas_h)
        rh, rw = int(H / ratio), int(W / ratio)
        pad_t, pad_l = canvas_h - rh, canvas_w - rw
        box = (-pad_t * ratio, -pad_l * ratio, rh * ratio, rw * ratio)  # the canvas in input pixels
        segments.append(
            Segment(key.rsplit('.', 1)[-1], i * per_image, per_image, (side, side), _to_frame(box, (h, w), (H, W)))
        )
    segments += [Segment('language', len(cameras) * per_image, lang), Segment('state', prefix - 1, 1)]
    labels = {'layer': [f'{i} {"cross" if _is_cross(vwe, i) else "self"}' for i in range(layers)]}
    return [
        Maps(
            'attention',
            ['step', 'layer', 'head', 'token'],
            stacked[:, b].reshape(steps, layers, -1, prefix).astype(np.float32),
            segments,
            labels,
        )
        for b in range(len(frames))
    ]


def _is_cross(vwe, layer: int) -> bool:
    """Expert layers alternate: cross-attention to the prefix, or self-attention over it and the actions."""
    every = getattr(vwe, 'self_attn_every_n_layers', 2) or 0
    return bool(every) and layer % every != 0


# -- occlusion: how far the action chunk moves when a region is grayed out ----------------------


def occlusion(policy, observation, tcp, grid=OCCLUSION_GRID, batch: int = 32) -> Maps:
    cameras = [k for k in policy.features if k.startswith('observation.images.') and k in observation]
    rows, cols = grid
    cells, segments = [], []  # (camera, (top, bottom, left, right) in frame pixels, gray (3,) uint8)
    for i, key in enumerate(cameras):
        frame = observation[key]
        h, w = frame.shape[:2]
        fill = frame.reshape(-1, 3).mean(0).astype(frame.dtype)
        cells += [
            (key, (r * h // rows, (r + 1) * h // rows, c * w // cols, (c + 1) * w // cols), fill)
            for r in range(rows)
            for c in range(cols)
        ]
        segments.append(Segment(key.rsplit('.', 1)[-1], i * rows * cols, rows * cols, grid, (0.0, 0.0, 1.0, 1.0)))
    if _on_gpu(policy, observation, cameras):
        chunks = _halving(lambda n: _occluded_on_gpu(policy, observation, cells, tcp, n), batch)
    else:
        variants = [observation]
        for key, (y0, y1, x0, x1), fill in cells:
            occluded = observation[key].copy()
            occluded[y0:y1, x0:x1] = fill
            variants.append({**observation, key: occluded})
        chunks = _halving(
            lambda n: [c for j in range(0, len(variants), n) for c in _predict(policy, variants[j : j + n], tcp)], batch
        )
    scores = [_change(chunks[0], c) for c in chunks[1:]]
    return Maps(
        'occlusion',
        ['target', 'token'],
        np.asarray(scores, np.float32).T,
        segments,
        {'target': ['position (mm)', 'rotation (deg)', 'gripper']},
        means=[],
    )  # units differ


def _on_gpu(policy, observation, cameras) -> bool:
    """Whether occluded variants can be made from the preprocessed frame: a batchable policy whose
    preprocessing keeps each camera's pixels in place (no letterbox)."""
    return (
        hasattr(policy, 'pre')
        and getattr(policy, 'n_obs_steps', 1) == 1
        and all(
            tuple(policy.image_shapes.get(k, observation[k].shape[:2])) == observation[k].shape[:2] for k in cameras
        )
    )


def _occluded_on_gpu(policy, observation, cells, tcp, n: int) -> list[np.ndarray]:
    """The frame's chunk and one per occluded cell, `n` per GPU call: the frame is preprocessed once
    and each variant grays a rectangle of its copy. The gray is the fill color run through the same
    preprocessing, so it matches graying the frame itself for any per-pixel normalization."""
    import torch

    with torch.inference_mode():
        base = _prepare(policy, [observation])
        grays = {}
        for key, _, fill in cells:
            if key not in grays:
                solid = {**observation, key: np.broadcast_to(fill, observation[key].shape).copy()}
                grays[key] = _prepare(policy, [solid])[key][0, :, :1, :1]  # (C, 1, 1)
        out = _chunks(policy, base, tcp)
        for j in range(0, len(cells), n):
            part = cells[j : j + n]
            batch = {
                k: v.repeat(len(part), *[1] * (v.ndim - 1))
                if torch.is_tensor(v)
                else (list(v) * len(part) if isinstance(v, (list, tuple)) else v)
                for k, v in base.items()
            }
            for b, (key, (y0, y1, x0, x1), _) in enumerate(part):
                batch[key][b, :, y0:y1, x0:x1] = grays[key]
            out += _chunks(policy, batch, tcp)
    return out


def _chunks(policy, batch: dict, tcp) -> list[np.ndarray]:
    """Unnormalized, TCP-relative action chunks for a preprocessed batch."""
    policy.policy.reset()
    actions = policy.post(policy.policy.predict_action_chunk(batch)).float().cpu().numpy().astype(np.float64)
    if getattr(policy, 'absolute', False):
        from scipy.spatial.transform import Rotation

        from adapter.labels import relative_actions

        return [relative_actions(a[:, :3], Rotation.from_quat(a[:, 3:7]), a[:, 7], *tcp) for a in actions]
    return list(actions)


def _predict(policy, observations: list[dict], tcp) -> list[np.ndarray]:
    """Action chunks for observations at the same TCP: one GPU call where the policy allows it."""
    import torch

    if not hasattr(policy, 'pre') or getattr(policy, 'n_obs_steps', 1) > 1:  # stand-ins; observation history
        out = []
        for obs in observations:
            torch.manual_seed(0)
            policy.reset()
            out.append(policy(obs, tcp=tcp))
        return out
    with torch.inference_mode():
        out = _chunks(policy, _prepare(policy, observations), tcp)
    policy.reset()
    return out


def _change(base, other) -> list[float]:
    """Mean change over the chunk: position (mm), rotation (deg), gripper (trigger units)."""
    d = other - base
    return [
        1000 * float(np.linalg.norm(d[:, :3], axis=1).mean()),
        float(np.degrees(np.linalg.norm(d[:, 3:6], axis=1)).mean()),
        float(np.abs(d[:, 6]).mean()),
    ]


class ClsAttention:
    """Where a MultiTask DiT's CLIP looks: the last layer's attention from the CLS token (the policy's
    conditioning, per image) to each patch, mean of heads, recomputed from that layer's input with its
    own layer norm and query and key projections (two small products; CLIP's forward is not rerun).

    It wraps the encoder's CLIP forward to keep the hidden states and the pixels of every call, so it
    costs nothing until `maps` is asked. Only the current step's images are mapped: the encoder
    flattens images as (batch, step, camera), so they are the last `cameras` of the call.
    """

    def __init__(self, encoder):
        self.cameras = [k.removeprefix('observation.images.') for k in encoder.camera_names]
        model = encoder.vision_encoder.model
        self.layer = model.vision_model.encoder.layers[-1]
        self.patch = model.config.patch_size
        self.pixels = self.states = None
        original = model.forward

        def keep(*args, pixel_values=None, **kwargs):
            out = original(*args, pixel_values=pixel_values, **{**kwargs, 'output_hidden_states': True})
            self.pixels, self.states = pixel_values, out.hidden_states
            return out

        model.forward = keep

    def maps(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Camera -> ((grid, grid) attention (the rest of 1 is on the CLS itself), (H, W, 3) uint8 view the CLIP saw) of the last
        call; empty before the first. The view is the normalized pixels stretched back to 0 .. 255 per
        channel: what the policy saw up to that channel's contrast."""
        import torch

        if self.states is None:
            return {}
        n = len(self.cameras)
        attn = self.layer.self_attn
        with torch.no_grad():
            h = self.layer.layer_norm1(self.states[-2][-n:])
            q = attn.q_proj(h[:, :1]).view(n, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(h).view(n, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
            w = torch.softmax(q @ k.transpose(-1, -2) * attn.head_dim**-0.5, dim=-1)[:, :, 0, 1:].mean(1)
            pixels = self.pixels[-n:].float()
            low = pixels.amin(dim=(2, 3), keepdim=True)
            view = (pixels - low) / (pixels.amax(dim=(2, 3), keepdim=True) - low).clamp_min(1e-6)
        side = pixels.shape[-1] // self.patch
        w = w.float().cpu().numpy().reshape(n, side, side)
        view = (view.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy()
        return {camera: (w[i], view[i]) for i, camera in enumerate(self.cameras)}


def overlay(weights: np.ndarray, view: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8: `view` dimmed where `weights` (grid, grid) is low and tinted magenta where it is
    high (scaled to its maximum), each patch a flat block."""
    scale = view.shape[0] // weights.shape[0]
    heat = np.kron(weights / max(weights.max(), 1e-12), np.ones((scale, scale)))[..., None]
    tint = 0.5 * heat
    out = (1 - tint) * view * (0.3 + 0.7 * heat) + tint * np.array([255.0, 60.0, 255.0])
    return out.clip(0, 255).astype(np.uint8)
