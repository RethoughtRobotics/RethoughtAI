"""Test-time training over cached latents (ttt) on a small randomly initialised VITA: the cache, the
corrector's gate, the update and its trainable sets, the open-loop evaluation, adapters and
checkpoints loading through the runtime's policy, and on-policy corrections from evaluator bags."""

import json

import numpy as np
import pytest
import torch

from inference import lora
from ttt import cache, corrector, evaluate, onpolicy, stats, update

DIM, H, STEPS = 64, 32, 16
CAMERAS = {'observation.images.left': (3, 40, 48), 'observation.images.third_eye': (3, 48, 48)}
EPISODES = {0: 12, 1: 12, 21: 10, 25: 10, 38: 8, 39: 8}


@pytest.fixture(scope='module')
def model(tmp_path_factory):
    from lerobot.configs.types import FeatureType, PolicyFeature

    from inference.vita import VitaConfig, VitaPolicy

    cfg = VitaConfig(
        input_features={
            **{k: PolicyFeature(FeatureType.VISUAL, s) for k, s in CAMERAS.items()},
            'observation.state': PolicyFeature(FeatureType.STATE, (15,)),
        },
        output_features={'action': PolicyFeature(FeatureType.ACTION, (8,))},
        horizon=H,
        n_action_steps=STEPS,
        pretrained_backbone_weights=None,
        resize_shape=(32, 40),
        crop_shape=(28, 36),
        latent_dim=DIM,
        flow_hidden_dim=DIM,
        ae_enc_hidden_dim=DIM,
        ae_dec_hidden_dim=DIM,
        device='cpu',
        push_to_hub=False,
    )
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp('vita')
    VitaPolicy(cfg).save_pretrained(path)
    stats_ = {
        'observation.state': {'mean': [0.1] * 15, 'std': [0.5] * 15},
        'action': {'mean': [0.5, 0.2, 0.1, 0, 1, 0, 0, 0.4], 'std': [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.4]},
    }
    (path / 'normalization_stats.json').write_text(json.dumps(stats_))
    return path


class Frames(torch.utils.data.Dataset):
    """Episodes of EPISODES' lengths: random images and states, a straight demonstrated path whose
    gripper closes in the middle, chunks padded with the last action as LeRobot pads them."""

    def __init__(self):
        g = torch.Generator().manual_seed(3)
        self.items = []
        for e, n in EPISODES.items():
            t = torch.arange(n + H, dtype=torch.float32).clamp(max=n - 1)
            path = torch.stack([0.5 + 0.01 * t + 0.01 * e, 0.2 - 0.005 * t, 0.1 + 0.002 * t], -1)
            grip = ((t >= n // 3) & (t < 2 * n // 3)).float() * 0.87
            actions = torch.cat([path, torch.tensor([0.0, 1.0, 0.0, 0.0]).expand(len(t), 4), grip[:, None]], -1)
            for f in range(n):
                self.items.append({
                    **{k: torch.rand(s, generator=g) for k, s in CAMERAS.items()},
                    'observation.state': torch.randn(15, generator=g),
                    'action': actions[f : f + H],
                    'action_is_pad': torch.arange(f, f + H) >= n,
                    'episode_index': torch.tensor(e),
                    'frame_index': torch.tensor(f),
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@pytest.fixture(scope='module')
def built(model, tmp_path_factory):
    out = tmp_path_factory.mktemp('cache')
    cache.build(model, 'local', None, out, batch_size=7, workers=0, frames=Frames())
    return out


def settings(**kw):
    base = {'steps': 30, 'batch': 16, 'lr': 1e-3, 'pool': (21, 25), 'anchor': (0, 1), 'gate': None}
    return update.Settings(**{**base, **kw})


def test_cache_holds_every_frame(built, model):
    tensors, meta = cache.load(built)
    n = sum(EPISODES.values())
    shapes = {k: tuple(v.shape) for k, v in tensors.items()}
    assert shapes == {
        'feat': (n, 2 * 512 + 15),
        'z0': (n, DIM),
        'z_pi': (n, DIM),
        'z_demo': (n, DIM),
        'chunk_pi': (n, H, 8),
        'chunk_demo': (n, H, 8),
        'pad': (n, H),
        'state': (n, 15),
        'episode': (n,),
        'frame': (n,),
    }
    assert all(tensors[k].dtype == torch.float16 for k in cache.LATENTS)
    assert meta['frames'] == n and meta['episodes'] == sorted(EPISODES) and meta['n_action_steps'] == STEPS

    lp = cache.load_policy(model)
    item = Frames()[n - 1]
    batch = torch.utils.data.default_collate([item])
    with torch.no_grad():
        plan = lp.post(lp.policy.predict_action_chunk(cache.normalise(lp, batch)))[0]
    assert torch.allclose(tensors['chunk_pi'][n - 1, :STEPS], plan, atol=1e-5)
    assert torch.allclose(tensors['chunk_demo'][n - 1], item['action'])
    with torch.no_grad():
        z0 = lp.policy.vita.obs_encoder(tensors['feat'][:4].float())
    assert torch.allclose(z0, tensors['z0'][:4].float(), atol=2e-2)


def test_corrector_gates_on_the_mean_executed_distance():
    n = 6
    demo = torch.zeros(n, H, 8)
    chunk = demo.clone()
    offsets = torch.tensor([0.0, 0.02, 0.029, 0.031, 0.05, 0.2])
    chunk[:, :, 0] = offsets[:, None]
    chunk[5, STEPS:, 0] = 5.0  # beyond the executed steps: ignored
    pad = torch.zeros(n, H, dtype=torch.bool)
    pad[4, 1:] = True  # only the first step is in the episode
    chunk[4, 1:, 0] = 1.0
    tensors = {'chunk_pi': chunk, 'chunk_demo': demo, 'pad': pad, 'episode': torch.tensor([21, 21, 21, 25, 25, 30])}
    assert torch.allclose(corrector.path_error(chunk, demo, pad, STEPS), offsets)
    assert corrector.corrected(tensors, (21, 25), 3.0, STEPS).tolist() == [False, False, False, True, True, False]
    assert corrector.corrected(tensors, (21, 25), 1.0, STEPS).tolist() == [False, True, True, True, True, False]
    assert corrector.corrected(tensors, (21, 25), None, STEPS).tolist() == [True] * 5 + [False]
    s = corrector.summary(tensors, (21, 25, 29), 3.0, STEPS)
    assert s['corrected_frames'] == 2 and s['frames'] == 5 and s['per_episode'] == {21: 0, 25: 2, 29: 0}


def test_update_lowers_the_loss(model, built, tmp_path):
    record = update.run(model, built, tmp_path / 'a', settings(steps=60))
    first, last = record['losses'][0], record['losses'][-1]
    assert last['corrected'] < 0.7 * first['corrected']
    assert record['corrected_frames'] == EPISODES[21] + EPISODES[25]
    assert record['anchor_frames'] == EPISODES[0] + EPISODES[1]
    assert record['trainable_parameters'] == 4 * 16 * (DIM + 4 * DIM) * 2
    assert record['update_s'] > 0 and record['device'].startswith('cpu')
    assert json.loads((tmp_path / 'a' / lora.META).read_text())['root'] == lora.FLOW_NET


def anchor_error(lp, built, episodes=(0, 1)):
    tensors, _ = cache.load(built)
    rows = corrector.in_episodes(tensors, episodes)
    vita = lp.policy.vita
    torch.manual_seed(0)
    t = torch.rand(int(rows.sum()))
    with torch.no_grad():
        z0 = vita.obs_encoder(tensors['feat'][rows].float())
        return update.pair_error(vita, z0, tensors['z_demo'][rows].float(), t, 'flow').mean().item()


def test_the_anchor_holds_the_base_pairs(model, built, tmp_path):
    errors = {}
    for lam in (0.0, 0.5):
        record = update.run(model, built, tmp_path / str(lam), settings(lam=lam, steps=80, lr=3e-3))
        assert ('anchor' in record['losses'][-1]) == (lam > 0)
        lp = cache.load_policy(model)
        lora.load_policy(lp.policy, tmp_path / str(lam))
        errors[lam] = anchor_error(lp, built)
    base = anchor_error(cache.load_policy(model), built)
    assert errors[0.5] < errors[0.0]
    assert errors[0.5] < base * 1.5


def test_alpha_zero_is_the_base(model, built, tmp_path):
    update.run(model, built, tmp_path / 'a', settings())
    results = evaluate.run(built, tmp_path / 'r.json', adapter=tmp_path / 'a', alphas=(0.0, 1.0), n_boot=200)
    zero, one = results['alphas']['0.0'], results['alphas']['1.0']
    assert zero['test'] == results['base']['test'] and zero['forget'] == results['base']['forget']
    assert zero['summary']['all']['minus_base_mm']['value'] == 0
    assert one['test'] != results['base']['test']
    assert set(results['base']['test']) == {38, 39} and set(results['base']['forget']) == set(range(21)) & set(EPISODES)
    assert json.loads((tmp_path / 'r.json').read_text())['alphas'].keys() == {'0.0', '1.0'}


@pytest.mark.parametrize('trainable', ['lora', 'lora_obs', 'lora_dec', 'lora_all'])
def test_adapters_load_through_the_runtime(model, built, tmp_path, trainable):
    from inference.policies import LeRobotPolicy

    update.run(model, built, tmp_path / 'a', settings(trainable=trainable, steps=10))
    meta = json.loads((tmp_path / 'a' / lora.META).read_text())
    assert meta['root'] == (lora.FLOW_NET if trainable == 'lora' else 'vita')
    tensors, _ = cache.load(built)
    rows = torch.arange(len(tensors['episode']))
    trained = cache.load_policy(model)
    lora.load_policy(trained.policy, tmp_path / 'a')
    runtime = LeRobotPolicy(str(model), 'cpu', adapter=tmp_path / 'a', alpha=1.0)
    base = LeRobotPolicy(str(model), 'cpu', adapter=tmp_path / 'a', alpha=0.0)
    plans = evaluate.decode(runtime, tensors, rows)
    assert torch.equal(plans, evaluate.decode(trained, tensors, rows))
    assert not torch.allclose(plans, evaluate.decode(base, tensors, rows))
    assert torch.equal(evaluate.decode(base, tensors, rows), evaluate.decode(cache.load_policy(model), tensors, rows))
    sites = {
        'lora': ['flow_net'],
        'lora_obs': ['obs_encoder'],
        'lora_dec': ['action_decoder'],
        'lora_all': ['obs_encoder', 'flow_net', 'action_decoder'],
    }[trainable]
    wrapped = {name.split('.')[1] for name in lora.adapters(runtime.policy)}
    assert wrapped == set(sites)


def test_adapter_without_root_is_on_the_flow_net(model, tmp_path):
    lp = cache.load_policy(model)
    lora.wrap(lora.flow_net(lp.policy), lora.targets(lora.flow_net(lp.policy)), rank=4)
    lora.save(lora.flow_net(lp.policy), tmp_path)
    assert 'root' not in json.loads((tmp_path / lora.META).read_text())
    fresh = cache.load_policy(model)
    assert len(lora.load_policy(fresh.policy, tmp_path)) == 8


@pytest.mark.parametrize('trainable,changed', [('flow', {'flow_net'}), ('flow_last', {'flow_net.out_proj'})])
def test_full_updates_save_a_runtime_checkpoint(model, built, tmp_path, trainable, changed):
    from inference.policies import LeRobotPolicy

    update.run(model, built, tmp_path / 'c', settings(trainable=trainable, steps=10))
    assert {f.name for f in (tmp_path / 'c').iterdir()} >= {'config.json', 'model.safetensors', 'update.json'}
    base, new = cache.load_policy(model).policy.state_dict(), LeRobotPolicy(str(tmp_path / 'c'), 'cpu')
    diff = {k for k, v in new.policy.state_dict().items() if not torch.equal(v, base[k])}
    assert diff and all(any(k.startswith(f'vita.{c}.') for c in changed) for k in diff)
    results = evaluate.run(built, tmp_path / 'r.json', checkpoint=tmp_path / 'c', alphas=(0.0, 1.0), n_boot=200)
    assert results['alphas']['0.0']['test'] == results['base']['test']
    assert results['alphas']['1.0']['test'] != results['base']['test']


def test_everything_trains_from_frames(model, built, tmp_path):
    update.run(model, built, tmp_path / 'c', settings(trainable='all', steps=3), frames=Frames(), workers=0)
    with pytest.raises(ValueError, match='observer'):
        evaluate.run(built, tmp_path / 'r.json', checkpoint=tmp_path / 'c')
    plans = tmp_path / 'plans'
    cache.build(tmp_path / 'c', 'local', None, plans, workers=0, frames=Frames())
    results = evaluate.run(built, tmp_path / 'r.json', plans=plans, n_boot=200)
    assert results['alphas']['1.0']['test'] != results['base']['test']


def test_preference_starts_at_log_two(model, built, tmp_path):
    record = update.run(model, built, tmp_path / 'a', settings(beta=10.0, steps=5))
    assert record['losses'][0]['pref'] == pytest.approx(np.log(2), abs=1e-6)
    assert all(np.isfinite(r['total']) for r in record['losses'])


def test_the_update_is_seeded(model, built, tmp_path):
    a = update.run(model, built, tmp_path / 'a', settings(steps=5, seed=1))['losses']
    b = update.run(model, built, tmp_path / 'b', settings(steps=5, seed=1))['losses']
    c = update.run(model, built, tmp_path / 'c', settings(steps=5, seed=2))['losses']
    assert a == b and a != c


def test_phases_from_the_gripper():
    closed = torch.tensor([0, 0, 0.9, 0.9, 0.9, 0, 0, 0, 0.9, 0])
    demo = torch.zeros(10, H, 8)
    demo[:, 0, 7] = closed
    tensors = {
        'chunk_demo': demo,
        'episode': torch.tensor([5] * 6 + [6] * 4),
        'frame': torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3]),
    }
    assert evaluate.phases(tensors).tolist() == [0, 0, 1, 1, 1, 2, 0, 0, 1, 2]


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    base = {e: float(v) for e, v in enumerate(rng.normal(10, 1, 18))}
    better = {e: v - 2 + 0.1 * rng.normal() for e, v in base.items()}
    same = {e: v + 0.5 * rng.normal() for e, v in base.items()}
    d = stats.paired(better, base, n=2000)
    assert d['excludes_zero'] and d['lo'] <= d['value'] <= d['hi'] and d['units'] == 18
    assert not stats.paired(same, base, n=2000)['excludes_zero']
    assert stats.paired({1: 1.0, 2: None}, {1: 0.0, 2: 0.0}, n=100)['units'] == 1


# On-policy corrections: evaluator bags of two attempts, cameras every 40 ms and the arm every 10 ms,
# with an expert.npz of FRAMES frames 5 ms after each camera message; frame NO_PLAN without a plan
T0, FRAMES, LEAD, NO_PLAN = 1_000_000_000, 10, 2, 3
SEEDS = (1000, 1001)
IMAGES = {'left': (30, 40), 'third_eye': (48, 48)}  # the left camera letterboxed into the policy's 40 x 48


def expert_frame(seed, i):
    """What frame i of attempt `seed` sees: (images, joints, TCP, gripper), the camera and joint
    messages of step 4 i."""
    images = {
        key: np.full((*shape, 3), (4 * i + 50 * k + seed) % 256, np.uint8)
        for k, (key, shape) in enumerate(IMAGES.items())
    }
    joints = 0.01 * 4 * i + 0.1 * np.arange(7)
    tcp = (np.array([0.5 + 0.01 * i, 0.2, 0.1 + 0.001 * (seed % 7)]), np.array([0.0, 1.0, 0.0, 0.0]))
    return images, joints, tcp, i / FRAMES


def evaluator_bag(path, seed):
    """An evaluator bag (cameras, joints, gripper) and its expert.npz: the planner's chunks a straight
    path from the TCP, its gripper closing."""
    import rosbag2_py
    from baxter_core_msgs.msg import EndEffectorState
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Image, JointState

    from adapter.lerobot_export import JOINTS

    topics = {
        '/cameras/left_hand_camera/image': 'sensor_msgs/msg/Image',
        '/cameras/third_eye/color/image_raw': 'sensor_msgs/msg/Image',
        '/robot/joint_states': 'sensor_msgs/msg/JointState',
        '/robot/end_effector/left_gripper/state': 'baxter_core_msgs/msg/EndEffectorState',
    }
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    for i, (topic, kind) in enumerate(topics.items()):
        writer.create_topic(rosbag2_py.TopicMetadata(id=i, name=topic, type=kind, serialization_format='cdr'))
    for k in range(4 * FRAMES):
        t = T0 + k * 10_000_000
        images, joints, _, gripper = expert_frame(seed, k // 4)
        if k % 4 == 0:
            for topic, key in zip(list(topics)[:2], IMAGES, strict=True):
                h, w = IMAGES[key]
                msg = Image(height=h, width=w, encoding='rgb8', step=3 * w, data=images[key].tobytes())
                writer.write(topic, serialize_message(msg), t)
        msg = JointState(name=[f'left_{j}' for j in JOINTS], position=list(joints + 0.001 * (k % 4)))
        writer.write('/robot/joint_states', serialize_message(msg), t)
        writer.write('/robot/end_effector/left_gripper/state', serialize_message(EndEffectorState(position=50.0)), t)
    del writer

    frames = [expert_frame(seed, i) for i in range(FRAMES)]
    tcp = np.array([np.r_[f[2][0], f[2][1]] for f in frames])
    steps = np.arange(H + LEAD)
    chunk = np.zeros((FRAMES, H + LEAD, 8))
    chunk[:, :, :3] = tcp[:, None, :3] + 0.002 * steps[:, None] * np.array([1.0, -1.0, 0.0])
    chunk[:, :, 3:7] = tcp[:, None, 3:7]
    chunk[:, :, 7] = np.clip(steps / H, 0, 0.87)
    valid = np.arange(FRAMES) != NO_PLAN
    chunk[~valid] = np.nan
    np.savez_compressed(
        path / onpolicy.EXPERT,
        meta=json.dumps({'fps': 25.0, 'horizon': H + LEAD, 'arm': 'left', 'task': 'cup'}),
        t=T0 + 5_000_000 + 40_000_000 * np.arange(FRAMES, dtype=np.int64),
        tcp=tcp,
        gripper=np.array([f[3] for f in frames]),
        cup=np.zeros((FRAMES, 7)),
        container=np.zeros((FRAMES, 7)),
        chunk=chunk,
        valid=valid,
        home=tcp[0],
    )
    return path


@pytest.fixture(scope='module')
def attempts(model, tmp_path_factory):
    root = tmp_path_factory.mktemp('bags')
    bags = [evaluator_bag(root / f'seed_{s}', s) for s in SEEDS]
    out = tmp_path_factory.mktemp('onpolicy')
    onpolicy.build(model, bags, out, batch_size=4, lead=LEAD)
    return bags, out


def test_planner_labels_lead_the_pose_not_the_gripper():
    chunk = np.arange(2 * 6 * 8, dtype=float).reshape(2, 6, 8)
    labels = onpolicy.relabel(chunk, 4, 2)
    assert np.array_equal(labels[..., :7], chunk[:, 2:6, :7]) and np.array_equal(labels[..., 7], chunk[:, :4, 7])
    with pytest.raises(ValueError, match='--horizon'):
        onpolicy.relabel(chunk, 5, 2)
    assert onpolicy.seed('bags/seed_1004') == 1004 and onpolicy.seed('bags/attempt') is None


def test_onpolicy_cache_from_evaluator_bags(model, attempts):
    bags, out = attempts
    tensors, meta = cache.load(out)
    planned = [i for i in range(FRAMES) if i != NO_PLAN]
    n = len(SEEDS) * len(planned)
    assert {k: tuple(v.shape) for k, v in tensors.items()} == {
        'feat': (n, 2 * 512 + 15),
        'z0': (n, DIM),
        'z_pi': (n, DIM),
        'z_demo': (n, DIM),
        'chunk_pi': (n, H, 8),
        'chunk_demo': (n, H, 8),
        'pad': (n, H),
        'state': (n, 15),
        'episode': (n,),
        'frame': (n,),
    }
    assert tensors['episode'].tolist() == [s for s in SEEDS for _ in planned]
    assert tensors['frame'].tolist() == planned * len(SEEDS) and not tensors['pad'].any()
    assert meta['source'] == 'on-policy' and meta['episodes'] == list(SEEDS) and meta['frames_without_plan'] == 2
    assert meta['lead_steps'] == LEAD and meta['cameras'] == {
        '/cameras/left_hand_camera/image': 'left',
        '/cameras/third_eye/color/image_raw': 'third_eye',
    }
    assert meta['model'] == str(model.resolve())

    e = onpolicy.load_expert(bags[1])
    row = len(planned) + 4  # seed 1001, frame 5
    i = planned[4]
    assert torch.allclose(tensors['chunk_demo'][row], torch.as_tensor(onpolicy.relabel(e['chunk'][i], H, LEAD)).float())
    lp = cache.load_policy(model)
    images, joints, tcp, gripper = expert_frame(SEEDS[1], i)
    item = onpolicy.item(lp, images, joints, tcp, gripper, onpolicy.relabel(e['chunk'][i], H, LEAD), SEEDS[1], i)
    expected = cache.rows(lp, torch.utils.data.default_collate([item]))
    for k in ('chunk_pi', 'state'):
        assert torch.allclose(tensors[k][row].float(), expected[k][0].float(), atol=1e-4)
    assert torch.allclose(tensors['z0'][row].float(), expected['z0'][0].float(), atol=2e-2)


def test_corrected_frames_export_as_a_lerobot_dataset(built, attempts, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _, onp = attempts
    tensors, meta = cache.load(onp)
    error = corrector.path_error(tensors['chunk_pi'], tensors['chunk_demo'], tensors['pad'], STEPS)
    gate = float(error.median()) * 100  # about half the frames
    mask = corrector.corrected(tensors, [SEEDS[0]], gate, STEPS)
    root = tmp_path / 'ds'
    n = onpolicy.export(
        onp, root, 'local/corrections', 'Pick up the cup', gate=gate, pool=[SEEDS[0]], log=lambda _: None
    )
    assert n == int(mask.sum()) > 0
    ds = LeRobotDataset('local/corrections', root=root)
    cols = ds.hf_dataset.select_columns(['episode_index', 'frame_index', 'action', 'observation.state'])
    assert ds.meta.total_episodes == n and len(cols) == n * H
    sources = json.loads((root / onpolicy.SOURCES).read_text())
    assert [e['frame'] for e in sources['episodes']] == tensors['frame'][mask].tolist()
    assert all(e['seed'] == SEEDS[0] and e['path_error_cm'] > gate for e in sources['episodes'])
    actions = torch.stack([torch.as_tensor(a) for a in cols['action']]).reshape(n, H, 8)
    states = torch.stack([torch.as_tensor(a) for a in cols['observation.state']]).reshape(n, H, 15)
    assert torch.allclose(actions, tensors['chunk_demo'][mask])
    assert torch.allclose(states[:, 0], tensors['state'][mask]) and (states == states[:, :1]).all()
    with pytest.raises(ValueError, match='not an on-policy cache'):
        onpolicy.export(built, tmp_path / 'x', 'local/x', 'Pick up the cup')
