"""Test-time training over cached latents (ttt) on a small randomly initialised VITA: the cache, the
corrector's gate, the update and its trainable sets, the open-loop evaluation, and adapters and
checkpoints loading through the runtime's policy."""

import json

import numpy as np
import pytest
import torch

from inference import lora
from ttt import cache, corrector, evaluate, stats, update

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
