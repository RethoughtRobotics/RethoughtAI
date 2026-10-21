"""Deployment contracts for raw-pose VITA; no weights, ROS nodes or robot commands."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from click.testing import CliRunner
from scipy.spatial.transform import Rotation

from adapter.labels import apply_actions
from inference.cli import deploy_task, main, policy_settings
from inference.policies import LeRobotPolicy, action_layout, chunk_length, raw_state, trained_tasks
from inference.runtime import check_merge


def config(state=15, action=8):
    return {
        'type': 'vita',
        'input_features': {'observation.state': {'type': 'STATE', 'shape': [state]}},
        'output_features': {'action': {'type': 'ACTION', 'shape': [action]}},
        'n_action_steps': 16,
        'horizon': 32,
    }


def settings():
    path = Path(__file__).resolve().parents[1] / 'configs/baxter.yaml'
    return yaml.safe_load(path.read_text())['inference']


def test_absolute_vita_has_separate_schedule_and_full_latency_coverage():
    cfg = config()
    s = policy_settings(settings(), cfg)
    effective = {**cfg, **s['overrides']}
    assert cfg['n_action_steps'] == 16  # never rewrite the downloaded training config
    assert chunk_length(effective) == 32 and s['every'] == 16
    assert s['merge'] == {'type': 'latest'} and not s.get('rtc')
    assert check_merge(s['merge'], chunk_length(effective), s['every']) is None


def test_checkpoints_trained_on_lead_labels_get_no_run_time_lead():
    cfg = settings()
    cfg['policies']['vita_absolute']['merge'] = {'type': 'latest', 'lead': 0.4}
    s = policy_settings(cfg, config(), 'andnetdeboer/BlueCupRedContainer_kin_lead640ms')
    assert s['merge'] == {'type': 'latest'}
    assert policy_settings(cfg, config())['merge']['lead'] == 0.4  # the config is not changed


def test_prompt_is_the_task_the_checkpoint_was_trained_on(tmp_path):
    trained = ('Sweep the trash into the dustpan and empty it into the bin',)
    assert deploy_task('Sweep the trash', trained) == trained[0]  # a stale config can't override it
    assert deploy_task(trained[0], trained) == trained[0]
    assert deploy_task('Pick up the cup', ()) == 'Pick up the cup'  # unknown dataset tasks: the config's
    assert deploy_task('b', ('a', 'b')) == 'b'
    with pytest.raises(ValueError, match='configure one of'):
        deploy_task('c', ('a', 'b'))
    assert trained_tasks(tmp_path) == ()  # a checkpoint without train_config.json


def test_old_relative_vita_schedule_is_unchanged():
    s = policy_settings(settings(), config(17, 7))
    assert s['every'] == 1 and s['merge'] == {'type': 'ensemble', 'coeff': -0.05}
    assert not s.get('overrides')


@pytest.mark.parametrize('dims, expected', [((17, 7), 'relative'), ((15, 8), 'absolute')])
def test_layout_detection(dims, expected):
    assert action_layout(config(*dims)) == expected


def bimanual_settings():
    path = Path(__file__).resolve().parents[1] / 'configs/baxter_bimanual.yaml'
    return yaml.safe_load(path.read_text())['inference']


def test_sweeptrash_uses_coordinated_absolute_schedule():
    cfg = config(28, 14)
    assert action_layout(cfg) == 'bimanual_absolute'
    s = policy_settings(bimanual_settings(), cfg)
    assert s['every'] == 16 and s['overrides'] == {'n_action_steps': 32}
    assert s['merge'] == {'type': 'latest'} and not s.get('rtc')


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_bimanual_config_requires_complete_arm_topics(arm):
    cfg = bimanual_settings()
    del cfg['arms'][arm]['tcp']
    with pytest.raises(ValueError, match='topics'):
        policy_settings(cfg, config(28, 14))


def test_single_arm_model_cannot_run_on_bimanual_config():
    with pytest.raises(ValueError, match='arm: both requires'):
        policy_settings(bimanual_settings(), config())


def test_sweeptrash_is_rejected_before_a_single_arm_runtime_can_start():
    with pytest.raises(ValueError, match='bimanual absolute poses without grippers'):
        policy_settings(settings(), config(28, 14))


def test_raw_state_matches_export_order_and_quaternion_hemisphere():
    q = Rotation.from_euler('y', 160, degrees=True).as_quat()
    expected = np.r_[[0.6, 0.2, 0.1], q, 0.3, np.arange(7)]
    assert raw_state([0.6, 0.2, 0.1], -q, 0.3, np.arange(7)) == pytest.approx(expected)


def test_absolute_policy_roundtrip_does_not_add_tcp_or_lead():
    """Exercise the actual wrapper and the policy node's inverse conversion."""
    tcp = (np.array([0.6, 0.2, 0.1]), Rotation.from_euler('xyz', [0.2, 0.5, -0.3]).as_quat())
    positions = np.array([[0.62, 0.19, 0.12], [0.64, 0.18, 0.13]])
    rotations = Rotation.from_euler('xyz', [[0.1, 0.4, 0.3], [0.2, 0.6, 0.4]])
    absolute = np.column_stack([positions, rotations.as_quat(), [0.2, 0.8]])
    p = LeRobotPolicy.__new__(LeRobotPolicy)
    p.absolute, p.rtc, p.type = True, None, 'vita'
    p.torch, p.device, p.task = torch, torch.device('cpu'), ''
    p.features, p.image_shapes = ['observation.state'], {}
    p.pre, p.post = lambda batch: batch, lambda actions: actions
    p.policy = SimpleNamespace(predict_action_chunk=lambda batch: torch.tensor(absolute)[None])
    relative = p({'observation.state': np.zeros(15, np.float32)}, tcp=tcp)
    actual_p, actual_R = apply_actions(relative, *tcp)
    assert actual_p == pytest.approx(positions)
    assert (actual_R.inv() * rotations).magnitude() == pytest.approx(np.zeros(2), abs=1e-7)
    assert relative[:, 6] == pytest.approx([0.2, 0.8])


def test_no_motion_check_reports_supported_and_rejects_bimanual(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    cfg_path = Path(__file__).resolve().parents[1] / 'configs/baxter.yaml'
    runner = CliRunner()
    (model / 'config.json').write_text(json.dumps(config()))
    result = runner.invoke(main, ['check', str(model), '--config', str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert '32 targets, predict every 16' in result.output and 'pose lead 0 s, gripper lead 0 s' in result.output
    (model / 'config.json').write_text(json.dumps(config(28, 14)))
    result = runner.invoke(main, ['check', str(model), '--config', str(cfg_path)])
    assert result.exit_code != 0 and 'bimanual' in result.output
    cfg_path = cfg_path.with_name('baxter_bimanual.yaml')
    result = runner.invoke(main, ['check', str(model), '--config', str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert 'bimanual_absolute poses, arm both' in result.output


def test_bimanual_wrapper_preserves_raw_absolute_actions_and_state_order():
    p = LeRobotPolicy.__new__(LeRobotPolicy)
    p.absolute, p.bimanual, p.rtc, p.type = True, True, None, 'vita'
    p.torch, p.device, p.task = torch, torch.device('cpu'), ''
    p.features, p.image_shapes = ['observation.state'], {}
    p.pre, p.post = lambda batch: batch, lambda actions: actions
    actions = np.tile(
        np.r_[[0.6, 0.2, 0.1, 0.0, -0.5, 0.0, 0.8660254], [0.6, -0.2, 0.1, 0.0, 0.5, 0.0, 0.8660254]], (32, 1)
    )
    p.policy = SimpleNamespace(predict_action_chunk=lambda batch: torch.tensor(actions)[None])
    tcp_p = {'left': actions[0, :3], 'right': actions[0, 7:10]}
    tcp_q = {'left': actions[0, 3:7], 'right': actions[0, 10:14]}
    joints = {'left': np.arange(7), 'right': np.arange(7) + 10}
    state = p.state(tcp_p, tcp_q, None, joints)
    assert state == pytest.approx(
        np.r_[tcp_p['left'], tcp_q['left'], joints['left'], tcp_p['right'], tcp_q['right'], joints['right']]
    )
    assert state.dtype == np.float32
    assert p({'observation.state': state}) == pytest.approx(actions)


def test_a_replay_as_a_checkpoint_runs_with_its_chunks_schedule_merge_and_layout():
    from inference.cli import replay_as

    model_cfg = config()  # absolute VITA: 32-step chunks, a new one every 16 steps, latest
    cfg, merge, absolute = replay_as(settings(), model_cfg, policy_settings(settings(), model_cfg))
    assert cfg['chunk_size'] == 32 and cfg['n_action_steps'] == 16 and merge == {'type': 'latest'} and absolute
    relative = config(17, 7)
    assert not replay_as(settings(), relative, policy_settings(settings(), relative))[2]
