"""Inspect launcher commands only; never execute ROS, tmux, or robot commands."""

import shlex
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / 'baxter_sdk' / 'scripts' / 'run_teleop.sh'
SAFETY_BOUNDARY = '# Two teleops on one arm fight'


def render_commands(tmp_path, *args):
    source = LAUNCHER.read_text()
    assert source.count(SAFETY_BOUNDARY) == 1
    # Stop before process checks, camera startup, and either terminal backend.
    prefix = source.split(SAFETY_BOUNDARY, 1)[0]
    assert 'setsid' not in prefix and 'tmux' not in prefix.split('set -e', 1)[1]
    script = tmp_path / 'baxter_sdk' / 'scripts' / 'run_teleop.sh'
    script.parent.mkdir(parents=True)
    (tmp_path / 'rethought_ai').mkdir()
    script.write_text(
        prefix
        + """
for arm in "${ARM_LIST[@]}"; do
    echo '===TELEOP==='
    teleop_cmd "${arm/given/}"
done
echo '===RECORDER==='
record_cmd
"""
    )
    return subprocess.run(
        ['bash', str(script), *args],
        capture_output=True,
        text=True,
        timeout=5,
    )


def teleop_arguments(result):
    assert result.returncode == 0, result.stderr
    teleops, _ = result.stdout.split('===RECORDER===', 1)
    return [
        shlex.split(command.split('-m baxter_teleop ', 1)[1].split(';', 1)[0])
        for command in teleops.split('===TELEOP===')[1:]
    ]


def option(arguments, name):
    # argparse uses the final occurrence when an option overrides a default.
    index = len(arguments) - 1 - arguments[::-1].index(name)
    return arguments[index + 1]


@pytest.mark.parametrize('selection', [(), ('--both',)])
def test_dual_policy_uses_each_arms_own_topics(tmp_path, selection):
    commands = teleop_arguments(render_commands(tmp_path, *selection, '--policy', '--no-gripper'))
    assert len(commands) == 2
    for arguments, arm in zip(commands, ('left', 'right')):
        assert option(arguments, '--arm') == arm
        assert option(arguments, '--input') == 'plan'  # the replay stream plays policies
        assert option(arguments, '--vr-config') == 'vr_axes_policy.yaml'
        assert option(arguments, '--vr-trigger-topic') == f'/policy/{arm}/trigger'
        assert '--vr-topic' not in arguments
        assert arguments.count('--vr-trigger-topic') == 1
        assert arguments.count('--no-gripper') == 1
        other = 'right' if arm == 'left' else 'left'
        assert not any(f'/policy/{other}/' in value for value in arguments)


@pytest.mark.parametrize('arm', ['left', 'right'])
@pytest.mark.parametrize('form', ['selector', 'separate', 'equals'])
@pytest.mark.parametrize('policy', [False, True])
def test_single_arm_and_recorder_selection(tmp_path, arm, form, policy):
    selection = {
        'selector': [f'--{arm}'],
        'separate': ['--arm', arm],
        'equals': [f'--arm={arm}'],
    }[form]
    result = render_commands(tmp_path, *selection, *(['--policy'] if policy else []), '--no-gripper')
    commands = teleop_arguments(result)
    assert len(commands) == 1
    arguments = commands[0]
    assert option(arguments, '--arm') == arm
    assert '--no-gripper' in arguments
    if policy:
        assert option(arguments, '--input') == 'plan'
        assert option(arguments, '--vr-trigger-topic') == f'/policy/{arm}/trigger'
        assert option(arguments, '--vr-config') == 'vr_axes_policy.yaml'
    else:
        assert '--vr-topic' not in arguments
        assert '--vr-trigger-topic' not in arguments
        assert '--vr-config' not in arguments  # the teleop's default, from teleop.yaml
    other = 'right' if arm == 'left' else 'left'
    recorder = result.stdout.split('===RECORDER===', 1)[1]
    assert f'--exclude {other}' in recorder


@pytest.mark.parametrize('selection', [(), ('--both',)])
def test_dual_vr_is_unchanged(tmp_path, selection):
    result = render_commands(tmp_path, *selection, '--no-gripper', '--linear-speed', '0.15')
    commands = teleop_arguments(result)
    assert [option(arguments, '--arm') for arguments in commands] == ['left', 'right']
    for arguments in commands:
        assert '--vr-config' not in arguments  # the teleop's default, from teleop.yaml
        assert '--vr-topic' not in arguments
        assert '--vr-trigger-topic' not in arguments
        assert '--no-gripper' in arguments
        assert option(arguments, '--linear-speed') == '0.15'
    assert '--exclude' not in result.stdout.split('===RECORDER===', 1)[1]


def test_named_selector_still_overrides_passthrough_arm(tmp_path):
    commands = teleop_arguments(render_commands(tmp_path, '--arm', 'right', '--both', '--policy'))
    assert [option(arguments, '--arm') for arguments in commands] == ['left', 'right']
    for arguments, arm in zip(commands, ('left', 'right')):
        assert option(arguments, '--vr-trigger-topic') == f'/policy/{arm}/trigger'


def test_last_passthrough_arm_wins(tmp_path):
    result = render_commands(tmp_path, '--arm', 'left', '--arm=right', '--policy')
    (arguments,) = teleop_arguments(result)
    assert option(arguments, '--vr-trigger-topic') == '/policy/right/trigger'
    assert '--exclude left' in result.stdout.split('===RECORDER===', 1)[1]


@pytest.mark.parametrize(
    'selection',
    [
        ('--arm',),
        ('--arm=',),
        ('--arm', 'middle'),
        ('--arm=middle',),
        ('--arm', '--policy'),
        ('--both', '--arm', 'middle'),
    ],
)
@pytest.mark.parametrize('policy', [False, True])
def test_invalid_arm_fails_before_command_generation(tmp_path, selection, policy):
    result = render_commands(tmp_path, *(['--policy'] if policy else []), *selection)
    assert result.returncode != 0
    assert '--arm' in result.stderr
    assert '===TELEOP===' not in result.stdout


def test_launcher_bash_syntax():
    result = subprocess.run(['bash', '-n', str(LAUNCHER)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
