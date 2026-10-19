"""Shell-control regressions: sandboxed files and stubs, never ROS or a robot."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_EXIT = 73
BOTH_ENGAGE = '/policy/both/engage'
LEFT_ENGAGE = '/policy/left/engage'
LEFT_HOME = '/vive/right/b_button'
RIGHT_HOME = '/vive/left/b_button'


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/bash\n' + text)
    path.chmod(0o755)


@pytest.fixture
def sandbox(tmp_path):
    """Exclude inherited ROS paths, shell startup hooks, and real control tools."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for command in ('dirname', 'sort', 'sed'):
        program = shutil.which(command)
        assert program is not None, f'Required shell utility missing: {command}'
        (bin_dir / command).symlink_to(program)
    (bin_dir / 'python3').symlink_to(sys.executable)
    log = tmp_path / 'calls.tsv'
    log.touch()
    env = {
        'PATH': str(bin_dir),
        'HOME': str(tmp_path),
        'LC_ALL': 'C',
        'CALL_LOG': str(log),
    }
    for command in ('ros2', 'timeout', 'tmux', 'pgrep'):
        executable(
            bin_dir / command,
            f'printf "unexpected\\t{command}\\n" >> "$CALL_LOG"\nexit 99\n',
        )
    return tmp_path, env, log


def run_script(root, env, name, *args):
    return subprocess.run(
        ['/bin/bash', str(root / 'scripts' / name), *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def calls(log):
    return [line.split('\t') for line in log.read_text().splitlines()]


@pytest.mark.parametrize(
    'model,args,expected_config',
    [
        ('vita_SweepTrash_Kin', [], 'configs/baxter_bimanual.yaml'),
        ('vita_SweepTrash_Kin', ['--config', 'configs/baxter.yaml'], 'configs/baxter.yaml'),
        ('vita_SweepTrash_Kin', ['--config', 'configs/custom controls.yaml'], 'configs/custom controls.yaml'),
        ('vita_demo', [], 'configs/baxter.yaml'),
    ],
)
def test_launcher_config_selected_before_env_and_preflight(sandbox, model, args, expected_config):
    root, env, log = sandbox
    (root / 'scripts').mkdir()
    shutil.copyfile(ROOT / 'scripts' / 'run_inference', root / 'scripts' / 'run_inference')
    checkpoint = root / 'models' / model / 'step-050000'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text('{}')
    launch = root / 'launch'
    launch.mkdir()
    (launch / 'env.sh').write_text('printf "env\\t%s\\n" "$CONFIG" >> "$CALL_LOG"\n')
    executable(
        root / '.venv' / 'bin' / 'inference',
        'printf "preflight\\t%s" "$CONFIG" >> "$CALL_LOG"\n'
        'printf "\\t%s" "$@" >> "$CALL_LOG"\n'
        'printf "\\n" >> "$CALL_LOG"\n'
        f'exit {PREFLIGHT_EXIT}\n',
    )

    result = run_script(root, env, 'run_inference', f'--{model}', '50000', *args)

    assert result.returncode == PREFLIGHT_EXIT, result.stderr
    assert calls(log) == [
        ['env', expected_config],
        ['preflight', expected_config, 'check', f'models/{model}/step-050000', '--config', expected_config],
    ]


@pytest.mark.parametrize(
    'config,expected',
    [
        (None, [LEFT_ENGAGE, LEFT_HOME, '']),
        ('baxter.yaml', [LEFT_ENGAGE, LEFT_HOME, '']),
        ('baxter_bimanual.yaml', [BOTH_ENGAGE, LEFT_HOME, RIGHT_HOME]),
        ('duplicate.yaml', [BOTH_ENGAGE, LEFT_HOME, '']),
    ],
)
def test_actual_env_config_extraction_without_ros_setup(sandbox, config, expected):
    root, env, log = sandbox
    launch = root / 'launch'
    launch.mkdir()
    configs = root / 'configs'
    configs.mkdir()
    for name in ('baxter.yaml', 'baxter_bimanual.yaml'):
        shutil.copyfile(ROOT / 'configs' / name, configs / name)
    (configs / 'duplicate.yaml').write_text(
        'inference:\n'
        '  arm: both\n'
        f'  engage: {BOTH_ENGAGE}\n'
        '  arms:\n'
        f'    left: {{home: {LEFT_HOME}}}\n'
        f'    right: {{home: {LEFT_HOME}}}\n'
    )
    # Source ONLY the real read statement and its continuation, not ROS setup.
    lines = (ROOT / 'launch' / 'env.sh').read_text().splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line.startswith('read -r ENGAGE '))
    statement = lines[index]
    while statement.rstrip().endswith('\\'):
        index += 1
        statement += lines[index]
    extraction = launch / 'extract.sh'
    extraction.write_text(statement)
    if config is not None:
        env['CONFIG'] = str(configs / config)

    result = subprocess.run(
        [
            '/bin/bash',
            '-c',
            'source "$1"; printf "%s\\t%s\\t%s\\n" "$ENGAGE" "$HOME_TOPIC" "$HOME_TOPIC_OTHER"',
            'extract-controls',
            str(extraction),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ''
    assert result.stdout.rstrip('\n').split('\t') == expected
    assert calls(log) == []


def control_sandbox(sandbox, *, both=True, stop_fails=False):
    root, env, log = sandbox
    (root / 'scripts').mkdir()
    for name in ('home', 'start', 'stop'):
        shutil.copyfile(ROOT / 'scripts' / name, root / 'scripts' / name)
    launch = root / 'launch'
    launch.mkdir()
    (launch / 'env.sh').write_text(
        f'ENGAGE={BOTH_ENGAGE if both else LEFT_ENGAGE}\n'
        f'HOME_TOPIC={LEFT_HOME}\n'
        f'HOME_TOPIC_OTHER={RIGHT_HOME if both else ""}\n'
    )
    env['STOP_FAILS'] = '1' if stop_fails else '0'
    executable(
        root / 'bin' / 'timeout',
        'printf "timeout" >> "$CALL_LOG"\n'
        'printf "\\t%s" "$@" >> "$CALL_LOG"\n'
        'printf "\\n" >> "$CALL_LOG"\n'
        '[[ "$1" == 10 && "$2" == ros2 ]] || exit 99\n'
        'shift\nexec "$@"\n',
    )
    executable(
        root / 'bin' / 'ros2',
        'printf "ros2" >> "$CALL_LOG"\n'
        'printf "\\t%s" "$@" >> "$CALL_LOG"\n'
        'printf "\\n" >> "$CALL_LOG"\n'
        'case "$1 $2" in\n'
        '  "service call")\n'
        '    [[ "$STOP_FAILS" == 1 && "$5" == "{data: false}" ]] && exit 1\n'
        '    exit 0 ;;\n'
        '  "topic info") printf "Subscription count: 2\\n" ;;\n'
        '  "topic pub") exit 0 ;;\n'
        '  *) exit 99 ;;\n'
        'esac\n',
    )
    return root, env, log


def assert_control_calls(log, expected):
    """Every request must pass through the fake timeout to the fake ROS CLI."""
    recorded = calls(log)
    assert recorded == [entry for args in expected for entry in (['timeout', '10', 'ros2', *args], ['ros2', *args])]


def home_calls(topic):
    return [
        ['topic', 'info', topic],
        ['topic', 'pub', '--once', '-w', '2', topic, 'std_msgs/msg/Empty', '{}'],
    ]


def test_two_arm_home_releases_before_publishing_both_topics(sandbox):
    root, env, log = control_sandbox(sandbox)

    result = run_script(root, env, 'home')

    assert result.returncode == 0, result.stderr
    assert_control_calls(
        log,
        [
            ['service', 'call', BOTH_ENGAGE, 'std_srvs/srv/SetBool', '{data: false}'],
            *home_calls(LEFT_HOME),
            *home_calls(RIGHT_HOME),
        ],
    )
    assert result.stdout.count('Homing (') == 2


def test_two_arm_home_stop_failure_prevents_all_home_requests(sandbox):
    root, env, log = control_sandbox(sandbox, stop_fails=True)

    result = run_script(root, env, 'home')

    assert result.returncode == 1
    assert 'refusing to home' in result.stderr
    assert 'Homing (' not in result.stdout
    assert_control_calls(
        log,
        [
            ['service', 'call', BOTH_ENGAGE, 'std_srvs/srv/SetBool', '{data: false}'],
        ],
    )


def test_single_arm_home_keeps_original_topic_without_stop_service(sandbox):
    root, env, log = control_sandbox(sandbox, both=False)

    result = run_script(root, env, 'home')

    assert result.returncode == 0, result.stderr
    assert_control_calls(log, home_calls(LEFT_HOME))
    assert result.stdout.count('Homing (') == 1


@pytest.mark.parametrize('both', [True, False])
@pytest.mark.parametrize('name,data', [('start', 'true'), ('stop', 'false')])
def test_start_and_stop_use_env_engage_service(sandbox, both, name, data):
    root, env, log = control_sandbox(sandbox, both=both)

    result = run_script(root, env, name)

    assert result.returncode == 0, result.stderr
    assert_control_calls(
        log,
        [
            ['service', 'call', BOTH_ENGAGE if both else LEFT_ENGAGE, 'std_srvs/srv/SetBool', f'{{data: {data}}}'],
        ],
    )
