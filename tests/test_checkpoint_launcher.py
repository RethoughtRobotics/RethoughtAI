"""Checkpoint selection tests stop before ROS, tmux, or robot interaction."""

import shutil
import subprocess
from pathlib import Path


def launcher(tmp_path, *names):
    script = tmp_path / 'scripts' / 'run_inference'
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).parents[1] / 'scripts' / 'run_inference', script)
    for name in names:
        checkpoint = tmp_path / 'models' / name
        checkpoint.mkdir(parents=True)
        (checkpoint / 'config.json').write_text('{}')
    return lambda *args: subprocess.run(['bash', str(script), *args], capture_output=True, text=True)


def test_numeric_steps_ignore_padding(tmp_path):
    run = launcher(tmp_path, 'vita_demo/step-3000', 'vita_demo/step-030000')
    assert run('--vita', '003000', '--resolve').stdout.strip() == 'models/vita_demo/step-3000'
    assert run('--vita', 'step-030000', '--resolve').stdout.strip().endswith('step-030000')


def test_latest_uses_numeric_order(tmp_path):
    run = launcher(tmp_path, 'vita_demo/step-9000', 'vita_demo/step-100000')
    assert run('--vita', '--resolve').stdout.strip().endswith('step-100000')


def test_missing_step_lists_real_choices(tmp_path):
    run = launcher(tmp_path, 'vita_demo/step-030000')
    result = run('--vita', '3000', '--resolve')
    assert result.returncode == 2
    assert '3000 is not 30000' in result.stderr
    assert 'step-030000' in result.stderr


def test_ambiguity_is_not_silently_resolved(tmp_path):
    run = launcher(tmp_path, 'vita_a/step-3000', 'vita_b/step-003000')
    assert run('--vita', '3000', '--resolve').returncode == 2
    assert run('--vita_a', '3000', '--resolve').returncode == 0


def test_duplicate_numeric_steps_rejected(tmp_path):
    run = launcher(tmp_path, 'vita_a/step-3000', 'vita_a/step-003000')
    assert run('--vita_a', '3000', '--resolve').returncode == 2


def test_list_never_starts_ros(tmp_path):
    run = launcher(tmp_path, 'vita_demo/step-3000')
    assert run('--list').stdout.strip() == 'models/vita_demo/step-3000'
