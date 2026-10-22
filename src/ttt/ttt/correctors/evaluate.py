"""Evaluation and export, the same for every corrector: each episode held out in turn, and the final
fit on all episodes written as a nudge adapter (inference.nudge)."""

import json
from pathlib import Path

import numpy as np

from .data import LEAD_UP, WEIGHT, plan_error, problem, weighted_mean


def per_step(offset, n):
    """Offsets as plan_error takes them: (N, 7) as they are, a per-step target's (N, steps x 7) as
    (N, steps, 7)."""
    return offset if offset.shape[1] == 7 else offset.reshape(n, -1, 7)


def moved_cm(offset):
    """How far each row's offset moves the plan (cm): per step, averaged over the steps."""
    return np.linalg.norm(offset[..., :3], axis=-1).reshape(len(offset), -1).mean(axis=1) * 100


def leave_one_out(tensors, method, target='plan', **options):
    """Each episode held out in turn, `method` (a name in METHODS) fitted on the others. On its
    correction frames: the plan's error against the person's, and its gripper decisions, for the base
    alone, a constant nudge (the training mean), the corrector and the true nudge (the best one nudge
    per row can do). On its policy frames: how far each moves the base where it was left to drive.
    `target`: 'plan' or 'step' (data.problem). One dict per held-out episode."""
    from . import METHODS

    z, y, w, corrected, untouched = problem(tensors, target=target)
    pi = tensors['chunk_pi'].numpy().astype(np.float64)
    demo = tensors['chunk_demo'].numpy().astype(np.float64)
    episode = tensors['episode'].numpy()
    results = []
    for held in np.unique(episode):
        train, test = episode != held, episode == held
        corrector = METHODS[method](**options).fit(z[train], y[train], w[train], episode[train])
        offsets = {
            'base': np.zeros_like(y[test]),
            'constant': np.broadcast_to(weighted_mean(y[train], w[train]), y[test].shape),
            method: corrector.predict(z[test]),
            'true nudge': y[test],
        }
        fix, keep = test & corrected, test & untouched
        row = {'held_out': int(held), 'frames': int(test.sum()), 'corrected': int(fix.sum()), **corrector.summary()}
        for name, offset in offsets.items():
            position, gripper, decision = plan_error(pi[fix], demo[fix], per_step(offset[corrected[test]], fix.sum()))
            row[name] = {
                'position_cm': float(position.mean()),
                'gripper': float(gripper.mean()),
                'gripper_decision': float(decision.mean()),
            }
            if keep.any():
                off = per_step(offset[untouched[test]], keep.sum())
                row[name]['policy_frames_moved_cm'] = float(moved_cm(off).mean())
                row[name]['policy_frames_gripper_decision'] = float(plan_error(pi[keep], demo[keep], off)[2].mean())
        results.append(row)
    return results


def save_adapter(tensors, meta, path, model, results, method, gripper=False, target='plan', **options):
    """Fit `method` on every episode and write it as a nudge adapter for the checkpoint `model`, with
    `results` (leave_one_out's) beside it. With `gripper` the adapter also offsets the trigger;
    `target` 'step' makes it offset each step of the plan by its own."""
    from inference import nudge as nudge_adapter

    from . import METHODS

    z, y, w, corrected, _ = problem(tensors, target=target)
    episode = tensors['episode'].numpy()
    corrector = METHODS[method](**options).fit(z, y, w, episode)
    outputs = nudge_adapter.POSE + 1 if gripper else nudge_adapter.POSE
    steps = y.shape[1] // 7 if target == 'step' else 0
    columns = [s * 7 + j for s in range(max(steps, 1)) for j in range(outputs)]  # each step's first `outputs`
    description, arrays = corrector.export(columns)
    path = nudge_adapter.save(
        path,
        description,
        arrays,
        model,
        state=int(tensors['state'].shape[1]) if 'state' in tensors else 0,
        steps=steps,
        corrections={
            'repo_id': meta.get('repo_id'),
            'root': meta.get('root'),
            'task': meta.get('task'),
            'episodes': [int(e) for e in np.unique(episode)],
            'frames': int(len(episode)),
            'corrected_frames': int(corrected.sum()),
        },
        fit={
            'method': method,
            'offsets': f'one per {target}',
            **corrector.summary(),
            'target': "whole-plan nudge: the person's plan minus the base's, averaged over the plan"
            + (', its steps the policy drove 0' if 'step_intervention' in tensors else '; 0 where the policy drove'),
            'weights': {'correction': WEIGHT, **({} if 'step_intervention' in tensors else {'lead_up_s': LEAD_UP})},
            'n_action_steps': meta.get('n_action_steps'),
            'plan_seed': meta.get('plan_seed'),
            'plans': meta.get('plans'),
        },
    )
    (Path(path) / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    return path
