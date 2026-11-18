# Sim experiments

Plan, 2026-10-20. Every prediction is written before any result exists. Companion to
[training_experiments.md](training_experiments.md).

## The question

**Where should corrections live in the architecture, what should they be trained on, and how much
is enough: answered one factor at a time in simulation before committing robot time?**

## Data

The cup task (`BlueCupRedContainer` scene, `baxter_mujoco/lab.toml`), recorded with the robot's
recorder config: measured TCP labels, no lead, left arm, 8-D action, 15-D state, left hand and
third_eye cameras, 25 fps. Layouts in the real ranges capped at the sim arm's reach (x ≤ 0.87 m);
lighting random per episode.

| Set | Source | Use |
|---|---|---|
| Base | episodes 0–20 (21) | trains the base policy |
| Correction pool | episodes 21–37 (17) | offline corrections |
| Test, open loop | episodes 38–55 (18) | path and progress error |
| Adaptation layouts | 20 layouts, seeds 500–519 | on-policy corrections and attempts |
| Test, closed loop | 50 layouts, seeds 1000–1049 | success; never used to adapt or choose anything |

`andnetdeboer/BlueCupRedContainer_sim` (seed 0) holds episodes 0–59.

### References

| | Reference |
|---|---|
| **S0, the base** | VITA, 30k steps, episodes 0–20, training seed 0; 32-step chunks, 16 run |
| **M0, the method** | LoRA rank 16 on the flow net's 8 linear layers, L₁ with anchor λ = 0.5, frames off by more than 3 cm, corrections from 4 pool episodes (21, 25, 29, 33), 200 steps, batch 256, lr 1e-4, α = 1 |

## Shared method

### Base and adapter

As in the real plan: VITA flows an observation latent z₀ to an action latent; the adapter adds
αBA to frozen weights; ℓ_φ is the flow-matching error and the anchor replays base pairs. S0 is
trained three times (seeds 0, 1, 2) and every adaptation is run with three update seeds; both
spreads together are the noise floor.

### Two correctors

- **Offline:** a pool demonstration's chunk is the target wherever the base's plan is more than
  3 cm off it. Only states the demonstrations visited.
- **On-policy:** the base runs on the adaptation layouts; wherever its plan is more than 3 cm from
  the scripted planner's, replanned from the current TCP and cup pose, the planner's chunk is the
  target. The states the policy itself reaches.

### Closed-loop evaluation

`run_sim_teleop.sh --cup --cameras --left --policy` with the policy runtime driving it. Per
attempt: the layout and lighting of its seed, arm and gripper at home, right arm parked, 40 s.
Scored from sim object poses:

1. **Success**: cup at rest in the container within 40 s.
2. **Phase reached**: reach, grasp, transfer, release.
3. **Time to success** (s).
4. **Failure mode**: missed grasp, dropped in transit, released outside, stalled (5 s without
   progress), timeout.
5. **Forgetting**: success on layouts the base solved minus the base's.
6. **Cost**: corrected frames, attempts, update compute (s).

### Statistics

Unit: the test layout (50), paired across conditions. Success differences with 95% bootstrap
interval over layouts. An effect counts only if the interval excludes zero and exceeds the noise
floor. Choices made on adaptation layouts, never on the 50 test layouts.

### Code to write

1. Closed-loop evaluator: per seed, set layout and lighting, home, run policy, watch objects and
   gripper, stop at success or 40 s, one row per attempt.
2. Planner as corrector: `CupTask.plan` from current TCP and cup pose.
3. Correction extraction from evaluator bags: gated frames into (z₀, z_h) pairs.
4. Latent cache, LoRA wrapper, corrector, update loop, evaluation.
5. Adapter loading and α in the policy runtime.
6. Residual head and latent-offset network (Q1 variants).
7. On-policy correction collection loop (Q2).

---

## Q1: Where do corrections belong in the architecture?

**Inquiry.** The base fails: does the fix belong in how it plans actions (flow net), how it
perceives (encoder), or as a post-hoc adjustment to its output?

| # | What gets updated | How |
|---|---|---|
| M0 | flow net internals | LoRA rank 16, same corrections/steps/anchor |
| Q1a | observation encoder | LoRA rank 16, same corrections/steps/anchor |
| Q1b | decoded action chunk | residual MLP, matched parameter count |
| Q1c | starting latent z₀ | learned offset network, trained through frozen flow |

All four use the same 4 correction episodes and 200 update steps.

**Hypothesis.** M0 wins: the failure is in which action the latent flows to, not in perception
(Q1a) or fine adjustment of a correct chunk (Q1b). The offset (Q1c) helps but less: it can shift
the flow's starting point but not reshape the flow itself.

**Falsified if** Q1b or Q1c matches M0 beyond noise.

**What the result means:**
- M0 wins: the correction model is LoRA on the flow net. The architecture question is answered.
- Q1a wins: the base sees the world wrong, not acts wrong. Corrections should target perception.
- Q1b wins: a separate corrector network (your diagram) is the right design. Simpler, modular.
- Q1c wins: you don't need weight changes at all, just better initialization.

---

## Q2: What corrections should it train on?

**Inquiry.** Offline corrections come from states the demonstrations visited. On-policy corrections
come from states the policy actually reaches. The policy fails in states the demos never saw.

| # | Corrections from | States covered |
|---|---|---|
| M0 | pool demonstrations (offline) | demo-visited states |
| Q2a | planner on S0's own attempts (on-policy) | policy-visited states, same frame count as M0 |

**Hypothesis.** Q2a beats M0: corrections at the states where the policy actually fails are more
useful than corrections at states it may never reach.

**Falsified if** the interval includes zero.

### Which frames matter?

| # | Gate threshold |
|---|---|
| Q2b | every frame (no gate) |
| Q2c | off by more than 1 cm |
| M0 | off by more than 3 cm |
| Q2d | off by more than 6 cm |

**Hypothesis.** Gated frames beat all frames: training on frames where the policy is already
correct adds noise. 3 cm is near the optimum.

**Falsified if** Q2b matches or beats M0.

---

## Q3: How much capacity does the corrector need?

| # | LoRA rank |
|---|---|
| Q3a | 1 |
| Q3b | 4 |
| M0 | 16 |
| Q3c | 64 |

**Hypothesis.** Ranks 4–64 within noise of each other; rank 1 below. The correction is low-rank.

**Falsified if** gain keeps rising with rank.

---

## Q4: How much training and how many corrections?

### How many corrections?

| # | Correction source |
|---|---|
| Q4a | 1 attempt |
| Q4b | 2 attempts |
| M0 | 4 attempts |
| Q4c | 8 attempts |
| Q4d | 16 attempts |

**Hypothesis.** Most of the gain by 4 attempts. Diminishing returns after.

**Falsified if** the gain from 4 to 16 exceeds the gain from 1 to 4.

### How long to train?

| # | Steps | Learning rate |
|---|---|---|
| Q4e | 50 | 1e-4 |
| M0 | 200 | 1e-4 |
| Q4f | 800 | 1e-4 |
| Q4g | 200 | 1e-3 |

**Hypothesis.** Gain saturates by 200 steps; 800 or higher lr forgets more without more gain.

**Falsified if** Q4f beats M0 beyond noise without more forgetting.

### How hard to regularize?

| # | Anchor λ |
|---|---|
| Q4h | 0 (no anchor) |
| Q4i | 0.25 |
| M0 | 0.5 |
| Q4j | 0.75 |

**Hypothesis.** Forgetting falls as λ rises; gain falls only above 0.5.

**Falsified if** Q4h forgets no more than M0.

---

## Run order

**Gate 1 (day 1).** S0, M0, Q1a, Q1b, Q1c. Does the method work? Where do corrections belong?
If M0 doesn't beat S0: stop. If Q1b (residual) wins: redesign around a separate corrector.

**Gate 2 (day 2, only if gate 1 passes).** Q2a, Q2b, Q2c, Q2d. On-policy vs offline, which
frames. Uses the winning architecture from gate 1.

**Gate 3 (day 3).** Q3a–Q3c, Q4a–Q4d. Capacity and data quantity.

**Gate 4 (day 4).** Q4e–Q4j. Training length and regularization.

Each gate's results inform the next gate's defaults. A gate that falsifies its hypothesis
triggers a redesign before proceeding.

---

## Results

| # | Success /50 | Δ vs S0 [95% CI] | Forgetting | Time (s) | Corrected frames | Update (s) | Main failure |
|---|---|---|---|---|---|---|---|
| S0 | | | | | | | |
| M0 | | | | | | | |
| Q1a | | | | | | | |
| Q1b | | | | | | | |
| Q1c | | | | | | | |
| Q2a | | | | | | | |
| Q2b | | | | | | | |
| Q2c | | | | | | | |
| Q2d | | | | | | | |
| Q3a | | | | | | | |
| Q3b | | | | | | | |
| Q3c | | | | | | | |
| Q4a | | | | | | | |
| Q4b | | | | | | | |
| Q4c | | | | | | | |
| Q4d | | | | | | | |
| Q4e | | | | | | | |
| Q4f | | | | | | | |
| Q4g | | | | | | | |
| Q4h | | | | | | | |
| Q4i | | | | | | | |
| Q4j | | | | | | | |

## Threats to validity

- **The sim is not the robot.** Shorter fingers, higher wall grasp, rendered images, scripted
  demonstrator. Sim results rank methods; the robot sizes them.
- **A scripted expert is a perfect corrector.** Real human corrections are late, partial, noisy.
- **One task.** The cup task tests grasp precision and placement, not sequential phase skipping
  (the dustpan shortcutting failure). Generality across task structures needs a second task.
- **Reach.** Layouts capped at 0.87 m.
