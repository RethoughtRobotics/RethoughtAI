# Sim experiments

Plan, 2026-10-20. Every prediction below is written before any result exists. Companion to
[training_experiments.md](training_experiments.md), whose methods, losses and figures this plan
reuses; read that first.

## The question

The real experiments can only score policies open loop, on recorded frames, with a demonstration
playing the corrector, on 18 test episodes from the same sessions. **Which features of the data, the
observations, the model and the correction actually change closed-loop success, each one tested on
its own?** The MuJoCo sim (BaxterSDK `baxter_mujoco`) runs the policy through the same runtime,
teleop, IK and gripper as the robot, scores success automatically, and generates as many scripted
demonstrations as needed under controlled variation, so every factor can be changed alone, with
paired layouts and several training seeds.

What the sim does not answer: its gripper fingers are shorter, it grasps the cup's wall higher and
its images are rendered, so a sim result shows whether a mechanism works, not how large it is on
the robot. Anything that matters is carried back to the real data (the "Real counterpart" column).

## Data

All sim data is the cup task (`BlueCupRedContainer` scene, the lab of `baxter_mujoco/lab.toml`),
recorded with the robot's recorder config and exported like the real cup data: measured TCP labels,
no lead, left arm, 8-D action, 15-D state, left hand and third_eye cameras, 25 fps.

| Set | Source | Use |
|---|---|---|
| Base | generated episodes 0 to 20 (21, as the real base) | trains every base policy |
| Correction pool | generated episodes 21 to 37 (17) | the only source of offline corrections |
| Test, open loop | generated episodes 38 to 55 (18) | path and progress error, as the real plan |
| Test, closed loop | 50 layouts, seeds 1000 to 1049, never generated as demonstrations | success; every condition runs the same 50, in the same order |

`andnetdeboer/BlueCupRedContainer_sim` (60 episodes, seed 0) already holds episodes 0 to 59 of the
reference generator. An experiment that changes the data (group A) generates its own episodes with
the same seeds, so its layouts match the reference episode for episode.

### The reference condition S0

Every experiment changes exactly one row of this table and keeps the rest.

| Factor | S0 |
|---|---|
| Demonstrations | 21 (episodes 0 to 20) |
| Layouts | cup x 0.46 to 0.87 m, y -0.20 to 0.44 m; container x 0.42 to 0.87 m, y -0.28 to 0.36 m (the real ranges, capped at the sim arm's reach) |
| Lighting | randomised per episode (`scenes.sample_lighting`) |
| Grasp | the cup's wall from straight above, fixed orientation |
| Demonstration noise | none: the scripted plan exactly |
| Labels | measured TCP, no lead |
| Cameras | left hand and third_eye |
| State | TCP pose, gripper, 7 joints |
| History | the current frame only |
| Policy | VITA, the repository configuration, 30k steps, training seed 0 |
| Chunk | 32 steps decoded, 16 run |
| Image augmentation | the repository default |
| Distractors | none |

## Shared method

### The base policy

![VITA](figures/vita.png)

VITA [1], as in the real plan: observation latent to action latent by flow matching, a 32-step
chunk of which 16 run. One S0 base is trained three times (training seeds 0, 1, 2: S0, S0b, S0c);
their spread is the noise floor of every comparison below.

### Closed-loop evaluation

`run_sim_teleop.sh --cup --cameras --left --policy` with the policy runtime (`scripts/run_inference`)
driving it, exactly as on the robot. Per attempt: the layout and lighting of its seed, the arm and
gripper at home, the right arm parked, 40 s. Scored from the sim's object poses:

1. **Success**: the cup at rest inside the container (`CupTask.success`), within 40 s.
2. **Phase reached**: reach, grasp (gripper closed on the cup), transfer (cup lifted), release (cup
   in the container). The furthest phase is the partial credit of a failure.
3. **Time to success** (s), successes only.
4. **Failure mode**, from the logged trajectory: missed grasp, dropped in transit, released outside,
   stalled (no progress for 5 s), timeout.

Open-loop metrics on the 18 open-loop test episodes are the real plan's (progress error, path error,
phase errors), so every sim result has an open-loop number comparable to the real ones.

### Statistics

The unit is the **layout** (50), paired across conditions: the same seeds, layouts and lighting.
Success is compared as the paired difference in success rate with a 95% bootstrap interval over
layouts [7, 8]; continuous metrics as the median paired difference. **An effect counts** only if its
interval excludes zero **and** it exceeds the spread of S0, S0b and S0c. Nothing is tuned on the 50
test layouts.

### Code to write (shared by the plans below)

1. **Generator switches** in `baxter_mujoco.datagen` (one flag each, defaults = S0): `--n`,
   layout region, `--lab-lighting` (exists), grasp yaw range, demonstration noise, distractor.
2. **Reach cap**: the S0 layout ranges in `scenes.py` (x at most 0.87 m; every failed attempt of the
   first 60-episode run had the cup or container beyond it).
3. **Closed-loop evaluator**: for each seed, set the layout and lighting (`/sim/layout`,
   `/sim/lighting`), home, start the policy, watch `/sim/objects` and the gripper, stop at success or
   40 s, write one row per attempt (seed, success, phase, time, failure mode) and keep the bag.
4. **Planner as corrector**: `CupTask.plan` from the arm's current TCP and the cup's current pose
   (`/sim/objects`), so the scripted expert can label any state the policy reaches.
5. **Dataset subsets** by episode list (`split_dataset`), as the real `_trainK` / `_heldoutM` copies.
6. **Results**: one table row per condition, intervals against S0, the seed spread.

## Group A: what in the data matters

### Plan S1: how many demonstrations?

**Inquiry.** Few-shot imitation degrades as demonstrations shrink; the real few-shot curve (the old
runs 6 and 7) costs a robot session per point and stops at 56. Data size is the first factor of
offline imitation studies [25].

**Hypothesis H-S1.** Success rises steeply from 5 to 21 demonstrations and flattens by 55. **Null:**
no change beyond the seed spread from 21 to 200.

| # | Demonstrations |
|---|---|
| S1a | 5 |
| S1b | 10 |
| S0 | 21 |
| S1c | 55 |
| S1d | 200 |

**Expected outcome.** S1a and S1b clearly below S0; S1c above S0; S1d within the seed spread of S1c.
**Falsified if** S1d beats S1c by more than S1c beats S0.
**Real counterpart.** The real 12 / 24 / 48 curve (`sweep_*_train12/24/48`).
**Code to write.** Item 5.

### Plan S2: does lighting randomisation buy robustness?

**Inquiry.** Randomising rendering during training makes vision policies robust to appearance they
never saw [26].

**Hypothesis H-S2.** A base trained under the lab's fixed lighting succeeds as often as S0 under the
lab's lighting but less under randomised test lighting. **Null:** no difference under either.

| # | Training lighting | Test lighting |
|---|---|---|
| S0 | randomised | randomised |
| S0-lab | randomised | the lab's |
| S2 | the lab's | randomised |
| S2-lab | the lab's | the lab's |

**Expected outcome.** S2-lab about S0-lab; S2 clearly below S0. **Falsified if** S2's interval
against S0 includes zero.
**Real counterpart.** None directly; it says whether real data needs lighting variety.
**Code to write.** Item 1 (`--lab-lighting` exists for the generator; the evaluator needs it too).

### Plan S3: does it extrapolate to unseen layouts?

**Inquiry.** Imitation policies interpolate between demonstrated states and fail outside them;
whether a flow policy extrapolates to an unseen region of the table is a question of coverage, not
of data size.

**Hypothesis H-S3.** Trained on cups at x below 0.70 m only, success on test cups beyond 0.70 m
drops well below S0's, while success below 0.70 m stays. **Null:** no drop beyond 0.70 m.

| # | Training cups | Scored on |
|---|---|---|
| S0 | x 0.46 to 0.87 | test cups below / beyond 0.70 m |
| S3 | x 0.46 to 0.70 (21 demonstrations) | the same |

**Expected outcome.** S3 beyond 0.70 m at most half of S0 beyond 0.70 m; below 0.70 m within noise.
**Falsified if** the beyond-0.70 m difference interval includes zero.
**Real counterpart.** Where the real cups were placed; whether real demonstrations need to span the
table.
**Code to write.** Layout region switch (item 1).

### Plan S4: do recovery demonstrations help?

**Inquiry.** Behaviour cloning compounds its errors because demonstrations never leave the expert's
path [5]; injecting noise during demonstration and recording the expert's recovery fixes the
coverage cheaply (DART [27]).

**Hypothesis H-S4.** Demonstrations whose arm is pushed off the plan and recovered raise success over
clean ones at the same 21 episodes. **Null:** no change.

| # | Demonstrations |
|---|---|
| S0 | clean |
| S4 | the commanded TCP perturbed (2 cm, 0.5 s, twice per episode), the planner recovering; labels as recorded |

**Expected outcome.** S4 above S0, mostly fewer stalled and missed-grasp failures.
**Falsified if** the S4 against S0 interval includes zero.
**Real counterpart.** Whether to perturb the real arm during demonstrations.
**Code to write.** Demonstration noise switch (item 1) using the planner of item 4.

### Plan S5: does orientation variety matter?

**Inquiry.** The real demonstrations turn the wrist (quaternion x from -0.57 to 0.04); the sim's
grasp never does. If the policy uses orientation as a cue for phase, variety changes what it learns.

**Hypothesis H-S5.** Randomising the grasp yaw by up to 30 degrees changes success by less than the
seed spread under the S0 test, and raises it when test layouts need a turned wrist. **Null:** no
change in either.

| # | Grasp yaw in demonstrations | Test |
|---|---|---|
| S0 | fixed | S0 layouts |
| S5 | uniform +-30 degrees | S0 layouts |
| S5-turn | uniform +-30 degrees | container rotated so the release needs a turned wrist |

**Expected outcome.** S5 within noise of S0; S5-turn above S0 on the turned test.
**Falsified if** S5 is clearly below S0 (variety costs data) or S5-turn shows no gain.
**Code to write.** Grasp yaw switch (item 1); a turned-release variant of `CupTask.plan`.

### Plan S6: measured pose or commanded goal as the label?

**Inquiry.** The adapter can label each frame with the measured TCP (`--labels measured`, the
kinesthetic choice) or with the goal the servo chased (`--labels goal`); the servo lags the goal by
about 1/K, so the two differ by a lag the policy must undo or reproduce. TODO section 5: gripper
labels are the measured opening, not the intent.

**Hypothesis H-S6.** Goal labels raise success through faster, less lagged motion, and command-based
gripper labels close earlier and drop less. **Null:** no difference.

| # | Pose labels | Gripper labels |
|---|---|---|
| S0 | measured | measured opening |
| S6 | goal | measured opening |
| S6g | measured | commanded (trigger) |

**Expected outcome.** S6 shorter time to success; S6g fewer missed grasps.
**Falsified if** neither interval excludes zero.
**Real counterpart.** Real teleop sessions have both; the kinesthetic ones only measured.
**Code to write.** Gripper command labels in `adapter/labels.py`; no generator change.

## Group B: what in the observation matters

### Plan S7: which camera carries the task?

**Inquiry.** Policies with a wrist and a scene camera often lean on one of them; the real combo /
drop augmentations (third_eye dropout 30%) assume the scene camera can mislead.

**Hypothesis H-S7.** The hand camera alone succeeds at grasping but fails more at finding the
container; the third_eye alone grasps worse. Both together are best. **Null:** one camera matches
both.

| # | Cameras |
|---|---|
| S0 | hand and third_eye |
| S7h | hand only |
| S7t | third_eye only |

**Expected outcome.** S7h more released-outside failures, S7t more missed grasps, both below S0.
**Falsified if** either single camera is within the seed spread of S0.
**Code to write.** None (the training config's camera list).

### Plan S8: does proprioception invite copying?

**Inquiry.** Given its own past state, a cloned policy can learn to continue its motion instead of
reading the scene (causal confusion [28], copycat agents [29]); TODO section 5 lists the copycat
shortcut.

**Hypothesis H-S8.** Removing the joint angles from the state (keeping TCP and gripper) changes
success by less than the seed spread; removing the whole state lowers it. **Null:** both within
noise.

| # | State |
|---|---|
| S0 | TCP, gripper, 7 joints (15-D) |
| S8a | TCP and gripper (8-D) |
| S8b | none (images only) |

**Expected outcome.** S8a within noise of S0; S8b below S0, mostly in the transfer phase.
**Falsified if** S8a differs from S0 beyond noise in either direction.
**Code to write.** State subsets in the training config.

### Plan S9: does a frame of history help or hurt?

**Inquiry.** The old real runs 3 and 3b asked whether a second frame, 40 ms or 1.52 s back, stops the
sweep skip; history can also feed copycat behaviour [29].

**Hypothesis H-S9.** A frame 1.52 s back raises success on attempts that pass through visually
ambiguous states (cup lifted, arm over the table); a frame 40 ms back does not. **Null:** neither
changes success.

| # | Observation frames |
|---|---|
| S0 | t |
| S9a | t - 1, t (40 ms) |
| S9b | t - 38, t (1.52 s) |

**Expected outcome.** S9a within noise; S9b above S0. **Falsified if** S9b's interval includes zero.
**Real counterpart.** The old runs 3 and 3b, now with closed-loop success.
**Code to write.** None (`n_obs_steps`, `obs_stride`).

### Plan S10: does a correlated distractor create a shortcut?

**Inquiry.** Networks learn whatever cue predicts the label most simply, including ones that only
happen to correlate in the training data [30]. The real data has one: the operator in frame,
correlated with the task's phase, which inpainting removes. The sim can create such a cue on
purpose, then break the correlation at test.

**Hypothesis H-S10.** A marker whose position follows the container in training (where the cup goes)
makes the policy release at the marker when the two disagree at test; masking the marker in
training (as inpainting removes the operator) restores S0's success. **Null:** no change.

| # | Training | Test |
|---|---|---|
| S0 | no marker | no marker |
| S10 | marker beside the container | marker moved 20 cm from the container |
| S10m | marker present but masked from the images | as S10 |

**Expected outcome.** S10 clearly below S0 with released-outside failures toward the marker; S10m
within noise of S0. **Falsified if** S10 is within noise of S0.
**Real counterpart.** Whether the operator in frame is a shortcut, and whether minimax inpainting is
what removes it.
**Code to write.** Distractor switch (item 1); marker masking in the image pipeline.

### Plan S11: does image augmentation earn its cost?

**Inquiry.** The real runs train with the default augmentations and third_eye dropout (combo / drop);
none was tested alone.

| # | Augmentation |
|---|---|
| S0 | repository default |
| S11a | none |
| S11b | default plus third_eye dropout 30% |

**Hypothesis H-S11.** S11a below S0 under randomised lighting; S11b within noise of S0. **Null:** all
within noise.
**Expected outcome / falsified if** as the hypothesis and null.
**Code to write.** None.

## Group C: what in the model matters

### Plan S12: VITA or DiT at few demonstrations?

![Multi-Task DiT](figures/dit.png)

**Inquiry.** The old real runs 8 to 9 asked whether the 278 M Multi-Task DiT falls off faster than
VITA with fewer demonstrations; the sim gives the whole curve.

**Hypothesis H-S12.** At 55 demonstrations the DiT matches VITA; at 10 it falls further. **Null:**
the gap is the same at both sizes.

| # | Policy | Demonstrations |
|---|---|---|
| S0 / S1b / S1c | VITA | 21 / 10 / 55 |
| S12a / S12b / S12c | Multi-Task DiT | 21 / 10 / 55 |

**Expected outcome.** S12c within noise of S1c; S12b clearly below S1b.
**Falsified if** the DiT minus VITA gap at 10 is no larger than at 55.
**Code to write.** None.

### Plan S13: how much of each chunk to run?

**Inquiry.** Running a long chunk open loop ignores new observations; running a short one re-plans
often and stitches plans (TODO section 5: plan seams, 55 mm steps between chunks).

| # | Chunk run of 32 |
|---|---|
| S13a | 8 |
| S0 | 16 |
| S13b | 32 |

**Hypothesis H-S13.** S13a raises success through re-planning; S13b lowers it. **Null:** within
noise. **Expected outcome / falsified if** as the hypothesis and null.
**Code to write.** None (runtime setting).

## Group D: test-time training, closed loop

![TTT designs](figures/ttt.png)

![TTT at tensor level](figures/ttt_lora.png)

The real plan's test plans 1 to 9 on the sim cup base, with the same LoRA [4], losses and settings
($\mathcal{L}_1$, $\lambda = 0.5$, rank 16, 200 steps), scored closed loop. The base is S0.

### Plan S14: offline corrections, closed-loop success

![Test plan 1](figures/test-plan-1.png)

**Inquiry.** The real test plan 1 scores corrections open loop only.

**Hypothesis H-S14.** T1's update (demonstration as corrector, pool episodes 21, 25, 29, 33) raises
closed-loop success over S0. **Null:** no change.

| # | Condition |
|---|---|
| S0 | base |
| S14 | the real plan's T1 on sim pool episodes |

**Expected outcome.** S14 above S0, and its open-loop gain of the same sign as the real T1-cup.
**Falsified if** the interval includes zero. **Code to write.** The real plan's items 4 to 8.

### Plan S15: on-policy corrections (DAgger) against offline ones

![One correction](figures/correction-2.png)

**Inquiry.** The real plan's corrector labels only the states the demonstrations visited; DAgger [5]
and HG-DAgger [6] label the states the learner itself reaches. The planner can do that in the sim,
which removes the real plan's first threat to validity.

**Hypothesis H-S15.** Corrections labelled by the planner on the policy's own failing attempts (the
3 cm rule, from 4 rounds of attempts) raise success more than S14's offline corrections at the same
number of corrected frames. **Null:** no difference.

| # | Corrections |
|---|---|
| S14 | offline, from demonstrations |
| S15 | on-policy, planner-labelled, same frame count |

**Expected outcome.** S15 above S14, mostly fewer stalled attempts.
**Falsified if** the S15 against S14 interval includes zero.
**Code to write.** Item 4; correction extraction from evaluator bags.

### Plan S16: how good must a human correction be?

**Inquiry.** A person corrects late, partly and from another pose; the real plan cannot measure how
much that costs. The sim's planner can be degraded on purpose.

**Hypothesis H-S16.** Corrections 0.5 s late keep most of S15's gain; corrections covering half the
chunk, or with 2 cm noise, lose more. **Null:** no degradation loses more than the seed spread.

| # | Planner corrections |
|---|---|
| S15 | exact |
| S16a | 0.5 s late |
| S16b | first half of the chunk only |
| S16c | 2 cm Gaussian noise on each waypoint |

**Expected outcome.** S16a within noise of S15; S16b and S16c below it, above S0.
**Falsified if** every degradation is within noise of S15 (corrections are robust: good news, but the
hypothesis fails).
**Code to write.** Degradation options on item 4.

### Plan S17: corrections against reward, piloted in the sim

![Test plan 9](figures/test-plan-9.png)

**Inquiry.** The real test plan 9 compares a human's corrections with reward on attempts, per minute
of robot time. In the sim, success is the reward, attempts cost no operator and K, beta and sigma
can be chosen before the robot.

**Hypothesis H-S17.** Design A (S15's planner corrections) reaches 40 of 50 test successes in fewer
attempts than design B (the reward-weighted update, $K = 8$, $\beta$ from $\mathrm{KL}(w \| U) =
\ln 2$ [13]). **Null:** B as fast or faster.

| # | Condition |
|---|---|
| S17a | A: 3 rounds of attempts with planner corrections |
| S17b | B: 3 rounds of $K = 8$ seeded latent offsets, success reward |

**Expected outcome.** A reaches 40 of 50 within 3 rounds; B needs more attempts. The settings of B
that work here are the robot run's.
**Falsified if** B reaches 40 of 50 in fewer attempts.
**Code to write.** The real plan's design B items, against the sim.

## Group E: does sim data help on the robot?

### Plan S18: co-training with sim

**Inquiry.** Mixing simulated demonstrations into real training data improves real policies when the
sim covers what the real data lacks [31].

**Hypothesis H-S18.** Real episodes 0 to 20 plus 200 sim episodes lower open-loop path error on the
real test episodes 38 to 55 against the real episodes alone. **Null:** no change, or worse.

| # | Training data |
|---|---|
| R-base | real cup episodes 0 to 20 |
| S18 | the same plus 200 sim episodes |

**Expected outcome.** S18 lower path error in the reach and transfer phases; grasp phase unchanged
(the sim grasps differently). **Falsified if** the interval includes zero.
**Code to write.** Dataset merge (`merge_datasets`), with the real and sim normalisation statistics
checked for disagreement first.

## Results

Closed loop, 50 test layouts; success difference against S0 with its 95% interval in brackets.

| # | Success of 50 | Difference | Time to success (s) | Main failure mode |
|---|---|---|---|---|
| S0 | | | | |
| S0b | | | | |
| S0c | | | | |
| S1a | | | | |
| S1b | | | | |
| S1c | | | | |
| S1d | | | | |
| S2 | | | | |
| S2-lab | | | | |
| S3 | | | | |
| S4 | | | | |
| S5 | | | | |
| S5-turn | | | | |
| S6 | | | | |
| S6g | | | | |
| S7h | | | | |
| S7t | | | | |
| S8a | | | | |
| S8b | | | | |
| S9a | | | | |
| S9b | | | | |
| S10 | | | | |
| S10m | | | | |
| S11a | | | | |
| S11b | | | | |
| S12a | | | | |
| S12b | | | | |
| S12c | | | | |
| S13a | | | | |
| S13b | | | | |
| S14 | | | | |
| S15 | | | | |
| S16a | | | | |
| S16b | | | | |
| S16c | | | | |
| S17a | | | | |
| S17b | | | | |

Real, open loop (S18), the real plan's format.

| # | Progress error | Path error (mm) | Transfer-phase error (mm) |
|---|---|---|---|
| R-base | | | |
| S18 | | | |

## Threats to validity

- **The sim is not the robot.** Shorter fingers, a higher wall grasp, rendered images, a scripted
  demonstrator. Sim results rank mechanisms; the robot sizes them (test plan 9, S18).
- **A scripted demonstrator is consistent.** Human demonstrations vary in timing and path; a policy
  trained on scripted data faces an easier imitation problem, which may hide effects (S1, S4) that
  matter with people. S4 and S16 add controlled imperfection.
- **One task.** Every sim plan is the cup task; the sweep's shortcut has no sim counterpart yet.
- **Reach.** Layouts are capped at the sim arm's reach (0.87 m): the far edge of the real table is
  untested.
- **Evaluation layouts are one draw.** The 50 seeds are fixed for pairing; a second set of 50 checks
  any result that is close to the noise floor.

## References

Numbers 1 to 24 are those of [training_experiments.md](training_experiments.md).

[25] A. Mandlekar et al., "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation," CoRL 2021. [arXiv:2108.03298](https://arxiv.org/abs/2108.03298)

[26] J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," IROS 2017. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)

[27] M. Laskey et al., "DART: Noise Injection for Robust Imitation Learning," CoRL 2017. [arXiv:1703.09327](https://arxiv.org/abs/1703.09327)

[28] P. de Haan, D. Jayaraman, S. Levine, "Causal Confusion in Imitation Learning," NeurIPS 2019. [arXiv:1905.11979](https://arxiv.org/abs/1905.11979)

[29] C. Wen et al., "Fighting Copycat Agents in Behavioral Cloning from Observation Histories," NeurIPS 2020. [arXiv:2010.14876](https://arxiv.org/abs/2010.14876)

[30] R. Geirhos et al., "Shortcut Learning in Deep Neural Networks," Nature Machine Intelligence 2020. [arXiv:2004.07780](https://arxiv.org/abs/2004.07780)

[31] A. Maddukuri et al., "Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation," RSS 2025. [arXiv:2503.24361](https://arxiv.org/abs/2503.24361)
