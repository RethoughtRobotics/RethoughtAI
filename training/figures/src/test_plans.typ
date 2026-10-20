// One diagram per test plan in training_experiments.md. Every offline plan shares one pipeline: a
// base VITA trained once on episodes 0 to 20 and frozen, a simulated corrector on the correction
// pool (21 to 37), a LoRA update, and metrics on the test episodes (38 to 55). What a plan varies is
// outlined in orange and listed under the diagram. The last page is the robot plan.
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#set page(width: auto, height: auto, margin: 10pt, fill: white)
#set text(font: "Noto Sans", size: 8.5pt)

#let ink = rgb("#333333")
#let robot = rgb("#e8e8e8")
#let human = rgb("#f6dcc4")
#let data = rgb("#fbf6e3")
#let model = rgb("#d9e4f2")
#let orange = rgb("#eb6834")
#let dim(body) = text(size: 7pt, fill: rgb("#555555"), body)

#let pipeline(title, varied, conditions, task: "sweep", corrector: [plan off by > 3 cm:\ the demo's chunk is the target], update: [Update\ #dim[200 steps, batch 256]], lora: [LoRA rank 16\ #dim[flow net, 0.33 M]], extra: none) = {
  let s(key) = if key in varied { 1.6pt + orange } else { 0.6pt + ink }
  let box(pos, body, key, fill: white) = node(pos, body, name: label(key), fill: fill, stroke: s(key), corner-radius: 0pt, inset: 6pt)
  let ds = if task == "sweep" { [SweepTrash, minimax] } else { [BlueCupRedContainer, minimax] }
  align(left, text(size: 10pt, weight: "bold", title))
  v(4pt)
  diagram(
    spacing: (9mm, 8mm),
    edge-stroke: 0.6pt + ink,
    box((0, 0), [Base demos\ #dim[ep 0 to 20, #ds]], "demos", fill: data),
    box((0, 1), [Base VITA\ #dim[trained once, frozen]], "base", fill: model),
    box((1, 0), [Correction pool\ #dim[ep 21 to 37]], "pool", fill: data),
    box((1, 1), [Simulated corrector\ #dim[#corrector]], "corr", fill: human),
    box((2, 1), [Pairs $(z_0, z_"h")$\ #dim[$z_"h"$: frozen action encoder]], "pairs", fill: data),
    box((3, 0), [Anchor\ #dim[replay of ep 0 to 20]], "anchor", fill: data),
    box((3, 1), update, "update", fill: human),
    box((4, 1), lora, "lora", fill: white),
    box((5, 1), [Policy\ #dim[base + $alpha dot$ LoRA]], "policy", fill: model),
    box((5, 0), [Test\ #dim[ep 38 to 55, never trained on]], "test", fill: data),
    box((6, 1), [Metrics\ #dim[skip rate, path error,\ forgetting on ep 0 to 20]], "eval", fill: robot),
    edge(<demos>, <base>, "-|>"),
    edge(<pool>, <corr>, "-|>"),
    edge(<base>, <corr>, "-|>", [plans]),
    edge(<corr>, <pairs>, "-|>"),
    edge(<pairs>, <update>, "-|>"),
    edge(<anchor>, <update>, "-|>"),
    edge(<demos>, (0, -0.55), (3, -0.55), <anchor>, "-|>"),
    edge(<update>, <lora>, "-|>"),
    edge(<lora>, <policy>, "-|>"),
    edge(<base>, (0, 1.75), (5, 1.75), <policy>, "-|>", [frozen weights], label-side: right),
    edge(<test>, <policy>, "-|>"),
    edge(<policy>, <eval>, "-|>"),
    ..if extra != none { extra } else { () },
  )
  v(4pt)
  block(width: 200mm, inset: 6pt, stroke: 0.6pt + orange, align(left, text(size: 8pt, conditions)))
}

#pipeline(
  [Test plan 1: do corrections fix the shortcut?],
  ("update", "lora"),
  [*Varied:* whether the adapter is updated. *T0* no update (the base alone). *T1* update on corrections
    from 4 pool episodes. *T1b* T1 with two other draws of 4 episodes (noise floor).],
)
#pagebreak()
#pipeline(
  [Test plan 2: does replaying the base demos prevent forgetting?],
  ("anchor",),
  [*Varied:* the anchor's share of each batch. *T1* 50%. *T2* 0% (corrections only).
    *Read on:* forgetting (episodes 0 to 20) and the test gain together.],
)
#pagebreak()
#pipeline(
  [Test plan 3: targeted corrections or simply more demos?],
  ("corr",),
  [*Varied:* which pool frames train the adapter. *T1* only the frames where the base was off by more
    than 3 cm. *T5* every frame of the same 4 episodes.],
  corrector: [T1: only frames off by > 3 cm\ T5: every frame],
)
#pagebreak()
#pipeline(
  [Test plan 4: how many corrections are enough?],
  ("pool",),
  [*Varied:* the number of pool episodes corrections come from. *T3* 1. *T1* 4. *T4* 16.
    Same rule, same steps.],
)
#pagebreak()
#pipeline(
  [Test plan 5: how much capacity does a fix need?],
  ("lora",),
  [*Varied:* what is trainable. *T6* LoRA rank 4. *T1* rank 16. *T7* rank 64. *T8* the whole flow net
    (12.6 M, no adapter).],
  lora: [rank 4 / 16 / 64,\ or the whole flow net],
)
#pagebreak()
#pipeline(
  [Test plan 6: does a negative (the base's wrong answer) help?],
  ("update", "pairs"),
  [*Varied:* the loss. *T1* flow loss toward $z_"h"$ only. *T9* the preference loss: lower flow loss on
    $z_"h"$ than on the base's own latent $z_pi$, both relative to the frozen base.],
  update: [Update\ #dim[T1: flow toward $z_"h"$\ T9: prefer $z_"h"$ over $z_pi$]],
)
#pagebreak()
#pipeline(
  [Test plan 7: does the blend dial interpolate?],
  ("policy",),
  [*Varied:* $alpha$ at test time only, on T1's adapter: 0, 0.25, 0.5, 0.75, 1. No new training.
    $alpha = 0$ must reproduce T0 exactly.],
)
#pagebreak()
#pipeline(
  [Test plan 8: does it carry over to a second task?],
  ("demos", "pool", "test"),
  [*Varied:* the task. T0 and T1 repeated on BlueCupRedContainer (one arm, 2 cameras, 56 episodes,
    same split). The skip metric is replaced by progress error, since the cup task has no sweep.],
  task: "cup",
)
#pagebreak()

// The robot plan: designs A and B on the same adapter.
#{
  let s(key) = if key in ("hum", "att", "rew") { 1.6pt + orange } else { 0.6pt + ink }
  let box(pos, body, key, fill: white) = node(pos, body, name: label(key), fill: fill, stroke: s(key), corner-radius: 0pt, inset: 6pt)
  align(left, text(size: 10pt, weight: "bold", [Test plan 9: corrections against reward, on the robot]))
  v(4pt)
  diagram(
    spacing: (10mm, 8mm),
    edge-stroke: 0.6pt + ink,
    box((0, 1), [Base VITA, frozen\ #dim[\+ $alpha dot$ LoRA]], "pol", fill: model),
    box((1, 1), [Runtime + feedforward\ #dim[bounded IK, raw_id]], "rt", fill: robot),
    box((2, 1), [Baxter\ #dim[3 fixed layouts]], "arm", fill: robot),
    box((3, 0), [A: human takes over\ #dim[cuff or VR; hold frames]], "hum", fill: human),
    box((3, 2), [B: $K = 8$ attempts\ #dim[$z_0 + sigma epsilon_i$, reward $r_i$]], "att", fill: human),
    box((4, 0), [Loss A\ #dim[flow toward $z_"h"$ + anchor]], "la", fill: human),
    box((4, 2), [Loss B\ #dim[$sum_i w_i$ flow toward $hat(z)_(1, i)$ + anchor]], "rew", fill: human),
    box((5, 1), [LoRA update\ #dim[between attempts]], "up", fill: white),
    box((6, 1), [Score\ #dim[robot and operator minutes\ to 8 of 10 successes]], "sc", fill: robot),
    edge(<pol>, <rt>, "-|>"),
    edge(<rt>, <arm>, "-|>"),
    edge(<arm>, <hum>, "-|>"),
    edge(<arm>, <att>, "-|>"),
    edge(<hum>, <la>, "-|>"),
    edge(<att>, <rew>, "-|>"),
    edge(<la>, <up>, "-|>"),
    edge(<rew>, <up>, "-|>"),
    edge(<up>, (5, 2.7), (0, 2.7), <pol>, "-|>", [next attempt], label-side: left),
    edge(<arm>, (2, -0.6), (6, -0.6), <sc>, "-|>", [success or failure per attempt]),
  )
  v(4pt)
  block(width: 200mm, inset: 6pt, stroke: 0.6pt + orange, align(left, text(size: 8pt)[
    *Varied:* where the adapter's target comes from. *R0* base, no update. *R1* design A (corrections).
    *R2* design B (reward). Same base, adapter, anchor, layouts and attempt budget. Needs the
    policy velocity feedforward first.
  ]))
}
