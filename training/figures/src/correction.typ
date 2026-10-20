// Learning from corrections: system, correction anatomy, model.
// Live preview: VS Code + Tinymist. Build: typst compile correction.typ out/correction-{p}.svg
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 10pt, fill: white)
#set text(font: "Noto Sans", size: 9pt)

// One colour per role: robot and runtime, human, data, model.
#let robot = rgb("#e8e8e8")
#let human = rgb("#f6dcc4")
#let data = rgb("#fbf6e3")
#let model = rgb("#d9e4f2")
#let ink = rgb("#333333")

#let box(pos, body, fill: white, name: none, ..args) = node(
  pos, body, name: name, fill: fill, stroke: 0.6pt + ink, corner-radius: 0pt, inset: 6pt, ..args,
)

// Figure 1: the loop. Run the policy, correct it by hand, cut the corrections out of the
// rollout bag, pair each with what the policy would have done, update between runs.
#diagram(
  spacing: (18mm, 9mm),
  edge-stroke: 0.6pt + ink,
  box((0, 0), [Cameras + joints], fill: robot, name: <obs>),
  box((1, 0), [Policy $pi_theta$], fill: model, name: <pi>),
  box((2, 0), [Runtime\ merge chunks], fill: robot, name: <rt>),
  box((3, 0), [Teleop servo\ + arm], fill: robot, name: <arm>),
  box((4, 0), [Human\ cuff or VR], fill: human, name: <hum>),
  box((3, 1), [Rollout bag\ (recorder)], fill: data, name: <bag>),
  box((2, 1), [Extract corrections\ pre / hold / post], fill: data, name: <ext>),
  box((2, 2), [Offline policy answers\ $a_pi = pi_theta (o)$ per frame], fill: model, name: <cf>),
  box((1, 2), [Pairs $(o, a_"human", a_pi)$\ + outcome], fill: data, name: <pairs>),
  box((1, 1), [Update between runs\ $theta arrow.l theta'$], fill: model, name: <upd>),

  edge(<obs>, <pi>, "-|>"),
  edge(<pi>, <rt>, "-|>", [chunk]),
  edge(<rt>, <arm>, "-|>", [target]),
  edge(<arm>, (3, -0.6), (0, -0.6), <obs>, "-|>"),
  edge(<hum>, <arm>, "-|>", [takes over], label-side: left),
  edge(<arm>, <bag>, "-|>", [logged]),
  edge(<bag>, <ext>, "-|>"),
  edge(<ext>, <cf>, "-|>"),
  edge(<cf>, <pairs>, "-|>"),
  edge(<pairs>, <upd>, "-|>"),
  edge(<upd>, <pi>, "-|>", [new checkpoint]),
)

#pagebreak()

// Figure 2: one correction on the timeline. The pre window has a fixed length set by
// reaction time plus one chunk; the hold window is as long as the human needs.
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  set-style(stroke: 0.6pt + ink)
  let seg(x0, x1, fill, title, trains) = {
    rect((x0, 0), (x1, 1.2), fill: fill)
    content(((x0 + x1) / 2, 0.6), align(center, title))
    content(((x0 + x1) / 2, -0.7), align(center, trains))
  }
  content((-1.3, 0.6), [*drives*])
  content((-1.3, -0.7), [*trains*])
  seg(0, 4.4, model, [pre: 1 to 3 s\ policy, cause of the mistake], [weak negatives,\ weight rises to the grab])
  seg(4.4, 11.9, human, [hold: any length\ human; labels: the measured pose, no lead], [positives $a_"human"$ vs. negatives $a_pi$\ (frames where they disagree)])
  seg(11.9, 16.3, model, [post: 2 to 5 s\ policy, did it work?], [trust weight\ for the correction])
  for (x, label) in ((4.4, [cuff squeezed]), (11.9, [released])) {
    line((x, 1.2), (x, 1.7), stroke: (dash: "dashed", thickness: 0.6pt, paint: ink))
    content((x, 2), label)
  }
})

#pagebreak()

// Figure 3: the scoring model. A candidate action chunk is projected into each camera
// (BIND), the image features at those pixels are read, and a head outputs one energy.
// Its gradient in the action generates chunks (Energy Matching); corrections train it
// contrastively: energy down on the human's chunk, up on the policy's.
#diagram(
  spacing: (12mm, 9mm),
  edge-stroke: 0.6pt + ink,
  box((0, 0), [Images], fill: robot, name: <img>),
  box((1, 0), [Frozen encoder\ (DINOv3)], fill: model, name: <enc>),
  box((2, 0), [Feature maps], fill: data, name: <feat>),
  box((0, 1), [Candidate chunk $a$\ 16 x 7], fill: data, name: <a>),
  box((1, 1), [FK + camera\ projection], fill: robot, name: <proj>),
  box((2, 1), [Features at\ projected pixels], fill: data, name: <samp>),
  box((3, 1), [Head\ $V_theta (o, a)$], fill: model, name: <v>),
  box((4, 0.5), [Act: follow $-nabla_a V$\ from noise to a chunk], fill: model, name: <act>),
  box((4, 1.5), [Score: $-V$ ranks\ candidate chunks], fill: model, name: <score>),
  box((3, 2.4), [Train: flow matching on demos\ + contrastive on corrections\ $V(o, a_"human") < V(o, a_pi)$], fill: human, name: <loss>),

  edge(<img>, <enc>, "-|>"),
  edge(<enc>, <feat>, "-|>"),
  edge(<a>, <proj>, "-|>"),
  edge(<proj>, <samp>, "-|>", [pixels]),
  edge(<feat>, <samp>, "-|>", [read]),
  edge(<samp>, <v>, "-|>"),
  edge(<v>, <act>, "-|>"),
  edge(<v>, <score>, "-|>"),
  edge(<loss>, <v>, "-|>", stroke: (dash: "dashed")),
)
