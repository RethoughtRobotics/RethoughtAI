// Test-time training on VITA: one frozen policy, one LoRA adapter, two training signals.
// Top: the shared policy (blend = LoRA scale alpha). Middle: design A, the corrective loop.
// Bottom: design B, reward-weighted RL. A and B differ only in where the LoRA's targets come from.
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#set page(width: auto, height: auto, margin: 10pt, fill: white)
#set text(font: "Noto Sans", size: 8.5pt)

#let ink = rgb("#333333")
#let robot = rgb("#e8e8e8")
#let human = rgb("#f6dcc4")
#let data = rgb("#fbf6e3")
#let model = rgb("#d9e4f2")
#let new = rgb("#eb6834")
#let dim(body) = text(size: 7pt, fill: rgb("#555555"), body)
#let box(pos, body, fill: white, name: none, stroke: 0.6pt + ink, ..args) = node(
  pos, body, name: name, fill: fill, stroke: stroke, corner-radius: 0pt, inset: 6pt, ..args,
)
#let lora = 1.4pt + new

#diagram(
  spacing: (11mm, 8mm),
  edge-stroke: 0.6pt + ink,

  // The policy: everything frozen except the LoRA, whose scale is the blend.
  node((0, -0.75), align(left)[*Policy at test time*], width: 40mm),
  box((0, 0), [Cameras + state], fill: robot, name: <obs>),
  box((1, 0), [ResNet-18 + linear\ #dim[frozen, cached]], fill: model, name: <enc>),
  box((2, 0), [$z_0 in RR^512$], fill: data, name: <z0>),
  box((3, 0), [Flow net $W_"base"$\ #dim[frozen, 12.6 M]], fill: model, name: <flow>),
  box((3, 1), [LoRA $alpha dot B A$\ #dim[rank 16, 0.33 M: the correcting model]], fill: white, stroke: lora, name: <lora>),
  box((4, 0), [$hat(z)_1$], fill: data, name: <z1>),
  box((5, 0), [Decoder\ #dim[frozen]], fill: model, name: <dec>),
  box((6, 0), [Arm], fill: robot, name: <arm>),
  box((1.5, 1), [Blend: $alpha = 0$ base,\ $alpha = 1$ corrected], fill: white, stroke: lora, name: <blend>),
  edge(<obs>, <enc>, "-|>"),
  edge(<enc>, <z0>, "-|>"),
  edge(<z0>, <flow>, "-|>"),
  edge(<flow>, <z1>, "-|>", [6 steps]),
  edge(<z1>, <dec>, "-|>"),
  edge(<dec>, <arm>, "-|>"),
  edge(<lora>, <flow>, "-|>", stroke: lora),
  edge(<blend>, <lora>, "-|>", stroke: lora),

  // Design A: the human's chunk is the target.
  node((0, 2.25), align(left)[*A: corrective loop*], width: 40mm),
  box((0, 3), [Human takes over\ #dim[cuff or VR]], fill: human, name: <hum>),
  box((1, 3), [Hold frames\ #dim[pre / hold / post]], fill: data, name: <hold>),
  box((2, 3), [Pairs $(z_0, z_"human")$\ #dim[$z_"human"$ = frozen action encoder]], fill: data, name: <pairsA>),
  box((3, 3), [Loss: flow $z_0$ to $z_"human"$\ + 50% demo anchor], fill: human, name: <lossA>),
  edge(<hum>, <hold>, "-|>"),
  edge(<hold>, <pairsA>, "-|>"),
  edge(<pairsA>, <lossA>, "-|>"),
  edge(<lossA>, <lora>, "-|>", [update on release], stroke: lora),

  // Design B: the policy's own good attempts are the target.
  node((0, 4.25), align(left)[*B: reward-weighted RL*], width: 40mm),
  box((0, 5), [$K = 8$ attempts\ $z_0 + sigma epsilon_i$, $epsilon_i$ fixed per attempt], fill: robot, name: <att>),
  box((1, 5), [Reward $r_i$\ #dim[success press + 0.5 progress]], fill: human, name: <rew>),
  box((2, 5), [$w = "softmax"(beta r)$\ #dim[$beta$: KL$(w || U) = ln 2$]], fill: data, name: <w>),
  box((3, 5), [Loss: $sum_i w_i$ flow $z_0$ to $hat(z)_(1, i)$\ + 50% demo anchor], fill: human, name: <lossB>),
  edge(<att>, <rew>, "-|>"),
  edge(<rew>, <w>, "-|>"),
  edge(<w>, <lossB>, "-|>"),
  edge(<lossB>, (4.4, 5), (4.4, 1), <lora>, "-|>", [update every $K$ attempts], label-pos: 0.45, label-side: left, stroke: lora),
)
