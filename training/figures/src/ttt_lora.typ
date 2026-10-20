// Test-time training on VITA at tensor level. Left: one flow-net layer with its LoRA adapters, every
// matrix drawn to scale (1 cm = 512). Middle: design A's training pair. Right: design B's group of
// perturbed starts and their entropic weights. z0 and the demo latent are the checkpoint's own
// (assets/vita.json); B's rewards are illustrative; its weights are computed by the ln 2 rule.
#import "lib.typ": *

#set page(width: auto, height: auto, margin: 10pt, fill: white)
#set text(font: "Noto Sans", size: 8pt)

#let t = json("assets/ttt.json")
#let new = 1.2pt + rgb("#eb6834")
#let orange = rgb("#eb6834")
#let lim = 0.1

// beta such that KL(softmax(beta r) || uniform) = ln 2, by bisection.
#let weights(r, beta) = {
  let m = calc.max(..r)
  let e = r.map(x => calc.exp(beta * (x - m)))
  let s = e.sum()
  e.map(x => x / s)
}
#let kl(w) = w.map(x => if x > 0 { x * calc.ln(x * w.len()) } else { 0 }).sum()
#let solve(r) = {
  let (lo, hi) = (0.0, 100.0)
  for _ in range(60) {
    let mid = (lo + hi) / 2
    if kl(weights(r, mid)) < calc.ln(2) { lo = mid } else { hi = mid }
  }
  (lo + hi) / 2
}
#let bt = solve(t.rewards)
#let w = weights(t.rewards, bt)

#cetz.canvas(length: 1cm, {
  // One flow-net layer: x + gamma * fc2(GELU(fc1(modulated LayerNorm(x)))), x 4 layers.
  let s = 1 / 512
  content((2.6, 6.9), small[one flow-net layer (x 4), matrices to scale])
  vector(0, 3.0, 1, 512, cw: 0.22, ch: s * 512, fill: data)
  content((0.11, 2.7), dim[$x$ 512])
  block(0.45, 3.2, 0.9, 0.6, [LN, $t$], fill: model)
  arrow((1.4, 3.5), (1.7, 3.5))
  // fc1: 2048 x 512, frozen, plus B (2048 x 16) A (16 x 512).
  rect((1.75, 1.5), (1.75 + 512 * s, 1.5 + 2048 * s), fill: model, stroke: line-w)
  content((2.25, 3.5), align(center, dim[$W_1$\ 2048\ x 512\ frozen]))
  rect((2.95, 1.5), (2.95 + 16 * s, 1.5 + 2048 * s), fill: orange, stroke: new)
  rect((3.15, 5.7), (3.15 + 512 * s, 5.7 + 16 * s), fill: orange, stroke: new)
  content((3.0, 5.75), dim[$B_1$], anchor: "east")
  content((3.65, 5.95), dim[$A_1$: 16 x 512])
  content((3.08, 1.25), dim[2048 x 16])
  content((2.6, 6.45), text(fill: orange, size: 6.5pt)[$+ alpha B_1 A_1$])
  arrow((3.1, 3.5), (3.4, 3.5))
  block(3.45, 3.2, 0.9, 0.6, [GELU], fill: model)
  arrow((4.4, 3.5), (4.7, 3.5))
  // fc2: 512 x 2048, frozen, plus its adapter.
  rect((4.75, 3.0), (4.75 + 2048 * s, 3.0 + 512 * s), fill: model, stroke: line-w)
  content((6.75, 2.75), dim[$W_2$: 512 x 2048, frozen])
  rect((4.75, 4.15), (4.75 + 2048 * s, 4.15 + 16 * s), fill: orange, stroke: new)
  rect((8.85, 3.0), (8.85 + 16 * s, 3.0 + 512 * s), fill: orange, stroke: new)
  content((6.75, 4.45), text(fill: orange, size: 6.5pt)[$+ alpha B_2 A_2$])
  arrow((8.95, 3.5), (9.25, 3.5))
  content((9.6, 3.5), [$dot gamma + x$])
  content((4.8, 0.6), align(center, dim[frozen $W_1 + W_2$: 2.1 M per layer; LoRA (orange): 82 k per layer,\ 0.33 M in all; $alpha$ is the blend]))

  // Design A: one training pair per held frame.
  let ax = 11.1
  content((ax + 2.5, 6.9), small[A: corrective loop, one pair per held frame])
  strip(ax, 5.6, t.z0, w: 5, limit: lim)
  content((ax - 0.1, 5.71), dim[$z_0$], anchor: "east")
  strip(ax, 4.9, t.z_demo, w: 5, limit: lim)
  content((ax - 0.1, 5.01), dim[$z_"human"$], anchor: "east")
  content((ax + 2.5, 4.55), dim[here the demo's latent stands in for a human's chunk])
  block(ax, 2.9, 5, 1.2, [$cal(L)_A = norm(f_(theta + alpha B A)(z_t, t) - (z_"human" - z_0))^2$\ #dim[\+ the same loss on cached demo pairs, half of each batch]], fill: human)
  content((ax + 2.5, 2.4), dim[gradients reach only $A$, $B$; ResNet features are cached])

  // Design B: K = 8 perturbed starts, rewards, entropic weights.
  let bx = 18.3
  content((bx + 3.0, 6.9), small[B: reward-weighted RL, one group of $K = 8$ attempts])
  for (i, z) in t.starts.enumerate() {
    let y = 6.2 - i * 0.32
    strip(bx, y, z, w: 4, h: 0.26, limit: lim)
    content((bx - 0.1, y + 0.13), dim[$i = #(i + 1)$], anchor: "east")
    let r = t.rewards.at(i)
    rect((bx + 4.2, y + 0.03), (bx + 4.2 + r * 0.6, y + 0.23), fill: human, stroke: 0.4pt + ink)
    rect((bx + 5.4, y + 0.03), (bx + 5.4 + w.at(i) * 3, y + 0.23), fill: orange, stroke: 0.4pt + ink)
  }
  content((bx + 2, 6.65), dim[$z_0 + sigma epsilon_i$, $sigma = #t.sigma$])
  content((bx + 4.6, 6.65), dim[reward])
  content((bx + 6.0, 6.65), dim[weight])
  content((bx + 3.5, 3.3), dim[rewards illustrative; weights computed: $beta = #calc.round(bt, digits: 2)$ puts KL$(w || U)$ at $ln 2$])
  block(bx, 1.5, 7.0, 1.3, [$cal(L)_B = sum_i w_i norm(f_(theta + alpha B A)(z_t, t) - (hat(z)_(1, i) - z_0))^2$\ #dim[learn to reach, without noise, the latents the good attempts reached with it; + demo anchor]], fill: human)
})
