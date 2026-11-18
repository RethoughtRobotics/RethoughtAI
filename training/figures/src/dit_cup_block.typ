// Inside the cup Multi-Task DiT's diffusion transformer (multitaskdit_BlueCupToContainer, step 30000),
// from LeRobot's DiffusionTransformer and TransformerBlock and the checkpoint's config: hidden 512,
// 4 blocks, 8 heads, RoPE, time embedding 256, adaLN-Zero (Peebles and Xie, 2023, Fig. 3), epsilon
// prediction, 10 DDPM steps at inference. Left: the denoising loop over action tokens; right: one block.
// LoRA (rethought-lerobot --lora_rank) wraps attn.qkv_proj, attn.out_proj, mlp.0 and mlp.2 of every
// block; adaLN is not adapted.
#import "lib.typ": *

#set page(width: auto, height: auto, margin: 12pt, fill: white)
#set text(font: "Noto Sans", size: 8pt, fill: ink)

#let lora-c = rgb("#e8833a")
#let cond-c = rgb("#f0d2d2")
#let tok-c = rgb("#d6d6d6")
#let dit-c = rgb("#dbe7f4")
#let label(body) = text(size: 7.5pt, weight: "bold", body)
#let lora-tab(x, y) = {
  rect((x, y), (x + 0.7, y + 0.34), fill: lora-c, stroke: 0.4pt + ink)
  content((x + 0.35, y + 0.17), text(size: 5.5pt, fill: white, weight: "bold")[LoRA])
}
#let plus(x, y) = {
  circle((x, y), radius: 0.2, fill: white, stroke: line-w)
  content((x, y), text(size: 8pt)[+])
}
#let token(x, y, body) = {
  rect((x, y), (x + 0.55, y + 0.75), fill: tok-c, stroke: 0.4pt + ink)
  content((x + 0.275, y + 0.375), text(size: 6.5pt, body))
}

#cetz.canvas(length: 1cm, {
  // ---- Left: the denoising loop over action tokens (as LeRobot draws the Multi-Task DiT). ----
  let (lx0, lx1) = (0.0, 6.4)
  rect((lx0, 3.6), (lx1, 7.4), fill: dit-c, stroke: line-w)
  content(((lx0 + lx1) / 2, 6.1), label[Diffusion transformer])
  content(((lx0 + lx1) / 2, 5.55), dim[Linear 7 to 512, 4 blocks (right),\ Linear 512 to 7])
  content(((lx0 + lx1) / 2, 4.5), dim[predicts the noise $hat(epsilon)$ in each token])
  let xs = (0.3, 1.3, 2.3, 4.4, 5.5)
  let names = ([$a_1^k$], [$a_2^k$], [$a_3^k$], [$a_31^k$], [$a_32^k$])
  for (x, n) in xs.zip(names) {
    token(x, 2.1, n)
    arrow((x + 0.275, 2.9), (x + 0.275, 3.55))
  }
  content((3.55, 2.47), text(size: 9pt)[…])
  content((3.2, 1.6), align(center, small[noisy action tokens at step $k$: 32 steps of 7\ (x y z, rotation, gripper)]))
  for (x, n) in xs.zip(([$hat(epsilon)_1$], [$hat(epsilon)_2$], [$hat(epsilon)_3$], [$hat(epsilon)_31$], [$hat(epsilon)_32$])) {
    arrow((x + 0.275, 7.45), (x + 0.275, 8.05))
    token(x, 8.1, n)
  }
  content((3.55, 8.47), text(size: 9pt)[…])
  content((3.2, 9.25), align(center, small[DDPM step: $a^(k-1) = $ step$(a^k, hat(epsilon))$;\ 10 steps from noise, $a^0$ is the plan]))
  arrow((lx1 + 0.05, 8.5), (lx1 + 0.6, 8.5), (lx1 + 0.6, 2.5), (lx1 + 0.05, 2.5), dashed: true)
  content((lx1 + 0.85, 5.5), angle: 90deg, dim[repeat with $k - 1$])

  // Conditioning: the step k and the observation embedding z.
  block(0.0, 0.0, 1.5, 0.8, [step $k$], fill: data)
  block(1.9, 0.0, 1.9, 0.8, [sinusoid + MLP\ 256])
  arrow((1.55, 0.4), (1.85, 0.4))
  block(4.2, 0.0, 1.5, 0.8, [$z$: 4,130], fill: rgb("#d9e4f2"))
  content((7.05, -0.45), small[$c = [k, z]$: 4,386, to every block], anchor: "west")
  line((3.85, 0.4), (4.0, 0.4), (4.0, -0.3), (7.4, -0.3), stroke: line-w)
  line((5.75, 0.4), (6.0, 0.4), (6.0, -0.3), stroke: line-w)

  // ---- Right: one block, adaLN-Zero (Peebles and Xie, 2023). ----
  let bx = 9.6  // left edge of the block's main column
  let w = 3.2
  let cx = bx + w / 2
  rect((bx - 0.8, 0.2), (bx + w + 3.5, 11.4), stroke: (paint: ink, thickness: 0.6pt, dash: "dashed"))
  content((cx + 1.0, 11.1), label[one block (× 4), 32 tokens of 512])
  content((cx, 0.55), small[tokens in])
  arrow((cx, 0.75), (cx, 1.05))
  let layers = (
    (1.1, 0.6, [LayerNorm], white, none),
    (1.95, 0.6, [scale, shift], cond-c, $gamma_1, beta_1$),
    (2.85, 1.0, [Self-attention\ #dim[8 heads, RoPE,\ across the 32 steps]], dit-c, none),
    (4.1, 0.6, [scale (gate)], cond-c, $alpha_1$),
    (5.55, 0.6, [LayerNorm], white, none),
    (6.4, 0.6, [scale, shift], cond-c, $gamma_2, beta_2$),
    (7.3, 1.0, [MLP\ #dim[512 to 2,048 to 512, GELU]], dit-c, none),
    (8.55, 0.6, [scale (gate)], cond-c, $alpha_2$),
  )
  for (i, (y, h, body, fill, tap)) in layers.enumerate() {
    rect((bx, y), (bx + w, y + h), fill: fill, stroke: line-w)
    content((cx, y + h / 2), align(center, text(size: 7pt, body)))
    if i + 1 < layers.len() and i != 3 { arrow((cx, y + h + 0.02), (cx, layers.at(i + 1).at(0) - 0.02)) }
  }
  lora-tab(bx + w + 0.15, 3.18)
  lora-tab(bx + w + 0.15, 7.63)
  content((bx + w + 0.95, 3.35), dim[qkv, out], anchor: "west")
  content((bx + w + 0.95, 7.8), dim[both layers], anchor: "west")
  // residuals
  plus(cx, 5.1)
  arrow((cx, 4.72), (cx, 4.88))
  arrow((cx, 5.32), (cx, 5.53))
  line((cx, 0.9), (bx - 0.4, 0.9), (bx - 0.4, 5.1), stroke: line-w)
  arrow((bx - 0.4, 5.1), (cx - 0.22, 5.1))
  plus(cx, 9.55)
  arrow((cx, 9.17), (cx, 9.33))
  line((cx, 5.38), (bx - 0.2, 5.38), (bx - 0.2, 9.55), stroke: line-w)
  arrow((bx - 0.2, 9.55), (cx - 0.22, 9.55))
  arrow((cx, 9.77), (cx, 10.3))
  content((cx, 10.5), small[tokens out])

  // adaLN: c through SiLU + Linear gives six 512-vectors, one per scale, shift or gate.
  let ax = bx + w + 2.4
  block(ax - 1.2, 0.3, 2.4, 1.15, [adaLN\ #dim[SiLU + Linear,\ 4,386 to 6 × 512]], fill: cond-c)
  arrow((7.4, -0.3), (ax, -0.3), (ax, 0.3))
  line((ax, 1.47), (ax, 8.85), stroke: (paint: rgb("#b05a5a"), thickness: 0.7pt))
  for (i, (y, h, body, fill, tap)) in layers.enumerate() {
    if tap != none {
      line((ax, y + h / 2), (bx + w + 0.05, y + h / 2), stroke: (paint: rgb("#b05a5a"), thickness: 0.7pt), mark: (end: "stealth", fill: rgb("#b05a5a"), scale: 0.55))
      content((ax - 0.95, y + h / 2 + 0.17), text(size: 6.5pt, fill: rgb("#b05a5a"), tap))
    }
  }

  // Legend.
  lora-tab(0, -1.45)
  content((0.85, -1.28), small[LoRA, trained on the corrections; everything else frozen], anchor: "west")
  rect((7.4, -1.45), (7.75, -1.11), fill: cond-c, stroke: 0.4pt + ink)
  content((7.85, -1.28), small[set by the step $k$ and the observation $z$ (adaLN-Zero)], anchor: "west")
})
