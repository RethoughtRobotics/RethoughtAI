// Multi-Task DiT as trained on SweepTrash with no label lead (multi_task_dit_SweepTrash_Kin-Full_
// inpainted_aug_30k_r1, step 30000), drawn from its config.json and model.safetensors.
// Data: SweepTrash_Kin-Full episode 0 at t = 20 s. The noised chunk is one noise draw at
// diffusion step 50 of 100 under the checkpoint's squaredcos_cap_v2 schedule.
#import "lib.typ": *

#set page(width: auto, height: auto, margin: 10pt, fill: white)
#set text(font: "Noto Sans", size: 8pt)

#let dit = json("assets/dit.json")
#let mp(n) = str(calc.round(n / 1e6, digits: 1)) + " M"
#let snow = text(font: "DejaVu Sans", fill: rgb("#2b7bb9"))[❄]
#let cam-c = rgb("#9cc3e6")
#let txt-c = attn
#let st-c = rgb("#bdbdbd")
#let time-c = rgb("#e7a3a3")

#cetz.canvas(length: 1cm, {
  // Two observation steps per camera (t - 1 behind t), true aspect ratios.
  let cams = (
    ("third_eye", 1.65, [third eye]),
    ("left", 1.375, [left wrist]),
    ("right", 1.375, [right wrist]),
  )
  content((1.2, 11.65), small[3 cameras x 2 steps])
  for ((name, h, label), top) in cams.zip((11.2, 9.25, 7.55)) {
    for (path, dx) in (("assets/" + name + "_prev.jpg", 0.18), ("assets/" + name + ".jpg", 0)) {
      content((dx, top + dx), image(path, width: 2.2cm), anchor: "north-west")
      rect((dx, top + dx - h), (dx + 2.2, top + dx), stroke: line-w)
    }
    content((0.05, top - h + 0.05), box(fill: white, inset: 1.5pt, small(label)), anchor: "south-west")
  }
  arrow((2.5, 8.6), (3.15, 8.6))
  content((2.82, 8.85), dim[$224^2$])

  // CLIP vision encoder, fine-tuned at a tenth of the policy learning rate.
  line((3.2, 6.6), (5.2, 7.4), (5.2, 9.8), (3.2, 10.6), close: true, fill: robot, stroke: line-w)
  content((4.2, 8.9), align(center)[*CLIP*\ ViT-B/16\ #dim(mp(dit.params.clip_vision))])
  content((4.2, 7.75), align(center, dim[trained,\ LR x 0.1]))

  // Its output: 196 patch tokens, all dropped, and the CLS token, kept.
  grid-cells(5.8, 10.6, 14, 14, cw: 0.09, ch: 0.09, fill: rgb("#f2f2f2"))
  line((5.8, 9.34), (7.06, 10.6), stroke: 0.8pt + rgb("#b03030"))
  line((5.8, 10.6), (7.06, 9.34), stroke: 0.8pt + rgb("#b03030"))
  content((6.43, 8.9), align(center, dim[196 patch tokens\ dropped]))
  arrow((5.25, 9.9), (5.75, 9.9))
  arrow((5.25, 7.7), (7.25, 7.7))
  for i in range(6) {
    vector(7.3, 8.75 - i * 0.3, 8, 768, cw: 0.16, ch: 0.2, fill: cam-c)
  }
  content((8.1, 9.25), small[CLS only])
  content((9.0, 7.35), dim[6 x 768], anchor: "west")

  // Task text: frozen CLIP text encoder, learned projection.
  block(0, 4.5, 2.7, 1.0, emph[Sweep the trash into the dustpan and empty it into the bin], fill: data)
  arrow((2.75, 5.0), (3.15, 5.0))
  block(3.2, 4.5, 2.0, 1.0, [CLIP text #snow\ #dim(mp(dit.params.clip_text) + ", frozen")], fill: robot)
  arrow((5.25, 5.0), (5.65, 5.0))
  block(5.7, 4.5, 1.3, 1.0, [linear\ #dim[512 to 512]], fill: model)
  arrow((7.05, 5.0), (7.25, 5.0))
  vector(7.3, 4.9, 8, 512, cw: 0.16, ch: 0.2, fill: txt-c, label: [512])

  // Robot state at both steps.
  vector(0, 3.55, 28, 28, cw: 0.1, ch: 0.2, fill: st-c)
  vector(0, 3.3, 28, 28, cw: 0.1, ch: 0.2, fill: st-c)
  content((2.9, 3.5), dim[state 2 x 28], anchor: "west")

  // The whole observation becomes one vector, drawn to scale (1 cm = 457 numbers).
  let s = 13.0 / 5944
  let segs = ((st-c, 28), (cam-c, 2304), (txt-c, 512)) * 2 + ((time-c, 256),)
  let x = 0
  for (c, n) in segs {
    rect((x, 1.0), (x + n * s, 1.5), fill: c, stroke: 0.4pt + ink)
    x += n * s
  }
  content((6.5, 1.85), small[conditioning vector $c in RR^5944$, to scale])
  content((2.6, 0.7), dim[cameras 2304])
  content((5.6, 0.7), dim[text 512])
  content((9.6, 0.7), dim[step 2: state 28, cameras 2304, text 512])
  content((12.75, 0.7), dim[time 256])
  arrow((6.5, 3.0), (6.5, 2.1))
  content((6.95, 2.55), dim[flatten 2 steps], anchor: "west")

  // Diffusion: clean chunk, one noise draw, noised chunk at step 50 of 100.
  content((11.6, 11.15), small[clean $x_0$, 32 x 14])
  heat-table(10.3, 10.75, dit.x0, cw: 0.18, ch: 0.12, limits: (-1, 1))
  content((10.93, 10.88), dim[left])
  content((12.19, 10.88), dim[right])
  content((14.85, 11.15), small[noised $x_t$, $t = 50$])
  heat-table(13.6, 10.75, dit.xt, cw: 0.18, ch: 0.12, limits: (-2, 2))
  content((14.23, 10.88), dim[left])
  content((15.49, 10.88), dim[right])
  arrow((12.88, 8.8), (13.55, 8.8))
  content((13.2, 9.05), dim[$+ epsilon$])
  content((12.95, 6.6), dim[$sqrt(macron(alpha)_t) = #calc.round(calc.sqrt(dit.abar_t), digits: 2)$])

  // Into the DiT: 32 tokens of 512.
  block(16.9, 8.3, 1.5, 1.0, [linear\ #dim[14 to 512]], fill: model)
  arrow((16.2, 8.8), (16.85, 8.8))
  grid-cells(18.8, 9.7, 32, 6, cw: 0.15, ch: 0.06, fill: feat)
  content((19.25, 10.0), dim[32 x 512])
  arrow((18.45, 8.8), (18.75, 8.8))

  // One DiT block, each part's area proportional to its parameters.
  let (bx0, bx1) = (20.2, 25.0)
  let w = bx1 - bx0 - 0.4
  let k = 1e6 * 4.6
  let per = dit.params.adaln / 6
  let rest = dit.params.attn_mlp / 6
  let h-ada = per / k / w * 4.6
  let h-att = rest / k / w * 4.6
  rect((bx0, 1.0), (bx1, 10.2), stroke: line-w)
  content(((bx0 + bx1) / 2, 10.5), small[DiT block x 6, area $prop$ parameters])
  rect((bx0 + 0.2, 8.4), (bx1 - 0.2, 8.4 + h-att), fill: attn, stroke: line-w)
  content(((bx0 + bx1) / 2, 8.4 + h-att + 0.3), dim[RoPE self-attention + MLP, 8 heads: #mp(rest)])
  rect((bx0 + 0.2, 2.0), (bx1 - 0.2, 2.0 + h-ada), fill: time-c.lighten(40%), stroke: line-w)
  content(((bx0 + bx1) / 2, 2.0 + h-ada / 2), align(center)[*adaLN*\ Linear(5944, 3072): #mp(per)\ #dim[shift, scale, gate for attention and MLP]])
  arrow(((bx0 + bx1) / 2, 2.05 + h-ada), ((bx0 + bx1) / 2, 8.35))
  content(((bx0 + bx1) / 2 + 0.15, (2.0 + h-ada + 8.4) / 2), dim[modulate], anchor: "west")
  arrow((19.75, 8.8), (20.35, 8.8))
  content((22.6, 1.4), dim[#calc.round(per / (per + rest) * 100) % of each block reads $c$])
  arrow((13.05, 1.25), (19.6, 1.25), (19.6, 4.0), (20.35, 4.0))

  // Out: predicted noise, trained against the drawn noise.
  block(25.4, 8.3, 1.5, 1.0, [linear\ #dim[512 to 14]], fill: model)
  arrow((24.85, 8.8), (25.35, 8.8))
  content((28.4, 10.15), small[$hat(epsilon)$ vs. $epsilon$, 32 x 14])
  heat-table(27.4, 9.75, dit.eps, cw: 0.14, ch: 0.09, limits: (-2.5, 2.5))
  arrow((26.95, 8.8), (27.35, 8.8))
  content((28.4, 6.55), align(center, small[$cal(L) = norm(epsilon - hat(epsilon))^2$]))
  content((28.4, 5.6), align(center, dim[sampling: 100 DDPM steps\ from $x_100 tilde cal(N)(0, I)$;\ runs 24 of 32 steps]))

  // Where the 278 M parameters are.
  let rows = (
    ([CLIP vision], dit.params.clip_vision, [LR x 0.1]),
    ([CLIP text], dit.params.clip_text, [frozen]),
    ([DiT adaLN x 6], dit.params.adaln, [trained]),
    ([DiT attention + MLP x 6], dit.params.attn_mlp, [trained]),
    ([text projection, time MLP, in/out], dit.params.text_proj + dit.params.time_mlp + dit.params.io_proj, [trained]),
  )
  let total = rows.map(r => r.at(1)).sum()
  content((25.6, 3.9), anchor: "north-west", text(size: 6.5pt, table(
    columns: 3,
    stroke: 0.4pt + ink,
    inset: 3pt,
    align: (left, right, left),
    [*module*], [*params*], [*training*],
    ..rows.map(((m, n, t)) => (m, mp(n), t)).flatten(),
    [*total*], [*#mp(total)*], [],
  )))
})
