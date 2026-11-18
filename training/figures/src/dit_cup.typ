// Multi-Task DiT for the cup task (multitaskdit_BlueCupToContainer, step 30000) and where its
// correction LoRA sits, drawn for overview: real inputs, the output as a real plan.
// One CLIP image encoder with shared weights runs on each camera image on its own
// (use_separate_rgb_encoder_per_camera false: LeRobot ObservationEncoder.encode flattens
// "b s n c h w -> (b s n) c h w" and keeps each image's CLS token), so it is drawn once per camera.
// z is drawn in input order, not to scale; its true layout per frame is [state 17, wrist 768,
// third eye 768, text 512], for frames t-1 and t: 4,130.
// Frames and the plan: assets/make_cup_assets.py (blue cup, base policy, 2026-10-28).
#import "lib.typ": *

#set page(width: auto, height: auto, margin: 12pt, fill: white)
#set text(font: "Noto Sans", size: 8pt, fill: ink)

#let chunk = json("assets/cup_chunk.json")
#let wrist-c = rgb("#a9cdee")
#let third-c = rgb("#5e93c9")
#let text-c = rgb("#f7e08a")
#let state-c = rgb("#b5b5b5")
#let lora-c = rgb("#e8833a")
#let expert-c = rgb("#eaf1f8")
#let frozen-c = rgb("#f2f2f2")
#let label(body) = text(size: 7.5pt, weight: "bold", body)
#let snow = text(font: "DejaVu Sans", fill: rgb("#2b7bb9"))[❄]
#let tag(body) = box(fill: white, inset: 1.5pt, text(size: 6pt, body))

#cetz.canvas(length: 1cm, {
  // Rows: each input, its encoder and its part of z share a height.
  let rows = (
    third: (6.1, 8.3, third-c),
    wrist: (3.7, 6.1, wrist-c),
    text: (2.0, 3.7, text-c),
    state: (1.2, 2.0, state-c),
  )
  let mid(k) = (rows.at(k).at(0) + rows.at(k).at(1)) / 2

  // Inputs.
  let img = 2.0
  for (k, path, lab) in (("third", "assets/cup_third_eye.jpg", [third-eye camera]), ("wrist", "assets/cup_left.jpg", [wrist camera])) {
    let top = mid(k) + img / 2
    content((0, top), image(path, width: img * 1cm, height: img * 1cm), anchor: "north-west")
    rect((0, top - img), (img, top), stroke: line-w)
    content((0.05, top - img + 0.05), tag(lab), anchor: "south-west")
  }
  rect((0, mid("text") - 0.6), (img, mid("text") + 0.6), fill: data, stroke: line-w)
  content((img / 2, mid("text")), box(width: 1.85cm, align(center, text(size: 6pt, style: "italic")[pick up the blue cup and place it in the red container])))
  for i in range(10) { rect((i * 0.2, mid("state") - 0.11), (0.2 + i * 0.2, mid("state") + 0.11), fill: state-c, stroke: 0.4pt + ink) }
  content((img / 2, mid("state") - 0.38), small[robot state])

  // Encoders: the same CLIP for each camera image, on its own; CLIP text frozen; state as is.
  let (e0, e1) = (2.8, 4.9)
  for k in ("third", "wrist") {
    rect((e0, mid(k) - 0.75), (e1, mid(k) + 0.75), fill: model, stroke: line-w)
    content(((e0 + e1) / 2, mid(k)), align(center)[#label[CLIP ViT-B/16]\ #dim[image encoder]])
    arrow((img + 0.05, mid(k)), (e0 - 0.05, mid(k)))
  }
  line(((e0 + e1) / 2, mid("third") - 0.75), ((e0 + e1) / 2, mid("wrist") + 0.75), stroke: (paint: ink, thickness: 0.6pt, dash: "dashed"))
  content(((e0 + e1) / 2 + 0.1, rows.wrist.at(1)), box(fill: white, inset: 2pt, dim[shared weights]), anchor: "west")
  rect((e0, mid("text") - 0.45), (e1, mid("text") + 0.45), fill: frozen-c, stroke: line-w)
  content(((e0 + e1) / 2, mid("text")), align(center)[#label[CLIP text] #snow\ #dim[frozen]])
  arrow((img + 0.05, mid("text")), (e0 - 0.05, mid("text")))

  // z: one vector per frame, the parts concatenated; the DiT reads frames t-1 and t.
  let (zx0, zx1) = (5.8, 6.9)
  for (k, n) in (("third", "768"), ("wrist", "768"), ("text", "512"), ("state", "17")) {
    let (a, b, c) = rows.at(k)
    rect((zx0, a), (zx1, b), fill: c, stroke: line-w)
    content(((zx0 + zx1) / 2, (a + b) / 2), text(size: 6.5pt, fill: if k == "third" { white } else { ink }, n))
  }
  for k in ("third", "wrist", "text") { arrow((e1 + 0.05, mid(k)), (zx0 - 0.05, mid(k))) }
  arrow((img + 0.05, mid("state")), (zx0 - 0.05, mid("state")))
  content(((zx0 + zx1) / 2, 9.05), align(center, small[$z$, one frame\ (concatenated)]))
  content(((zx0 + zx1) / 2, 0.75), align(center, dim[× 2 frames: 4,130]))

  // Action expert: the diffusion transformer, its blocks conditioned on z; LoRA beside each layer.
  let ex = 7.7
  let (eb, et) = (0.4, 8.7)
  rect((ex, eb), (ex + 5.6, et), fill: expert-c, stroke: line-w)
  content((ex + 2.8, et - 0.45), label[Diffusion transformer])
  content((ex + 2.8, et - 0.85), dim[action expert: 4 blocks, each conditioned on $z$])
  for k in range(4) {
    let y = eb + 0.45 + k * 1.75
    rect((ex + 0.5, y), (ex + 3.9, y + 1.05), fill: white, stroke: line-w)
    rect((ex + 0.65, y + 0.15), (ex + 2.15, y + 0.9), fill: model, stroke: 0.5pt + ink)
    content((ex + 1.4, y + 0.525), small[attention])
    rect((ex + 2.3, y + 0.15), (ex + 3.75, y + 0.9), fill: model, stroke: 0.5pt + ink)
    content((ex + 3.025, y + 0.525), small[MLP])
    rect((ex + 4.05, y + 0.25), (ex + 4.75, y + 0.8), fill: lora-c, stroke: 0.5pt + ink)
    content((ex + 4.4, y + 0.525), text(size: 6pt, fill: white, weight: "bold")[LoRA])
    line((ex + 3.9, y + 0.525), (ex + 4.05, y + 0.525), stroke: (paint: lora-c, thickness: 0.8pt))
  }
  let zc = 4.6
  arrow((zx1 + 0.05, zc), (ex - 0.05, zc))

  // Noise in, a plan out, refined 10 times.
  for i in range(6) { for j in range(3) {
    let v = calc.rem(i * 7 + j * 13, 9) / 9
    rect((ex + 1.3 + i * 0.22, eb - 1.25 + j * 0.2), (ex + 1.52 + i * 0.22, eb - 1.05 + j * 0.2), fill: color.mix((wrist-c, v * 100%), (white, 100% - v * 100%)), stroke: 0.3pt + ink)
  }}
  content((ex + 3.0, eb - 0.95), dim[noise, refined in 10 denoising steps], anchor: "west")
  arrow((ex + 1.95, eb - 0.65), (ex + 1.95, eb - 0.05))

  // The plan, drawn from top view: 24 waypoints of the cup policy (x forward, y left).
  let px = 14.9
  let scale = 75  // cm of drawing per m of motion: the plan spans about 2 cm
  let pts = chunk.xyz_mm.map(p => (px + 0.4 + p.at(0) / 1000 * scale, zc - 0.75 + p.at(1) / 1000 * scale))
  arrow((ex + 5.65, zc), (px - 0.25, zc))
  for (i, p) in pts.enumerate() {
    let f = i / (pts.len() - 1)
    circle(p, radius: 0.09, fill: color.mix((lora-c, (1 - f) * 100%), (third-c, f * 100%)), stroke: none)
  }
  line(..pts, stroke: 0.5pt + ink)
  content((px + 1.6, zc + 1.6), label[Action chunk])
  content((px + 1.6, zc + 1.05), dim[24 TCP waypoints, 0.96 s,\ + gripper])
  content((px + 1.6, zc - 1.7), dim[top view at 0.75 scale, start orange])

  // To the arm.
  arrow((px + 3.4, zc), (px + 4.1, zc))
  rect((px + 4.15, zc - 1.0), (px + 7.15, zc + 1.0), fill: robot, stroke: line-w)
  content((px + 5.65, zc), align(center)[#label[Baxter]\ #dim[a new plan every 0.16 s;\ overlapping plans\ are averaged]])

  // Legend.
  let ly = eb - 2.0
  rect((0, ly), (0.35, ly + 0.25), fill: lora-c, stroke: 0.4pt + ink)
  content((0.45, ly + 0.125), small[LoRA, trained on the corrections (0.5 M of 217 M)], anchor: "west")
  rect((7.7, ly), (8.05, ly + 0.25), fill: model, stroke: 0.4pt + ink)
  content((8.15, ly + 0.125), small[trained on demonstrations, then frozen], anchor: "west")
  rect((13.7, ly), (14.05, ly + 0.25), fill: frozen-c, stroke: 0.4pt + ink)
  content((14.15, ly + 0.125), small[pre-trained, frozen], anchor: "west")
})
