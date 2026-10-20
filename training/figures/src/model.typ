// The scoring model V(o, a), with every tensor at its real size.
// Data: SweepTrash_Kin-Full episode 0 at t = 20 s (three camera frames and the 16-step
// action chunk that follows), from assets/. Sizes past the encoder are design choices.
#import "lib.typ": *

#set page(width: auto, height: auto, margin: 10pt, fill: white)
#set text(font: "Noto Sans", size: 8pt)

#let chunk = json("assets/chunk.json")
#let snow = text(font: "DejaVu Sans", fill: rgb("#2b7bb9"))[❄]

#cetz.canvas(length: 1cm, {
  // Cameras: real frames, true aspect ratios.
  let cams = (
    ("assets/third_eye.jpg", 1.95, [third eye 480 x 640]),
    ("assets/left.jpg", 1.625, [left wrist 400 x 640]),
    ("assets/right.jpg", 1.625, [right wrist 400 x 640]),
  )
  let tops = (11.0, 8.85, 7.0)
  for ((path, h, name), top) in cams.zip(tops) {
    content((0, top), image(path, width: 2.6cm), anchor: "north-west")
    rect((0, top - h), (2.6, top), stroke: line-w)
    content((0.05, top - h + 0.05), box(fill: white, inset: 1.5pt, small(name)), anchor: "south-west")
  }
  arrow((2.7, 8.2), (3.35, 8.2))
  content((3.02, 8.45), dim[$224^2$])

  // Frozen encoder.
  line((3.4, 5.6), (5.4, 6.4), (5.4, 10.0), (3.4, 10.8), close: true, fill: robot, stroke: line-w)
  content((4.4, 8.4), align(center)[*DINOv3*\ ViT-S/16\ #dim[21 M]])
  content((4.4, 7.2), [#snow #small[frozen]])
  arrow((5.45, 8.2), (6.1, 8.2))

  // Patch features, one tensor per camera, and a CLS vector per camera.
  for (i, y) in (9.45, 7.6, 5.6).enumerate() {
    tensor(6.3, y, 1.0, 1.0, 0.9, label: [384 x 14 x 14])
  }
  content((7.1, 11.15), small[patch features])
  arrow((7.95, 10.4), (15.1, 10.4), (15.1, 4.77), (15.35, 4.77))
  content((10.6, 10.65), dim[CLS token per camera, 3 x 384])

  // Candidate chunk: the real 16 x 14 goal poses.
  let hdr = ("x", "y", "z", "qx", "qy", "qz", "qw") * 2
  content((1.89, 5.05), small[candidate chunk $a in RR^(16 times 14)$])
  content((0.95, 4.72), dim[left goal])
  content((2.84, 4.72), dim[right goal])
  heat-table(0, 4.35, chunk.chunk, cw: 0.27, ch: 0.2, header: hdr)
  content((-0.3, 2.75), std.rotate(-90deg, reflow: true, dim[16 steps, 0.64 s]))
  content((1.89, 0.95), dim[SweepTrash ep 0, t = 20 s; colour per column])

  // State drives the wrist cameras' poses.
  vector(0, 0.2, 28, 28, cw: 0.135, ch: 0.22, fill: robot)
  content((1.89, -0.12), dim[state $s in RR^28$: per arm TCP 7 + joints 7])
  block(5.0, 0.0, 2.4, 1.1, [FK$(q)$: wrist\ camera poses], fill: robot)
  arrow((3.85, 0.31), (4.95, 0.31))

  // Project each goal position into each camera.
  block(5.0, 2.2, 2.4, 1.4, [project $p_(x y z)$\ $u = pi(K [R | t] p)$], fill: robot)
  arrow((3.85, 2.9), (4.95, 2.9))
  arrow((6.2, 1.15), (6.2, 2.15))
  content((6.75, 1.65), dim[$[R | t]$])

  // Pixel coordinates: 3 cameras x 32 points (2 arms x 16 steps) x (u, v).
  stacked(3, 0.13, 0.13, (dx, dy) => grid-cells(8.0 + dx, 3.95 + dy, 16, 4, cw: 0.18, ch: 0.12))
  content((8.55, 1.75), dim[pixels: 3 x 32 x 2])
  arrow((7.45, 2.9), (7.95, 2.9))

  // Read the patch features at those pixels.
  block(9.9, 2.2, 1.9, 1.4, [bilinear\ sample], fill: model)
  arrow((8.95, 2.9), (9.85, 2.9))
  arrow((7.95, 8.2), (10.85, 8.2), (10.85, 3.65))
  content((9.4, 8.45), dim[3 x 384 x 14 x 14])

  // Per-point token: features from all cameras + the pose + a step embedding.
  block(12.3, 2.2, 2.6, 1.4, [concat + linear\ #dim[1152 + 7 + 64 in]\ #dim[256 out]], fill: model)
  arrow((11.85, 2.9), (12.25, 2.9))
  content((10.85, 1.95), dim[32 x 1152])

  // Token sequence: 3 global (CLS) + 1 readout + 32 action points, 256 wide.
  let tx = 15.4
  grid-cells(tx, 4.95, 3, 8, cw: 0.2, ch: 0.12, fill: model)
  grid-cells(tx, 4.59, 1, 8, cw: 0.2, ch: 0.12, fill: human)
  grid-cells(tx, 4.40, 32, 8, cw: 0.2, ch: 0.09, fill: feat)
  arrow((14.95, 2.9), (15.35, 2.9))
  content((tx + 0.8, 5.25), small[36 tokens x 256])
  content((tx + 1.75, 4.77), dim[CLS x 3], anchor: "west")
  content((tx + 1.75, 4.53), dim[readout], anchor: "west")
  content((tx + 1.75, 2.5), dim[points x 32], anchor: "west")

  // Transformer.
  let (x0, x1) = (18.7, 21.7)
  rect((x0, 1.3), (x1, 4.9), stroke: line-w)
  content(((x0 + x1) / 2, 5.2), small[transformer, d = 256, 4 heads])
  for (i, y) in (4.05, 3.3, 2.55).enumerate() {
    block(x0 + 0.2, y, x1 - x0 - 0.4, 0.55, [self-attention + MLP], fill: attn)
  }
  content(((x0 + x1) / 2, 2.25), [$dots.v$])
  content(((x0 + x1) / 2, 1.7), align(center, dim[4 layers\ about 3.5 M trainable]))
  arrow((17.75, 2.9), (18.65, 2.9))

  // Readout: one number.
  rect((22.3, 2.75), (22.6, 3.05), fill: human, stroke: line-w)
  arrow((21.75, 2.9), (22.25, 2.9))
  content((22.45, 2.45), small[$V_theta (o, a) in RR$])

  // Two uses of V.
  arrow((22.65, 2.9), (23.1, 2.9), (23.1, 3.9), (23.45, 3.9))
  arrow((23.1, 2.9), (23.1, 1.5), (23.45, 1.5))
  grid-cells(23.5, 4.6, 16, 14, cw: 0.12, ch: 0.085, fill: model)
  content((24.35, 4.85), small[act: $nabla_a V in RR^(16 times 14)$])
  content((23.5, 2.95), dim[descend $V$ from noise], anchor: "west")
  block(23.5, 1.0, 2.2, 1.0, [score: $-V$\ ranks chunks], fill: model)

  // Training signal.
  block(17.6, -0.5, 8.1, 1.1, [$cal(L) = cal(L)_"FM" ("demos") + lambda [V(o, a_"human") - V(o, a_pi)]$], fill: human)
  arrow((20.2, 0.65), (20.2, 1.25), dashed: true)
})
