// Multi-Task DiT + BIND: the trained DiT (dit.typ) unchanged, plus one added path. CLIP's patch
// features, which the DiT drops, are read at the pixels where each noisy waypoint projects and
// added to that waypoint's token. Everything outlined in orange is new; the rest is dit.typ.
#import "lib.typ": *

#set page(width: auto, height: auto, margin: 10pt, fill: white)
#set text(font: "Noto Sans", size: 8pt)

#let dit = json("assets/dit.json")
#let new = 1.4pt + rgb("#eb6834")
#let new-block(x, y, w, h, body, fill: model) = {
  block(x, y, w, h, body, fill: fill)
  rect((x, y), (x + w, y + h), stroke: new)
}

#cetz.canvas(length: 1cm, {
  // Cameras and CLIP, as in the DiT.
  for ((name, h, label), top) in (("third_eye", 1.65, [third eye]), ("left", 1.375, [left wrist]), ("right", 1.375, [right wrist])).zip((10.6, 8.7, 7.05)) {
    content((0, top), image("assets/" + name + ".jpg", width: 2.2cm), anchor: "north-west")
    rect((0, top - h), (2.2, top), stroke: line-w)
    content((0.05, top - h + 0.05), box(fill: white, inset: 1.5pt, small(label)), anchor: "south-west")
  }
  arrow((2.3, 8.0), (2.85, 8.0))
  line((2.9, 6.0), (4.8, 6.8), (4.8, 9.2), (2.9, 10.0), close: true, fill: robot, stroke: line-w)
  content((3.85, 8.4), align(center)[*CLIP*\ ViT-B/16\ #dim[85.8 M, as DiT]])

  // Unchanged path: CLS tokens into the conditioning vector and adaLN.
  arrow((4.85, 9.0), (5.2, 9.0), (5.2, 10.4), (14.9, 10.4))
  content((9.5, 10.65), dim[CLS 3 x 768, state, text, time: $c in RR^5944$ into adaLN, as DiT])

  // New path: keep the patch features.
  for (i, y) in (8.15, 6.85, 5.55).enumerate() {
    tensor(5.6, y, 0.8, 0.8, 0.8, label: none)
    rect((5.6, y), (6.4, y + 0.8), stroke: new)
  }
  content((6.25, 5.2), dim[768 x 14 x 14 per camera, kept])
  arrow((4.85, 7.4), (5.5, 7.4))

  // Noisy chunk, back in metres, projected into each camera.
  content((1.4, 4.65), small[noised $x_t$, 32 x 14])
  heat-table(0, 4.35, dit.xt, cw: 0.2, ch: 0.1, limits: (-2, 2))
  arrow((2.85, 2.75), (3.35, 2.75))
  new-block(3.4, 2.15, 2.2, 1.2, [unnormalise\ $p_(x y z)$ per arm], fill: robot)
  arrow((5.65, 2.75), (6.05, 2.75))
  new-block(6.1, 2.15, 2.2, 1.2, [project\ $u = pi(K [R | t] p)$], fill: robot)
  stacked(3, 0.1, 0.1, (dx, dy) => grid-cells(8.7 + dx, 3.6 + dy, 16, 4, cw: 0.16, ch: 0.11))
  content((9.2, 1.65), dim[3 x 32 x 2 pixels])
  arrow((8.35, 2.75), (8.65, 2.75))

  // Read the patch features there; every denoising step reads them again.
  new-block(10.3, 2.15, 1.9, 1.2, [bilinear\ sample])
  arrow((9.75, 2.75), (10.25, 2.75))
  arrow((6.75, 6.0), (11.25, 6.0), (11.25, 3.4))
  content((9.0, 6.25), dim[patch features])
  arrow((12.25, 2.75), (12.65, 2.75))
  content((12.45, 1.85), dim[32 x 2304])
  new-block(12.7, 2.15, 1.8, 1.2, [linear\ #dim[2304 to 512\ 1.2 M]])

  // Added to each waypoint's token, which the DiT already makes from x_t.
  block(12.7, 4.3, 1.8, 1.0, [linear\ #dim[14 to 512, as DiT]], fill: model)
  arrow((2.85, 4.8), (12.65, 4.8))
  content((15.15, 3.75), text(size: 12pt)[$plus.o$])
  arrow((14.55, 4.8), (15.15, 4.8), (15.15, 4.0))
  arrow((14.55, 2.75), (15.15, 2.75), (15.15, 3.5))
  rect((14.95, 3.55), (15.35, 3.95), stroke: new)
  arrow((15.4, 3.75), (15.9, 3.75))
  grid-cells(15.95, 4.7, 32, 6, cw: 0.15, ch: 0.06, fill: feat)
  content((16.4, 4.95), dim[32 x 512])

  // The DiT itself, unchanged.
  block(17.3, 2.7, 3.0, 2.1, [DiT blocks x 6\ #dim[adaLN reads $c$; RoPE self-attention + MLP]\ #dim[128.7 M, as DiT]], fill: attn)
  arrow((16.9, 3.75), (17.25, 3.75))
  arrow((14.9, 10.4), (18.8, 10.4), (18.8, 4.85))
  arrow((20.35, 3.75), (20.75, 3.75))
  content((21.5, 3.75), small[$hat(epsilon)$, 32 x 14])
  content((20.8, 3.2), dim[$cal(L) = norm(epsilon - hat(epsilon))^2$, as DiT], anchor: "west")

  // Why it matters.
  content((17.3, 1.55), anchor: "north-west", box(width: 6.5cm, text(size: 6.8pt)[
    *One change from the DiT:* each noisy waypoint also sees the image features where it would put
    the gripper. As denoising moves the waypoints, the features they read move with them.
    *Added:* 1.2 M parameters and one projection per step; needs camera calibration.
  ]))
})
