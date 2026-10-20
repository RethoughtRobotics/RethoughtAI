// VITA as trained on SweepTrash (vita_SweepTrash_Kin-Full_inpainted_minimax_lead640ms_aug_30k_r1,
// step 30000), run on episode 0 at t = 20 s. Every latent and chunk drawn here is the checkpoint's
// own output, traced by assets/vita_trace.py into assets/vita.json.
#import "lib.typ": *

#set page(width: auto, height: auto, margin: 10pt, fill: white)
#set text(font: "Noto Sans", size: 8pt)

#let v = json("assets/vita.json")
#let lim = 0.1  // one colour scale for every latent; larger values saturate

#cetz.canvas(length: 1cm, {
  // Cameras: one observation step.
  let cams = (
    ("assets/third_eye.jpg", 1.65, [third eye]),
    ("assets/left.jpg", 1.375, [left wrist]),
    ("assets/right.jpg", 1.375, [right wrist]),
  )
  for ((path, h, name), top) in cams.zip((11.0, 9.1, 7.45)) {
    content((0, top), image(path, width: 2.2cm), anchor: "north-west")
    rect((0, top - h), (2.2, top), stroke: line-w)
    content((0.05, top - h + 0.05), box(fill: white, inset: 1.5pt, small(name)), anchor: "south-west")
  }
  arrow((2.3, 8.5), (2.95, 8.5))
  content((2.62, 8.95), align(center, dim[224 x\ 308]))

  // One ResNet-18 for all cameras.
  line((3.0, 6.4), (4.9, 7.2), (4.9, 9.8), (3.0, 10.6), close: true, fill: robot, stroke: line-w)
  content((3.95, 9.0), align(center)[*ResNet-18*\ #dim[11.2 M]])
  content((3.95, 7.75), align(center, dim[shared, trained;\ BN frozen]))
  arrow((4.95, 8.5), (5.4, 8.5))

  // Its 7 x 10 maps, averaged to one vector each: real values, in the observer's camera order.
  let rows = ((9.85, [left wrist]), (8.25, [right wrist]), (6.65, [third eye]))
  for (i, (y, name)) in rows.enumerate() {
    tensor(5.5, y, 0.75, 0.53, 0.9, label: [512 x 7 x 10])
    arrow((6.95, y + 0.4), (7.55, y + 0.4))
    strip(7.6, y + 0.29, v.img_feats.at(i), w: 4.0, limit: lim)
    content((7.6, y + 0.75), dim[#name: 512], anchor: "west")
  }
  content((7.25, 10.95), small[mean over 70 positions])
  strip(7.6, 5.75, v.state, w: 1.2, limit: calc.max(..v.state.map(calc.abs)))
  content((8.9, 5.86), dim[state 28], anchor: "west")

  // Concatenate and project to the observation latent.
  for y in (10.14, 8.54, 6.94, 5.86) { line((11.65, y), (12.0, y), stroke: line-w) }
  line((12.0, 10.14), (12.0, 5.86), stroke: line-w)
  content((12.0, 10.8), dim[concat 1564])
  arrow((12.0, 8.0), (12.35, 8.0))
  block(12.4, 7.4, 1.7, 1.2, [linear\ #dim[1564 to 512\ 0.8 M]], fill: model)

  // The flow: 6 Euler steps from the observation latent to an action latent, every value real.
  let (fx, fw, top, rh) = (15.4, 5.0, 10.0, 0.3)
  content((fx + fw / 2, 10.75), small[flow in latent space, 6 Euler steps, no noise])
  content((fx + fw / 2, 10.4), dim[$z_(k+1) = z_k + 1 / 6 f_theta (z_k, k / 6)$])
  for (k, z) in v.steps.enumerate() {
    strip(fx, top - (k + 1) * rh, z, w: fw, h: rh - 0.04, limit: lim)
    content((fx - 0.1, top - (k + 0.5) * rh), dim[$k = #k$], anchor: "east")
  }
  arrow((14.15, 8.0), (14.5, 8.0), (14.5, top - 0.5 * rh), (fx - 0.65, top - 0.5 * rh))
  content((fx + fw + 0.1, top - 0.5 * rh), dim[$z_0$: observation], anchor: "west")
  content((fx + fw + 0.1, top - 6.5 * rh), dim[$hat(z)_1$: action], anchor: "west")
  strip(fx, 7.25, v.z_demo, w: fw, h: rh - 0.04, limit: lim)
  content((fx - 0.1, 7.38), dim[demo], anchor: "east")
  content((fx + fw + 0.1, 7.38), dim[cos #calc.round(v.cos, digits: 3)], anchor: "west")
  content((fx + fw / 2, 7.05), dim[demo chunk through the action encoder])
  for (i, x) in (-1, 0, 1).enumerate() { strip(fx + 1.5 + i * 0.7, 6.55, (x * lim,) * 4, w: 0.7, h: 0.15, limit: lim) }
  content((fx + 1.5, 6.4), dim[$-0.1$], anchor: "north")
  content((fx + 2.55, 6.4), dim[0], anchor: "north")
  content((fx + 3.6, 6.4), dim[$+0.1$], anchor: "north")
  block(fx, 4.9, fw, 1.1, [flow net $f_theta$, applied at every step $k$: 4 MLP layers, 512 wide x 4, time-modulated\ #dim[12.6 M]], fill: model)

  // Decode the action latent to a 32-step chunk; the first 16 run.
  block(22.3, 7.75, 1.6, 1.2, [MLP decoder\ #dim[512 to 448\ 2.6 M]], fill: model)
  arrow((21.55, top - 6.5 * rh), (22.0, top - 6.5 * rh), (22.0, 8.35), (22.25, 8.35))
  arrow((23.95, 8.35), (24.25, 8.35))
  let ref = v.pred + v.demo.slice(0, 16)
  content((25.65, 10.65), small[VITA, steps 1 to 16])
  heat-table(24.3, 10.2, v.pred, cw: 0.2, ch: 0.16, ref: ref)
  content((28.85, 10.65), small[demo (lead 0.64 s)])
  heat-table(27.5, 10.2, v.demo.slice(0, 16), cw: 0.2, ch: 0.16, ref: ref)
  for x in (24.3, 27.5) {
    content((x + 0.7, 10.35), dim[left])
    content((x + 2.1, 10.35), dim[right])
  }
  content((27.1, 7.15), dim[left TCP: #calc.round(v.pos_err_mm, digits: 1) mm mean distance; one colour scale per column across both])

  // Training: the action autoencoder, drawn as channels x time.
  content((5.4, 3.95), small[action encoder (training only): 4 stride-2 convolutions, kernel 5])
  let convs = ((14, 32, [14 x 32]), (512, 16, [512 x 16]), (512, 8, [512 x 8]), (512, 4, [512 x 4]), (512, 2, [512 x 2]))
  let cx = 0.0
  for (i, (c, t, label)) in convs.enumerate() {
    let (w, h) = (t * 0.07, if c == 14 { 0.35 } else { 1.8 })
    rect((cx, 1.2), (cx + w, 1.2 + h), fill: if c == 14 { data } else { feat }, stroke: line-w)
    content((cx + w / 2, 0.85), dim(label))
    if i < convs.len() - 1 { arrow((cx + w + 0.1, 1.5), (cx + w + 0.6, 1.5)) }
    cx += w + 0.7
  }
  content((1.12, 1.85), dim[demo $a$])
  arrow((cx - 0.5, 2.1), (cx, 2.1))
  vector(cx + 0.05, 2.0, 8, 1024, cw: 0.13, ch: 0.2, fill: feat)
  content((cx + 0.57, 2.45), dim[flatten 1024])
  arrow((cx + 1.15, 2.1), (cx + 1.5, 2.1))
  block(cx + 1.55, 1.5, 1.6, 1.2, [linear\ #dim[1024 to 512\ 0.5 M]], fill: model)
  arrow((cx + 3.2, 2.1), (cx + 3.55, 2.1))
  content((cx + 3.6, 2.1), small[$z_"demo"$], anchor: "west")

  // The four loss terms.
  block(13.6, 0.9, 8.5, 2.1, align(left)[
    $cal(L) = underbrace(norm(f_theta (z_t, t) - (z_"demo" - z_0))^2, "flow, OT-paired") + 1.0 dot underbrace(norm(hat(z)_1 - z_"demo")^2, "consistency")$\
    #std.v(2pt)
    $quad + 0.5 dot underbrace(abs(D(hat(z)_1) - a), "decode flowed") + 0.5 dot underbrace(abs(D(z_"demo") - a), "autoencoder")$
  ], fill: human)

  // Where the 31.9 M parameters are.
  let params = (
    ([ResNet-18], 11.19, [BN frozen]),
    ([flow net], 12.60, [trained]),
    ([action encoder], 4.50, [training only]),
    ([decoder], 2.59, [trained]),
    ([observation linear], 0.80, [trained]),
  )
  content((24.1, 5.9), anchor: "north-west", text(size: 6.5pt, table(
    columns: 3,
    stroke: 0.4pt + ink,
    inset: 3pt,
    align: (left, right, left),
    [*module*], [*params*], [*training*],
    ..params.map(((m, n, t)) => (m, str(n) + " M", t)).flatten(),
    [*total*], [*31.9 M*], [],
  )))
})
