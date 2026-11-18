// The kernel corrector, drawn in the style of corrector_axb.typ (the ridge corrector) to show what
// changed: nothing is fitted; the nudged rows are stored as keys K (unit-length standardised z) and
// values V (their nudges), and a new z is answered by a compact-support tricube kernel on its cosine
// similarity to each key (ttt.correctors.kernel on feat/kernel-scope):
//   f(z) = sum_i phi(u_i) v_i / max(1, sum_i phi(u_i)),  u_i = (1 - s_i) / (1 - s_min),
//   phi(u) = (1 - u^3)^3 for u < 1, else 0.
// The values are illustrative; the inference row is computed from them with s_min = 0.8.
#import "lib.typ": *

#set page(width: auto, height: auto, margin: 14pt, fill: white)
#set text(font: "New Computer Modern", size: 8pt, fill: black)

#let cw = 0.62  // cell width
#let ch = 0.42  // cell height
#let nudge-c = rgb("#f6d2b0")
#let learned-c = luma(232)

// A row of cells; `fill(j)` colours cell j.
#let cells(x0, y0, vals, fill: j => white, size: 6.5pt) = {
  for (j, v) in vals.enumerate() {
    rect((x0 + j * cw, y0 - ch), (x0 + (j + 1) * cw, y0), fill: fill(j), stroke: 0.4pt + black)
    content((x0 + (j + 0.5) * cw, y0 - ch / 2), text(size: size, v))
  }
  rect((x0, y0 - ch), (x0 + vals.len() * cw, y0), stroke: 0.8pt + black)
}
#let mul(x, y) = circle((x, y), radius: 0.055, fill: black, stroke: none)

#cetz.canvas(length: 1cm, {
  let top = 4.0
  let xw = 6 * cw

  // Memory: the nudged rows only, each a key and its value. Nothing is solved for.
  let names = ($k_1$, $k_2$, $dots.v$, $k_M$)
  let vals = (
    ("1.2", "−0.4", "0", $dots.c$, "0"),
    ("0.3", "0.1", "0", $dots.c$, "0"),
    ($dots.v$, $dots.v$, $dots.v$, [], $dots.v$),
    ("0.8", "−0.2", "0", $dots.c$, "0"),
  )
  for (i, n) in names.enumerate() {
    let y = top - i * ch
    rect((0, y - ch), (xw, y), fill: white, stroke: 0.5pt + black)
    content((xw / 2, y - ch / 2), text(size: 8pt, n))
  }
  rect((0, top - 4 * ch), (xw, top), stroke: 0.8pt + black)
  content((xw / 2, top + 0.95), text(size: 12pt)[$K$])
  content((xw / 2, top - 4 * ch - 0.35), text(size: 7.5pt)[$M times d$])

  let vx = xw + 0.5
  for (i, r) in vals.enumerate() {
    cells(vx, top - i * ch, r, fill: j => if i != 2 { nudge-c } else { white })
  }
  rect((vx, top - 4 * ch), (vx + 5 * cw, top), stroke: 0.8pt + black)
  for (j, h) in ($Delta_x$, $Delta_y$, $Delta_z$, $Delta_"rot"$, $Delta_"grip"$).enumerate() {
    content((vx + (j + 0.5) * cw, top + 0.3), text(size: 7.5pt, h))
  }
  content((vx + 2.5 * cw, top + 0.95), text(size: 12pt)[$V$])
  content((vx + 2.5 * cw, top - 4 * ch - 0.35), text(size: 7.5pt)[$M times 7$])
  content((-0.35, top - 2 * ch), text(size: 7pt)[memory], anchor: "east")

  // Inference: similarity to every key, a weight that is exactly 0 beyond s_min, a weighted mean.
  let ry = top - 4 * ch - 0.8
  line((-1.6, ry), (vx + 5 * cw + 4.2, ry), stroke: 0.4pt + luma(170))
  content((-0.35, ry - 0.7), text(size: 7pt)[inference], anchor: "east")

  let y1 = ry - 0.7
  rect((0, y1 - ch / 2), (xw, y1 + ch / 2), stroke: 0.8pt + black)
  content((xw / 2, y1), text(size: 8pt)[$z$])
  mul(xw + 0.35, y1)
  content((xw + 0.6, y1), text(size: 11pt)[$K^T =$], anchor: "west")
  let sx = xw + 1.85
  cells(sx, y1 + ch / 2, ("0.97", "0.42", $dots.c$, "0.90"))
  content((sx + 2 * cw, y1 - 0.5), text(size: 7.5pt)[$s$, cosine similarity to each key])

  let y2 = y1 - 1.2
  content((sx - 0.15, y2), text(size: 11pt)[$phi(s) =$], anchor: "east")
  cells(sx, y2 + ch / 2, ("0.99", "0", $dots.c$, "0.67"), fill: j => if j in (0, 3) { learned-c } else { white })
  content((sx + 2 * cw, y2 - 0.5), text(size: 7.5pt)[$phi_i = 0$ when $s_i <= s_min$, else $(1 - u_i^3)^3$ with $u_i = (1 - s_i) slash (1 - s_min)$])

  let y3 = y2 - 1.35
  content((sx - 0.15, y3), text(size: 11pt)[$hat(y)_"nudge" = (phi V) / max(1, sum_i phi_i) =$], anchor: "east")
  cells(sx, y3 + ch / 2, ("1.0", "−0.3", "0", $dots.c$, "0"), fill: j => nudge-c)

  // Caption.
  content((-1.6, y3 - 0.8), box(width: (vx + 5 * cw + 5.8) * 1cm, text(size: 8pt)[*Figure 2.* Kernel corrector. Nothing is fitted: the nudged rows of the correction rollouts are stored as $M$ keys $K$ (each $z$, standardised and scaled to unit length) and values $V$ (the nudges). At inference the current $z$ is compared with every key; a key whose similarity is at most $s_min$ gets weight exactly 0 ($k_2$ here), so where no key is similar $hat(y)_"nudge" = 0$ and the base policy runs unchanged. Weights sum to at most 1 in the mean, so the offset never exceeds the largest stored nudge. Values are illustrative ($s_min = 0.8$).]), anchor: "north-west")
})
