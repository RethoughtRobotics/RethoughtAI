// The ridge nudge corrector as X W ≈ Y with example entries: one row per planning timestep t (every
// 0.16 s); X's columns are the parts of what the policy saw, z = [state, wrist camera, third-eye camera
// features] (the policy's encoding z and the robot state), and each part has matching rows in W; Y's
// columns are the person's nudge (ttt.correctors.data.problem, target 'mean'). The values are
// illustrative, not from the data. For the 10-22 PaperCup nudge rows n = 722 and d = 4,147.
#import "lib.typ": *

#set page(width: auto, height: auto, margin: 14pt, fill: white)
#set text(font: "New Computer Modern", size: 8pt, fill: black)

#let cw = 0.62  // cell width
#let ch = 0.42  // cell height
#let nudge-c = rgb("#f6d2b0")
#let learned-c = luma(232)  // what training solves for: W and b

// A matrix of cells; `fill(i, j)` colours cell (i, j).
#let grid(x0, y0, rows, fill: (i, j) => white, size: 6.5pt) = {
  for (i, r) in rows.enumerate() {
    let y = y0 - i * ch
    for (j, v) in r.enumerate() {
      rect((x0 + j * cw, y - ch), (x0 + (j + 1) * cw, y), fill: fill(i, j), stroke: 0.4pt + black)
      content((x0 + (j + 0.5) * cw, y - ch / 2), text(size: size, v))
    }
  }
  rect((x0, y0 - rows.len() * ch), (x0 + rows.at(0).len() * cw, y0), stroke: 0.6pt + black)
}
// Rules between z's parts: after columns 2 and 4 of X, after rows 2 and 4 of W.
#let vrules(x0, y0, y1) = for j in (2, 4) { line((x0 + j * cw, y0), (x0 + j * cw, y1), stroke: 0.6pt + black) }
#let part(j) = white
#let rowvec(x0, y0, w, body) = {
  let (yc, xc) = (y0 - ch / 2, x0 + w / 2)
  rect((x0, y0 - ch), (x0 + w, y0), fill: white, stroke: 0.25pt + luma(200))
  if body != none {
    line((x0 + 0.2, yc), (xc - 0.5, yc), stroke: 0.5pt + luma(90))
    line((xc + 0.5, yc), (x0 + w - 0.2, yc), stroke: 0.5pt + luma(90))
    content((xc, yc), text(size: 8pt, body))
  } else { content((xc, yc), text(size: 7pt)[⋮]) }
}
#let mul(x, y) = circle((x, y), radius: 0.055, fill: black, stroke: none)

#cetz.canvas(length: 1cm, {
  let top = 4.0
  // Rows: each episode's timesteps t_1 .. t_n, the episodes stacked (N rows in all).
  let xrows = (
    ("0.2", "…", "0.8", "…", "-1.1", "…"),
    ("0.3", "…", "0.7", "…", "-1.0", "…"),
    ("⋮", "", "⋮", "", "⋮", ""),
    ("0.3", "…", "0.1", "…", "0.9", "…"),
    ("0.5", "…", "0.6", "…", "-0.8", "…"),
    ("0.5", "…", "0.4", "…", "-0.2", "…"),
    ("⋮", "", "⋮", "", "⋮", ""),
    ("0.4", "…", "0.2", "…", "0.7", "…"),
  )
  let yrows = (
    ("0", "0", "0", $dots.c$, "0"),
    ("0", "0", "0", $dots.c$, "0"),
    ($dots.v$, $dots.v$, $dots.v$, "", $dots.v$),
    ("1.2", "−0.4", "0", $dots.c$, "0"),
    ("0", "0", "0", $dots.c$, "0"),
    ("0.3", "0.1", "0", $dots.c$, "0"),
    ($dots.v$, $dots.v$, $dots.v$, "", $dots.v$),
    ("0.8", "−0.2", "0", $dots.c$, "0"),
  )
  // Each episode drawn as its own block, a gap between them.
  let per = 4
  let gap = 0.55
  let nr = xrows.len()
  let rowy(i) = top - i * ch - if i >= per { gap } else { 0 }  // top edge of row i
  let bottom = rowy(nr - 1) - ch
  let mid = (top + bottom) / 2
  let nudged = (3, 5, 7)
  let blocks(x0, rows, fill) = {
    grid(x0, top, rows.slice(0, per), fill: fill)
    grid(x0, rowy(per), rows.slice(per), fill: (i, j) => fill(i + per, j))
    if rows.at(0).len() == 6 {
      vrules(x0, top, rowy(per - 1) - ch)
      vrules(x0, rowy(per), bottom)
    }
    content((x0 + rows.at(0).len() * cw / 2, top - per * ch - gap / 2), text(size: 7pt)[⋮])
  }


  // Episodes: the rows stack every correction episode's timesteps.
  let brace(i0, i1, body) = {
    let (ya, yb) = (rowy(i0) - 0.05, rowy(i1) - ch + 0.05)
    line((-0.85, ya), (-0.95, ya), (-0.95, yb), (-0.85, yb), stroke: 0.6pt + black)
    content((-1.05, (ya + yb) / 2), text(size: 7pt, body), anchor: "east")
  }
  brace(0, 3, [correction 1])
  brace(4, 7, [correction $m$])

  // X: columns grouped by the part of z they come from.
  // Row labels.
  for (i, n) in ([$t_1$], [$t_2$], [], [$t_n$], [$t_1$], [$t_2$], [], [$t_n$]).enumerate() {
    content((-0.15, rowy(i) - ch / 2), text(size: 8pt, n), anchor: "east")
  }
  let xw = 6 * cw
  let znames = ($z_1$, $z_2$, $dots.v$, $z_n$) * 2
  for (i, n) in znames.enumerate() {
    rect((0, rowy(i) - ch), (xw, rowy(i)), fill: white, stroke: 0.5pt + black)
    content((xw / 2, rowy(i) - ch / 2), text(size: 8pt, n))
  }
  rect((0, rowy(per - 1) - ch), (xw, top), stroke: 0.8pt + black)
  rect((0, bottom), (xw, rowy(per)), stroke: 0.8pt + black)
  content((xw / 2, top - per * ch - gap / 2), text(size: 7pt)[⋮])
  content((xw / 2, top + 0.95), text(size: 12pt)[$X_#text(size: 8pt)[$"obs"$]$])
  content((xw / 2, bottom - 0.35), text(size: 7.5pt)[$N times d$])

  // W: its rows match X's columns, part by part.
  let wcw = 0.85  // W's column width, room for w_grip
  let wx = xw + 0.8
  mul(xw + 0.4, mid)
  // W's rows pair with X's columns, its columns with Y's: column j holds the weights for output j.
  let wtop = mid + 3.5 * ch
  // One weight vector per output, drawn as whole columns like X's rows.
  for (j, n) in ($w_1$, $dots.c$, $w_7$).enumerate() {
    rect((wx + j * wcw, wtop - 7 * ch), (wx + (j + 1) * wcw, wtop), fill: learned-c, stroke: 0.5pt + black)
    content((wx + (j + 0.5) * wcw, wtop - 3.5 * ch), text(size: 8pt, n))
  }
  rect((wx, wtop - 7 * ch), (wx + 3 * wcw, wtop), stroke: 0.8pt + black)
  content((wx + 1.5 * wcw, top + 0.95), text(size: 12pt)[$W$])
  content((wx + 1.5 * wcw, wtop - 7 * ch - 0.35), text(size: 7.5pt)[$d times 7$])

  // b: a 7-vector, drawn as a column; its transpose is added to every row of X_obs W.
  let bx = wx + 3 * wcw + 0.85
  content((bx - 0.42, mid), text(size: 12pt)[$+$])
  let btop = mid + 3.5 * ch
  rect((bx, btop - 7 * ch), (bx + cw, btop), fill: learned-c, stroke: 0.8pt + black)
  content((bx + cw / 2, btop - 3.5 * ch), text(size: 8pt)[$b$])
  content((bx + cw / 2, top + 0.95), text(size: 12pt)[$b$])
  content((bx + cw / 2, btop - 7 * ch - 0.35), text(size: 7.5pt)[$7 times 1$])

  // Y.
  let yx = bx + cw + 0.85
  content((yx - 0.42, mid), text(size: 12pt)[$approx$])
  blocks(yx, yrows, (i, j) => if i in nudged { nudge-c } else { white })
  for (j, h) in ($Delta_x$, $Delta_y$, $Delta_z$, $Delta_"rot"$, $Delta_"grip"$).enumerate() {
    content((yx + (j + 0.5) * cw, top + 0.3), text(size: 7.5pt, h))
  }
  content((yx + 2.5 * cw, top + 0.95), text(size: 12pt)[$Y_#text(size: 8pt)[$"nudge"$]$])
  content((yx + 2.5 * cw, bottom - 0.35), text(size: 7.5pt)[$N times 7$])

  // Inference, set apart from the training data by a rule.
  line((-2.2, -0.8), (yx + 5 * cw + 0.3, -0.8), stroke: 0.4pt + luma(170))
  let sy = -1.5
  content((-1.05, sy), text(size: 7pt)[inference], anchor: "east")
  rect((1.5, sy - ch / 2), (1.5 + xw, sy + ch / 2), stroke: 0.8pt + black)
  content((1.5 + xw / 2, sy), text(size: 8pt)[$z$])
  mul(1.5 + xw + 0.35, sy)
  content((1.5 + xw + 0.65, sy), text(size: 11pt)[$W + b^T =$], anchor: "west")
  grid(1.5 + xw + 2.55, sy + ch / 2, (("1.0", "−0.3", "0", $dots.c$, "0"),), fill: (i, j) => nudge-c)
  content((1.5 + xw + 2.55 + 2.5 * cw, sy - 0.45), text(size: 8pt)[$hat(y)_"nudge"$])

  // Caption, paper style: how to read the figure.
  content((-2.2, sy - 1.0), box(width: (yx + 5 * cw + 2.5) * 1cm, text(size: 8pt)[*Figure 1.* Ridge corrector. Each row of $X_"obs"$ is the policy's observation vector $z$ (with the robot state, $d$ numbers) at one timestep of a correction rollout; the matching row of $Y_"nudge"$ is the person's nudge (orange where nonzero). Training solves $X_"obs" W + bold(1) b^T approx Y_"nudge"$ in closed form for the parameters in gray, with $b in RR^7$ the weighted mean nudge added to every row; at inference the current $z$ gives the offset $hat(y)_"nudge" = z W + b^T$. Values are illustrative.]), anchor: "north-west")
})
