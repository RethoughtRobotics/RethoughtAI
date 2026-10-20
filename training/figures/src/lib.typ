// Shared drawing primitives for paper figures: tensors, vectors, tables, layer stacks.
// Every shape has square corners; sizes are in canvas units (1 = 1 cm).
#import "@preview/cetz:0.3.4"
#import cetz.draw: *

#let ink = rgb("#333333")
#let robot = rgb("#e8e8e8")
#let human = rgb("#f6dcc4")
#let data = rgb("#fbf6e3")
#let model = rgb("#d9e4f2")
#let attn = rgb("#f7e08a")
#let feat = rgb("#b9d7a8")
#let line-w = 0.6pt + ink
#let arrow-mark = (end: "stealth", fill: ink, scale: 0.6)

// Small label text.
#let small(body) = text(size: 6.5pt, body)
#let dim(body) = text(size: 6.5pt, fill: rgb("#555555"), body)

// An arrow through any number of points.
#let arrow(..pts, dashed: false) = line(
  ..pts,
  stroke: if dashed { (paint: ink, thickness: 0.6pt, dash: "dashed") } else { line-w },
  mark: arrow-mark,
)

// A labelled box. `(x, y)` is the bottom-left corner.
#let block(x, y, w, h, body, fill: white) = {
  rect((x, y), (x + w, y + h), fill: fill, stroke: line-w)
  content((x + w / 2, y + h / 2), box(width: (w - 0.2) * 1cm, align(center, text(size: 6.8pt, body))))
}

// A C x H x W tensor drawn as a cuboid: the front face is H x W, the depth is the channels,
// with `slices` lines on the top face to read as stacked feature maps.
#let tensor(x, y, w, h, d, fill: feat, slices: 6, label: none) = {
  let (dx, dy) = (d * 0.7, d * 0.5)
  line((x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h), close: true,
    fill: fill.lighten(30%), stroke: line-w)
  line((x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h), close: true,
    fill: fill.darken(15%), stroke: line-w)
  for i in range(1, slices) {
    let f = i / slices
    line((x + dx * f, y + h + dy * f), (x + w + dx * f, y + h + dy * f), stroke: 0.3pt + ink)
  }
  rect((x, y), (x + w, y + h), fill: fill, stroke: line-w)
  if label != none { content((x + (w + dx) / 2, y - 0.25), dim(label)) }
}

// A vector of length n drawn as a row of `cells` cells (a long vector is drawn with fewer
// cells than entries; the label carries the true length).
#let vector(x, y, cells, n, cw: 0.16, ch: 0.22, fill: data, label: none, vertical: false) = {
  for i in range(cells) {
    let (a, b) = if vertical { ((x, y + i * cw), (x + ch, y + (i + 1) * cw)) } else {
      ((x + i * cw, y), (x + (i + 1) * cw, y + ch))
    }
    rect(a, b, fill: fill, stroke: 0.4pt + ink)
  }
  if label != none {
    if vertical { content((x + ch / 2, y - 0.22), dim(label)) } else {
      content((x + cells * cw + 0.1, y + ch / 2), dim(label), anchor: "west")
    }
  }
}

// A heat-map table: `rows` is a list of lists of numbers, coloured per column between
// that column's min and max (with a floor on the range so near-constant columns stay pale),
// or between fixed `limits: (lo, hi)` for values already normalised. `ref`: other rows whose
// column ranges set the colours, so two tables can share one scale.
#let heat-table(x, y, rows, cw: 0.3, ch: 0.2, header: none, floor: 0.02, limits: none, ref: none) = {
  let ncol = rows.at(0).len()
  let cols = range(ncol).map(j => (if ref != none { ref } else { rows }).map(r => r.at(j)))
  let grad = gradient.linear(rgb("#3b6ea8"), white, rgb("#d9822b"))
  for (i, r) in rows.enumerate() {
    for (j, v) in r.enumerate() {
      let (lo, hi) = if limits != none { limits } else { (calc.min(..cols.at(j)), calc.max(..cols.at(j))) }
      let mid = (lo + hi) / 2
      let half = calc.max((hi - lo) / 2, floor)
      let t = calc.clamp((v - mid) / half / 2 + 0.5, 0, 1)
      let top = y - i * ch
      rect((x + j * cw, top - ch), (x + (j + 1) * cw, top), fill: grad.sample(t * 100%), stroke: 0.25pt + ink)
    }
  }
  if header != none {
    for (j, h) in header.enumerate() {
      content((x + (j + 0.5) * cw, y + 0.15), text(size: 4.5pt, h))
    }
  }
}

// A plain grid of cells, `n` rows by `m` columns, for shapes without data.
#let grid-cells(x, y, n, m, cw: 0.16, ch: 0.12, fill: data) = {
  for i in range(n) {
    for j in range(m) {
      rect((x + j * cw, y - (i + 1) * ch), (x + (j + 1) * cw, y - i * ch), fill: fill, stroke: 0.25pt + ink)
    }
  }
}

// `k` copies of a shape drawn back to front with an offset, for stacks of per-camera tensors.
#let stacked(k, dx, dy, draw-one) = {
  for i in range(k - 1, -1, step: -1) { draw-one(i * dx, i * dy) }
}

// A long vector of real values as a thin heat strip, one unbordered cell per entry, coloured
// symmetrically about zero by `limit` (default: the largest magnitude in `values`).
#let strip(x, y, values, w: 6, h: 0.22, limit: none) = {
  let lim = if limit != none { limit } else { calc.max(..values.map(calc.abs)) }
  let grad = gradient.linear(rgb("#3b6ea8"), white, rgb("#d9822b"))
  let cw = w / values.len()
  for (i, v) in values.enumerate() {
    let t = calc.clamp(v / lim / 2 + 0.5, 0, 1)
    rect((x + i * cw, y), (x + (i + 1) * cw, y + h), fill: grad.sample(t * 100%), stroke: none)
  }
  rect((x, y), (x + w, y + h), stroke: 0.4pt + ink)
}
