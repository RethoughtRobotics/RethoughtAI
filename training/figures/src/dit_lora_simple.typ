// Where the correction LoRA sits in the cup Multi-Task DiT. Left, faded: the conditioning path (inputs,
// encoders, z). Middle: the diffusion transformer, one block of four shown, with the four linear layers
// LoRA adapts (attn.qkv_proj, attn.out_proj, mlp.0, mlp.2; rethought-lerobot --lora_rank through Hugging
// Face PEFT). Right: one adapted layer drawn as in Hu et al. (2022), Fig. 1, with the adapter weight alpha: h = W x + alpha B A x, W frozen,
// A (r x d) and B (d x r) trained, r = 16. Noisy action tokens enter below, denoised ones leave above.
#import "lib.typ": *

#set page(width: auto, height: auto, margin: 14pt, fill: white)
#set text(font: "New Computer Modern", size: 10pt, fill: black)

#let lora-c = rgb("#e8833a")
#let lora-fill = rgb("#fbe3cf")
#let dit-c = rgb("#e8eff8")
#let faint = luma(150)
#let faint-fill = luma(245)
#let stroke-w = 0.6pt + black
#let fbox(x0, y0, x1, y1, body) = {
  rect((x0, y0), (x1, y1), fill: faint-fill, stroke: 0.5pt + faint)
  content(((x0 + x1) / 2, (y0 + y1) / 2), align(center, text(size: 8.5pt, fill: luma(110), body)))
}
#let farr(..pts) = line(..pts, stroke: 0.5pt + faint, mark: (end: "stealth", fill: faint, scale: 0.5))
#let arr(..pts) = line(..pts, stroke: stroke-w, mark: (end: "stealth", fill: black, scale: 0.5))
#let token(x, y, body) = {
  rect((x, y), (x + 0.55, y + 0.75), fill: luma(232), stroke: 0.5pt + black)
  content((x + 0.275, y + 0.375), text(size: 8.5pt, body))
}
// A linear layer of the block, with its LoRA beside it.
#let linear(y, body) = {
  rect((7.1, y), (9.4, y + 0.6), fill: white, stroke: stroke-w)
  content((8.25, y + 0.3), body)
  rect((9.4, y), (10.6, y + 0.6), fill: lora-fill, stroke: (paint: lora-c, thickness: 0.6pt))
  content((10.0, y + 0.3), text(size: 8.5pt)[$+ B A$])
}

#cetz.canvas(length: 1cm, {
  // Conditioning, faded and small: context only.
  for (i, name) in ([cameras], [instruction], [robot state], [step $k$]).enumerate() {
    let y = 4.8 - i * 0.62
    rect((3.4, y - 0.5), (5.0, y), fill: faint-fill, stroke: 0.5pt + faint)
    content((4.2, y - 0.25), text(size: 7.5pt, fill: luma(110), name))
    farr((5.05, y - 0.25), (5.45, y - 0.25))
  }
  rect((5.5, 2.4), (5.75, 4.8), fill: luma(228), stroke: 0.5pt + faint)
  content((5.625, 5.05), text(size: 9pt, fill: luma(110))[$z$])
  farr((5.8, 3.6), (6.05, 3.6))

  // The diffusion transformer: one block of four, its adapted linear layers.
  let (dx0, dx1, dy0, dy1) = (6.1, 11.2, 0.4, 7.2)
  rect((dx0, dy0), (dx1, dy1), fill: dit-c, stroke: 0.9pt + black)
  content(((dx0 + dx1) / 2, dy1 - 0.35), text(size: 12pt)[Diffusion transformer])
  rect((6.5, 0.8), (10.8, 6.5), fill: white, stroke: (paint: black, thickness: 0.5pt, dash: "dashed"))
  content((10.7, 6.25), text(size: 8.5pt)[block × 4], anchor: "east")
  // Each pair of layers grouped, its name along the side.
  for (y0, y1, name) in ((0.95, 3.1, [self-attention]), (3.6, 5.75, [MLP])) {
    rect((6.65, y0), (10.7, y1), fill: none, stroke: 0.4pt + luma(170))
    content((6.88, (y0 + y1) / 2), angle: 90deg, text(size: 8pt, style: "italic", name))
  }
  linear(1.1, [$W_"qkv"$])
  linear(2.35, [$W_"out"$])
  linear(3.75, [$W_1$])
  linear(5.0, [$W_2$])
  for (a, b) in ((1.7, 2.35), (2.95, 3.75), (4.35, 5.0)) { arr((8.25, a), (8.25, b - 0.02)) }

  // Noisy action tokens in, denoised out.
  let xs = (6.6, 7.6, 8.6, 10.3)
  for (x, n) in xs.zip(($a_1^k$, $a_2^k$, $a_3^k$, $a_32^k$)) {
    token(x, -0.75, n)
    arr((x + 0.275, 0.02), (x + 0.275, dy0 - 0.02))
  }
  content((9.65, -0.37), [$dots$])
  content(((dx0 + dx1) / 2, -1.1), text(size: 9pt)[noisy actions, step $k$])
  for (x, n) in xs.zip(($a_1^0$, $a_2^0$, $a_3^0$, $a_32^0$)) {
    arr((x + 0.275, dy1 + 0.02), (x + 0.275, dy1 + 0.45))
    token(x, dy1 + 0.47, n)
  }
  content((9.65, dy1 + 0.85), [$dots$])
  content(((dx0 + dx1) / 2, dy1 + 1.55), text(size: 9pt)[action chunk, after 10 steps])

  // One adapted layer, as in Hu et al. (2022), Fig. 1.
  let lx = 12.6
  line((10.6, 1.7), (lx, 1.6), stroke: (paint: luma(140), thickness: 0.4pt, dash: "dashed"))
  line((10.6, 1.1), (lx, 1.25), stroke: (paint: luma(140), thickness: 0.4pt, dash: "dashed"))
  content((lx + 2.0, 6.5), text(size: 10pt)[one adapted layer])
  // x in, h out.
  content((lx + 2.0, 0.55), [$x$, the layer's input])
  line((lx + 2.0, 0.8), (lx + 2.0, 1.1), stroke: stroke-w)
  line((lx + 0.9, 1.1), (lx + 3.2, 1.1), stroke: stroke-w)
  arr((lx + 0.9, 1.1), (lx + 0.9, 1.55))
  arr((lx + 3.2, 1.1), (lx + 3.2, 1.55))
  // W, frozen.
  rect((lx, 1.6), (lx + 1.8, 4.4), fill: luma(235), stroke: stroke-w)
  content((lx + 0.9, 3.0), text(size: 12pt)[$W$])
  content((lx + 0.9, 2.3), text(size: 8pt)[frozen])
  // A down to rank r, B back up, trained.
  let trap(y0, y1, wide-bottom, body) = {
    let (a, b) = if wide-bottom { (0.8, 0.25) } else { (0.25, 0.8) }
    line((lx + 3.2 - a, y0), (lx + 3.2 + a, y0), (lx + 3.2 + b, y1), (lx + 3.2 - b, y1), close: true,
         fill: lora-fill, stroke: (paint: lora-c, thickness: 0.7pt))
    content((lx + 3.2, (y0 + y1) / 2), body)
  }
  trap(1.6, 2.8, true, [$A$])
  trap(3.2, 4.4, false, [$B$])
  line((lx + 3.2, 2.8), (lx + 3.2, 3.2), stroke: stroke-w)
  content((lx + 3.65, 3.0), text(size: 8pt)[$r = 16$], anchor: "west")
  // Sum.
  arr((lx + 0.9, 4.45), (lx + 0.9, 5.0), (lx + 1.78, 5.0))
  arr((lx + 3.2, 4.45), (lx + 3.2, 5.0), (lx + 2.22, 5.0))
  circle((lx + 2.0, 5.0), radius: 0.2, fill: white, stroke: stroke-w)
  content((lx + 2.0, 5.0), text(size: 9pt)[$+$])
  arr((lx + 2.0, 5.22), (lx + 2.0, 5.7))
  content((lx + 2.0, 5.95), [$h = W x + alpha thin B A x$])
})
