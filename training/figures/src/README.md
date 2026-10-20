# Figure sources

The figures in [`training_experiments.md`](../../training_experiments.md), written in
[Typst](https://typst.app) with [Fletcher](https://typst.app/universe/package/fletcher) (boxes and
arrows) and [CeTZ](https://typst.app/universe/package/cetz) (exact coordinates). Typst downloads both
packages on first build.

| File | Figure |
|---|---|
| `vita.typ` | VITA as trained on SweepTrash (step 30000); every latent and chunk is its own output on episode 0, t = 20 s |
| `dit.typ` | Multi-Task DiT as trained on SweepTrash (step 30000), from its config and weights, with a real noised chunk |
| `dit_bind.typ` | DiT + BIND: the DiT unchanged plus one path, patch features read where each noisy waypoint projects |
| `model.typ` | the energy potential V (BIND + Energy Matching) at real tensor sizes |
| `ttt.typ` | test-time training on VITA: frozen policy, LoRA as the correcting model, its scale as the blend; designs A and B |
| `ttt_lora.typ` | the same at tensor level: LoRA matrices to scale, A's latent pair, B's group and its ln 2 weights |
| `correction.typ` | 1: the correction loop, 2: one correction on the timeline, 3: the scoring model (overview) |
| `test_plans.typ` | one diagram per test plan (9 pages); the varied part outlined in orange |
| `lib.typ` | shared primitives: `tensor`, `vector`, `heat-table`, `strip`, `grid-cells`, `stacked`, `block`, `arrow` |

## Data

`assets/` holds what the figures read, made from the local SweepTrash dataset and checkpoints. From
`rethought_ai/`:

```bash
uv run python training/figures/src/assets/vita_trace.py    # vita.json: runs the VITA checkpoint (GPU)
uv run python training/figures/src/assets/make_assets.py   # frames, chunk.json, dit.json, ttt.json
```

## Build

Edit with a live preview: open this folder in VS Code (Tinymist extension), then `Ctrl+K V`.
Write the PNGs the document shows, from this folder:

```bash
for f in vita dit dit_bind model ttt ttt_lora; do typst compile $f.typ ../$f.png --ppi 200; done
typst compile correction.typ "../correction-{p}.png" --ppi 200
typst compile test_plans.typ "../test-plan-{p}.png" --ppi 180
```

`.pdf` or `.svg` instead of `.png` for papers and slides.
