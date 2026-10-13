# pi0.5

Physical Intelligence's π₀.₅: a PaliGemma 3B VLM (SigLIP vision encoder plus Gemma 2B language
decoder) with a Gemma 300M flow-matching action expert. It is fine-tuned from `lerobot/pi05_base`
with LoRA, through `lerobot-train`.

```bash
training/pi05/train.sh --mode queue --cron
```

## Config: openpi's LoRA layout

These settings follow openpi's pi0.5 LoRA configs (`paligemma_variant="gemma_2b_lora"`,
`action_expert_variant="gemma_300m_lora"`), as used in
[saifahmadgit/openpi_franka](https://github.com/saifahmadgit/openpi_franka):

| Part | Training | Params |
|---|---|---|
| SigLIP vision encoder + projector | fully trained | 415M |
| Gemma 2B language decoder | frozen, LoRA r=16 α=16 on q/k/v/o + gate/up/down | 20M |
| Gemma 300M action expert | frozen, LoRA r=32 α=32 on q/k/v/o + gate/up/down | 14M |
| action in/out + time MLP projections | fully trained | 2M |

That is 450M trainable params out of 4.1B. In LeRobot this is `--peft.*`: a regex for the LoRA
targets, `full_training_modules` for the fully trained parts, and `rank_pattern`/`alpha_pattern`
for the expert's higher rank.

| Setting | Value | Why |
|---|---|---|
| dataset | `blue_cup_to_container_46ep` | the same episodes as GR00T |
| chunk / executed | 16 / 16 | the same as GR00T and Saif's teleop configs |
| batch / steps | 32 / 60k | |
| LR | 2.5e-5 peak, 1k warmup, cosine to 2.5e-6 over 60k | openpi's schedule, stretched to the run length |
| dtype | bf16 + gradient checkpointing | checkpointing is required: LoRA sends gradients through every VLM layer |
| normalization | quantile (q01/q99) for state and action | pi0.5 default; the dataset has the stats |
| uploads | steps 20k, 40k, 60k | as `step-*` branches |

## Cost

Measured on an RTX 6000 Ada (48 GB), batch 32. Every variant tried while setting this up:

| Variant | Trainable | Speed | Peak memory | 60k steps |
|---|---|---|---|---|
| full fine-tune, grad checkpointing | 4.1B | 7.3 s/step | 44.7 GB | ~5 days |
| action expert only, grad checkpointing | 693M | 5.2 s/step | 18.4 GB | ~3.6 days |
| action expert only, no checkpointing | 693M | out of memory | | |
| **LoRA (this config)** | 450M | 8.4 s/step | 30.5 GB | ~5.8 days |

The VLM forward pass dominates the cost, so freezing parameters barely changes speed. Fewer steps
is the main way to make it faster.

## Gotchas

- **Gated tokenizer**: pi0.5 loads the `google/paligemma-3b-pt-224` tokenizer. Accept its license with
  the HF account that owns your token. Accepting the Gemma family is a separate gate that does
  not cover it. The config's `precheck` makes a queued run wait until access works.
- **`--policy.pretrained_path` loads weights only**: set chunk size and the like explicitly, as here.
- **LoRA checkpoints** hold `adapter_model.safetensors` (adapters plus the fully trained modules),
  about 1.8 GB. Loading one also loads `lerobot/pi05_base`.

## Differences from Saif's openpi_franka setup

Not adopted here, but worth trying:
- Saif starts from **`pi05_droid`** (pi0.5 already fine-tuned on DROID Franka data) instead of
  `pi05_base`. DROID is a Franka dataset, so it is less clearly useful for Baxter.
- Saif trains on **delta joint actions** for the 7 arm joints (gripper absolute):
  `use_delta_joint_actions=True`. The LeRobot equivalent is `--set args.policy.use_relative_actions=true`
  (`relative_exclude_joints` defaults to `["gripper"]`; check it matches the Baxter action names).
- Saif uses three cameras with 30k steps; our Baxter dataset has two.

## Runs

- `andnetdeboer/pi05_BlueCupToContainer` (started 2026-10-13, 60k steps), W&B project `rethoughtai-pi05`
