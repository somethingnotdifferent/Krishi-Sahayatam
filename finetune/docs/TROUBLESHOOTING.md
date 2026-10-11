# Troubleshooting

Errors are typed, carry an actionable hint, and map to distinct exit codes.
This page lists the failures you are most likely to hit.

---

## Quick triage

```bash
uv run lfm-finetune doctor              # environment
uv run lfm-finetune inspect-hardware    # machine
uv run lfm-finetune stages              # the ten stages and their commands
uv run lfm-finetune -v <command>        # full traceback
```

Exit codes: `2` config · `3` environment/hardware · `4` model/tokenizer ·
`5` dataset · `6` LoRA · `7` training · `8` checkpoint · `9` export ·
`10` evaluation.

---

## Environment and hardware

### `No CUDA device. Full GPU SFT cannot run here`

`torch.cuda.is_available()` is False. Either no GPU, or a CPU-only torch build.

```bash
uv run lfm-finetune doctor
```

CPU-only build on a CUDA host (this repo defaults to CPU wheels):

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch
uv sync --no-build-isolation
```

Meanwhile: `configs/smoke.yaml` (350M) runs the whole pipeline on CPU.

### `dtype 'bfloat16' is not supported`

Compute capability < 8.0 (pre-Ampere). Use `float16` or `float32`. Check:

```bash
uv run lfm-finetune inspect-hardware | grep -E "GPU|bf16|fp16"
```

### `Estimated peak training memory X GiB exceeds usable VRAM Y GiB`

Preflight failed **before** allocating. The log names the reduction path:

- enable `model.gradient_checkpointing: true`
- lower `training.per_device_train_batch_size`, raise
  `training.gradient_accumulation_steps` (effective batch is preserved)
- lower `tokenization.max_seq_length`
- switch `model.dtype` to `bfloat16`

### `CUDA out of memory` mid-run

Preflight estimates are order-of-magnitude. If it still OOMs:

- halve the per-device batch and double accumulation;
- confirm gradient checkpointing is actually on (`train.log`:
  `Gradient checkpointing enabled`);
- `training.dataloader_pin_memory: false` if host RAM is tight;
- `--smoke-test` to confirm the model itself fits before debugging the config.

---

## Model and tokenizer

### `does not support` (architecture)

The model is not LFM2/LFM2.5. Use a `LiquidAI/LFM2.5-*` checkpoint, or set
`model.allow_unsupported_architecture: true` **and** supply
`lora.target_modules` explicitly.

### `max_seq_length exceeds the checkpoint's max_position_embeddings`

`max_position_embeddings` is 128000 for LFM2.5 but it was *trained* on 32,768.
For domain SFT use 512–2048. A warning fires above 32768.

### `does not ship a chat template`

Use a checkpoint that includes `chat_template.jinja` (all LFM2.5 checkpoints do),
or point `tokenization.chat_template` at a `.jinja` file you have audited.

### Generation never stops

Two causes:

1. `tokenization.append_eos_token: false` — the model was never taught to emit
   the end-of-turn token. Set it to `true`.
2. Training formatting ≠ inference formatting. Verify:

```bash
uv run lfm-finetune inspect-template -c <config> --samples 3
```

The supervised span must be exactly the assistant text plus the terminator, and
the inference prompt must be a prefix of the training sequence.

### Outputs look like they have no terminator

They do. LFM2.5's end-of-turn token decodes to an empty string, and
`special_tokens_map.json` registers it with empty `content`. Check termination via
the stop reason, not by eye.

---

## LoRA

### `Target modules auto not found` / `matched no module`

`lora.target_modules` names modules the model does not have. Inspect the real
names:

```bash
uv run lfm-finetune inspect-model -c <config>
```

On LFM2.5 the names are `self_attn.out_proj` (**not** `o_proj`),
`feed_forward.w1/w2/w3` (**not** `gate_proj/up_proj/down_proj`), and
`conv.in_proj`/`conv.out_proj`. Default `target_modules: auto` avoids this
entirely.

### `Zero parameters are trainable`

`target_modules` matched nothing. Same fix as above.

### `X% of parameters are trainable, above the ceiling`

Not a LoRA run. Usually `lora.modules_to_save` contains `lm_head` or
`embed_tokens`. On LFM2.5 base checkpoints those are the **same tensor**
(`tie_embedding: true`); training either breaks the tie and rewrites the output
distribution. The config layer rejects `modules_to_save: [lm_head]` outright —
remove it, or first set `tie_embedding: false` in a local copy of the
checkpoint.

### `lora.r=64 is far above lora.alpha=8`

Effective update scale is `alpha / r` = 0.125; training will stall. Use
`alpha >= 2 * r`, or enable `lora.use_rslora`.

---

## Dataset

### `All N raw rows were rejected`

Read `data/validated/report.json → rejections[]`. Each has `reason`, `detail` and
an excerpt. Common:

| Reason | Fix |
|---|---|
| `unrecognised_format` | add `messages` or an instruction pair |
| `missing_turn` | the record needs a user turn **followed by** an assistant turn |
| `empty_assistant_response` | answers below `min_assistant_chars` (8; agriculture sets 80) |
| `multiple_system_prompts` | keep only the first system message |
| `placeholder_text` | remove TODO/placeholder scaffolding |
| `prompt_leakage` | the answer repeats the prompt |

### `Exact duplicate conversations span split boundaries`

Deduplication runs globally before splitting precisely to prevent this. It means
raw data bypassed `prepare` or the quality gates were disabled. Re-run `prepare`
with `dataset.quality.deduplicate: true`.

### `Split produced empty validation set`

Too few examples for the ratios. Lower `validation_ratio`, add data, or change
`dataset.splitting.seed`. For a smoke corpus use `train_ratio: 0.6,
validation_ratio: 0.2`.

### Validation looks implausibly good

Check `leakage.hard` and `leakage.soft` in `report.json`, and the
`evaluation_set.disjointness_vs_train` block in the comparison report. A
`disjoint: false` invalidates the comparison.

### Manifest warning: `license: unknown`

Training on data of unknown provenance is not acceptable. Add a `manifest.yaml`
next to the raw file with a real SPDX identifier or `internal-use-only`.

---

## Training

### `loss: NaN`

1. `learning_rate` too high — drop to 1e-4.
2. bf16 on pre-Ampere — use fp16 or fp32.
3. Check `grad_norm` in `metrics.jsonl`; a spike before NaN points at the LR or a
   pathological example.

### Validation loss flat from step 1

No learning signal. Check in order: are the loss masks right
(`inspect-template`)? Is the dataset actually the domain data? Is the LR sane?

### Validation loss rises while train loss falls

Overfitting. `training.early_stopping_patience`, fewer epochs, or lower
`lora.r`. With fewer than ~200 examples, memorisation is expected — fix the data,
not the hyperparameters.

### Supervised coverage is very low (5–20 %)

Expected and correct when the system prompt is long: a large MICO system prompt
dominates each example and is deliberately masked. The warning fires only above
**95 %**, which is the signature of masking not working.

### `CUDA requested but no device`

Set `model.device: cpu` and `model.dtype: float32` for CPU runs, or move to a CUDA
host.

---

## Checkpoints and resume

### `resume_from_checkpoint does not exist` / `is incomplete`

A resumable checkpoint needs `trainer_state.json`, `optimizer.pt`,
`scheduler.pt`, `rng_state.pth` and the adapter weights. Or use
`--resume auto` to take the newest `checkpoint-*` in the run directory.

Resume continues the **same** run directory. If you pass `--output`, it wins over
inference from the checkpoint path — check which directory you are writing to.

### `No adapter weight file in <dir>`

`adapter_model.safetensors` missing. Copy the whole adapter directory; a partial
copy cannot be reloaded.

### Adapter will not load

Compare `export_info.json → base_model_id/base_model_revision` with the base you
are loading. Mismatched id or revision is the usual cause.

---

## Evaluation

### `No 'test' split available`

`train_ratio + validation_ratio` reached 1.0. Leave room:

```yaml
train_ratio: 0.80
validation_ratio: 0.10   # test gets 0.10
```

### Comparison looks identical between base and tuned

Check `generation.strategy` in the report. It must say greedy/deterministic.
Sampling makes the two runs incomparable.

### `got multiple values for keyword argument 'pad_token_id'`

An internal invariant was violated: `pad_token_id` must come from one place only
(the generation loop). `ChatFormatter.generation_kwargs()` intentionally omits it.
If you hit this, you have re-added it in a custom generation path.

---

## Diagnostics

```bash
uv run lfm-finetune -v <command>        # DEBUG traceback
export LOG_DEBUG_LIBS=1                # raise third-party logger levels
export NO_COLOR=1                      # plain logs
export HF_HOME=/data/hf-cache          # relocate the multi-GB model cache
cat checkpoints/<run>/train.log         # full run log
cat checkpoints/<run>/metrics.jsonl     # streamed metrics
cat checkpoints/<run>/run_metadata.json # versions, seeds, hashes, LoRA audit
cat data/validated/report.json          # dataset counts, warnings, leakage
```

HF `huggingface_hub` warnings about unauthenticated requests are benign; set
`HF_TOKEN` only for gated repos.