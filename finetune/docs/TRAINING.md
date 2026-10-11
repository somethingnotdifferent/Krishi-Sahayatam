# Training

Reference for the parts of this pipeline where LFM2.5 behaves unlike an ordinary
Transformer, and for hyperparameter selection.

---

## 1. Measured LFM2.5 chat-template facts

Everything below was measured against the real checkpoints with
transformers 5.3.0, not inferred from documentation. Reproduce with:

```bash
uv run lfm-finetune inspect-template -c configs/smoke.yaml --samples 2
```

### 1.1 Templates are not uniform across sizes

| Checkpoint | template len | ChatML markers | `{% generation %}` | `assistant_tokens_mask` |
|---|---|---|---|---|
| `LFM2.5-230M-Base` | 4621 | yes | **yes** | works (7 tokens, answer covered) |
| `LFM2.5-350M-Base` | 1296 | yes | **no** | **0 tokens masked** |
| `LFM2.5-1.2B-Base` | 1296 | yes | **no** | **0 tokens masked** |
| `LFM2.5-2.6B-Base` | 4294 | yes | yes | works |
| `LFM2.5-8B-A1B-Base` | 4294 | yes | yes | works |
| `LFM2.5-1.2B-Instruct` | 5487 | yes | yes | works |

**This is the single most important finding for this project.** The primary
target checkpoint, `LFM2.5-1.2B-Base`, does **not** ship `{% generation %}`
blocks, so

```python
tokenizer.apply_chat_template(msgs, tokenize=True, return_assistant_tokens_mask=True)
# -> assistant_masks with 0 set bits
```

silently returns an all-zero mask. Anyone using TRL's `assistant_only_loss=True`
on this checkpoint would train on **100 % of tokens** while believing the prompt
was masked. `tests/test_chat_template.py` pins this behaviour so a future
transformers upgrade that changes it will be caught.

### 1.2 Rendered form

Every LFM2.5 template wraps each turn in ChatML control tokens:

```text
<|startoftext|><|im_start|>system
SYS<|im_end|>
<|im_start|>user
USER<|im_end|>
<|im_start|>assistant
ASST<|im_end|>
```

with `<|im_start|>` = id 6 and `<|im_end|>` = id 7. For the 128 k-vocab
checkpoints (2.6B, 8B) those ids differ (124899 / 124900) — never hard-code them;
read `tokenizer.eos_token_id`.

The render is **prefix-consistent**: `apply_chat_template(messages[:n-1],
add_generation_prompt=True)` reproduces the beginning of
`apply_chat_template(messages, add_generation_prompt=False)`. This property is
what makes prefix-length masking safe, and the pipeline **asserts** it rather
than assuming it.

### 1.3 `add_special_tokens=False` is mandatory

The template emits `{{ bos_token }}` itself, and the tokenizer's own
post-processor prepends another BOS:

```python
tokenizer(rendered, add_special_tokens=False)[ "input_ids"][:3]  # [1, 6, 24131]
tokenizer(rendered)[ "input_ids"][:3]                            # [1, 1, 24131]  <- wrong
```

`ChatFormatter.encode` always passes `add_special_tokens=False`.

### 1.4 Termination

`eos_token_id` is the end-of-turn marker. `ChatFormatter` strips the literal
newline the template places after it, so the supervised target is exactly

```text
... <|im_start|>assistant\n ANSWER <|im_end|>
```

which is precisely what inference produces. Leaving the newline in would train
the model to emit the stop token and then keep going.

Gotcha: `special_tokens_map.json` registers ids 6 and 7 with an **empty**
`content` field, so reading the raw JSON makes the terminator look absent. The
loaded tokenizer resolves `eos_token` to the real marker, and the **id** is what
stops generation.

---

## 2. Loss masking

Because 1.2B-Base has no `{% generation %}` blocks, the span is derived from
prefix length:

```text
prompt_ids = template(messages[:-1], add_generation_prompt=True)
full_ids   = template(messages,     add_generation_prompt=False)
labels     = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

Guarantees, each covered by a test:

- every prompt token is `-100`;
- no assistant token is `-100`;
- the inference prompt is a byte-identical prefix of the training sequence;
- a non-prefix-consistent template **raises** instead of training on the wrong
  span;
- an example with zero supervised tokens **raises**.

HF shifts labels internally, so `labels[i] == input_ids[i]` is correct; masking
position `i` removes the prediction of token `i`.

Coverage looks low (4–20 %) on realistic data because a long MICO system prompt
dominates each example. That is expected and correct. A warning fires if coverage
exceeds 95 %, which is the signature of masking not working.

### Truncation

`keep_head_tail` guarantees the supervised span survives:

```text
reserve = max(1, min(answer_len, limit // 2))
head    = min(prompt_len, (limit - reserve) // 2)
tail    = limit - reserve - head
result  = ids[:head] + ids[-tail-reserve:]
```

An earlier head+tail implementation masked **every** token whenever the prompt
alone exceeded `max_seq_length` — such records trained on nothing while looking
normal in the dataset statistics. `tests/test_chat_template.py::
test_long_prompt_still_produces_supervision` is the regression test.

---

## 3. LoRA

### 3.1 Discovered targets

Never hard-code PEFT target names on this architecture:

```text
feed_forward.w1 / w2 / w3        SwiGLU (16 each on 1.2B)
self_attn.q_proj / k_proj /
        v_proj / out_proj        attention (6 each)
conv.in_proj / conv.out_proj      short conv (10 each)
```

`lora.target_modules: auto` derives these from `model.named_modules()` and
cross-checks against Liquid's reference presets. The check is exact:

```text
reference check     exact match with Liquid AI leap-finetune preset
excluded (tied)     ['lm_head']
```

Generic names would silently match nothing or the wrong layers — `o_proj`,
`gate_proj`, `up_proj`, `down_proj` do not exist on LFM2.5.

### 3.2 Weight tying

`config.tie_embedding: true` means `lm_head.weight is model.embed_tokens.weight`.
The loader verifies the tie at runtime, discovery excludes both, and the config
layer rejects `modules_to_save: [lm_head]` with an explanation. Adapting either
would desynchronise the output distribution.

### 3.3 Parameter audit

Training refuses to start unless:

- trainable parameters > `lora.min_trainable_params` (default 1024);
- trainable share <= `lora.max_trainable_pct` (default 25 %);
- LoRA A and B matrices are both non-empty;
- **no non-LoRA parameter is trainable.**

Measured on the validated run:

```text
total parameters      357,482,240
trainable parameters  2,998,272 (0.839%)
frozen parameters     354,483,968
  LoRA A matrices     1,212,416
  LoRA B matrices     1,785,856
```

---

## 4. Hyperparameters

All configurable, all commented in `configs/base.yaml`.

| Parameter | Default | Effect |
|---|---|---|
| `lora.r` | 16 | Adapter capacity and speed. Higher overfits a small dataset faster. |
| `lora.alpha` | 32 | Effective scale = `alpha / r`. `alpha << r` stalls training; config rejects it. |
| `lora.dropout` | 0.05 | Adapter regulariser. Liquid's preset uses 0.1. |
| `lora.use_rslora` | false | Scale = `alpha / sqrt(r)`. Only for high rank with low alpha. |
| `lora.use_dora` | false | ~30–50 % slower. Try after plain LoRA plateaus. |
| `lora.bias` | `none` | Correct for LFM2.5 (`conv_bias=false`). |
| `training.learning_rate` | 2e-4 | LoRA SFT range 1e-4…3e-4. |
| `training.weight_decay` | 0.0 | LoRA convention; decay mostly double-penalises. |
| `training.warmup_ratio` | 0.03 | Small; most LoRA runs are short. |
| `training.lr_scheduler_type` | cosine | |
| `training.max_grad_norm` | 1.0 | Clipping. |
| `tokenization.max_seq_length` | 2048 | Logits memory scales with `batch × seq × 65 536`. |
| `model.gradient_checkpointing` | true | ~30 % compute for a large activation saving. |
| `training.assistant_only_loss` | true | Suppress the prompt gradient. |
| `training.packing` | false | Packing erases per-example boundaries. |

### Presets

| Preset | batch × accum | epochs | max_steps | Use |
|---|---|---|---|---|
| `smoke` | 1×1 | 1 | 4 | CI / validation |
| `pilot` | 2×4 | 1 | 50 | "does it train and behave?" |
| `balanced` | 4×4 | 3 | — | default |
| `memory_saver` | 1×16 | 3 | — | 8 GB cards |
| `full` | 8×4 | 3 | — | 24 GB cards |

None is universally optimal. Presets may only set keys that exist on the
training section — a preset that leaks a key into another section is rejected at
load time with a precise message.

### Effective batch size

`per_device × grad_accum × world_size`, computed and logged before training. It
is divided by `world_size` in distributed runs so the effective batch — and
therefore the LR schedule — stays comparable between single- and multi-GPU runs.

### Dataset size

Liquid's guidance is 500–5000 task examples. Below that, `prepare` and `train`
warn explicitly. Fewer than ~200 examples with 3 epochs will memorise the set;
reduce epochs or collect data instead.

---

## 5. Pilot before full SFT

```bash
uv run lfm-finetune train -c configs/agriculture/lfm25_1.2b.yaml --pilot \
  -s dataset.max_train_examples=500
```

A pilot answers six questions: does it train, is memory stable, are the LoRA
modules correct, does validation loss behave, does the adapter reload, does
inference produce valid output. Only then run the full SFT.

---

## 6. Quantisation (QLoRA) — deliberately not the default

`bitsandbytes` is an **optional extra** (`uv sync --extra quant`) and is not used
by any shipped config. Reasons:

- it has no working CPU path, so it cannot be validated here;
- LFM2.5's hybrid conv blocks are not a well-trodden QLoRA path;
- it costs accuracy and complicates merging and edge export;
- the memory profiles above already fit commodity GPUs at bf16.

Introduce it only after confirming support for the exact checkpoint and GPU, and
only when bf16 genuinely does not fit. Never quantise during training for a
project whose goal is edge deployment — quantise once, at export time
(`docs/DEPLOYMENT.md`).

---

## 7. Validation status on this machine

| Path | Status |
|---|---|
| Full pipeline on synthetic data, CPU | **Validated end to end** |
| bf16 / flash-attn / multi-GPU | Not runnable here (no CUDA) — enforced by preflight guards |
| 1.2B / 2.6B configs | Parsed, validated, memory estimates checked; not trained here |
| Real agriculture data | Not available — **no domain capability claim is made** |

The synthetic fixture uses arithmetic and string operations with *computed*
answers, so a low loss is meaningful for the pipeline and means nothing about
agriculture.