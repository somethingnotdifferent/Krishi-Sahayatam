# Deployment

The goal is MICO: an agricultural assistant that runs on edge hardware. This
document covers what to ship, what to convert, and what is verified.

---

## 1. The artifact hierarchy

Ship the adapter. Merge only when a runtime cannot load one.

```text
HF base model (LiquidAI/LFM2.5-1.2B-Base)
  +
LoRA adapter (r=16, ~12 MiB)          <-- canonical MICO artifact
  |
  ├─ base + adapter        transformers / vLLM / TGI     (no conversion)
  └─ merged model          standalone safetensors        (convenience)
        |
        ├─ GGUF             llama.cpp, edge CPU
        └─ ONNX             ONNX Runtime, mobile/edge accelerators
```

Why the adapter first:

- **16 MiB vs 2.4 GiB.** The adapter is ~0.2 % of the merged size.
- **It pins the base checkpoint.** `export_info.json` records the base id and
  revision, so a delivered adapter is unambiguous.
- **It survives re-training and re-merging.** You can improve the adapter without
  re-downloading or re-merging 2.4 GiB.
- **It is verifiable.** Shards load, tensor shapes match the declared rank,
  target modules exist. Cheap and meaningful.

Merging is a convenience for runtimes that cannot compose an adapter, not a
better artifact.

---

## 2. Export the adapter

```bash
uv run lfm-finetune export-adapter \
  -c configs/agriculture/lfm25_1.2b.yaml \
  --run-dir checkpoints/agri_lfm25_1.2b_20260101-120000
```

```text
adapters/agri_lfm25_1.2b/
  adapter_config.json      r, alpha, target_modules, base_model_name_or_path
  adapter_model.safetensors
  tokenizer.json           included: inference needs no network access
  tokenizer_config.json
  chat_template.jinja      included: training/inference formats stay identical
  export_info.json         base id + revision, package versions, reload command
```

`export_info.json` example:

```json
{
  "base_model_id": "LiquidAI/LFM2.5-350M-Base",
  "base_model_revision": "main",
  "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "bias": "none",
           "target_modules": "auto", "exclude_modules": ["lm_head", "embed_tokens"]},
  "weights_files": ["adapter_model.safetensors"],
  "packages": {"torch": "2.11.0", "transformers": "5.3.0", "peft": "0.21.2"},
  "reload_command": "PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained('...'), '...')"
}
```

Shipping the tokenizer and template with the adapter is deliberate: production
must format prompts character-for-character as training did.

---

## 3. Merge (optional)

```bash
uv run lfm-finetune merge-adapter \
  -c configs/agriculture/lfm25_1.2b.yaml --run-dir <run>
```

Validated on `LFM2.5-350M-Base`:

```text
merge_dtype              bfloat16
weight_tying_preserved   True
reload verified          True
size                     680.7 MiB
```

### Merge safety on LFM2.5

`merge_and_unload` performs `W += B @ A` on the targeted Linears. Every module in
Liquid's reference set — including `conv.in_proj`/`conv.out_proj` inside
`Lfm2ShortConv` — is a plain `nn.Linear`, so the arithmetic is well-defined.
Merging **is** supported.

Two checks the pipeline performs rather than assuming:

1. **Weight tying.** `tie_embedding: true` means `lm_head.weight is
   embed_tokens.weight`. Because discovery never targets either, the tie is
   intact; the merge verifies it explicitly and aborts if tying changed.
2. **Convolution blocks.** No conv-specific handling is needed, but they *are*
   adapted when `conv.in_proj/out_proj` are targeted, and the merged output was
   verified to generate identically to base+adapter under greedy decoding.

Merge in **bfloat16** by default: it keeps the merged weights bit-consistent with
how they were trained and halves export size. Use `--float32` only when a
downstream runtime lacks bf16 support.

---

## 4. Edge runtimes for LFM2.5

Verify support for your exact checkpoint before planning around it. Liquid
publishes LFM2.5 inference targets; the official docs
(<https://docs.liquid.ai/lfm/inference/>) are authoritative.

| Runtime | Status for LFM2.5 | Artifact | Notes |
|---|---|---|---|
| **transformers** | supported | base + adapter | reference implementation; used by this project |
| **vLLM** | supported | base + adapter, or merged | throughput / GPU serving |
| **llama.cpp** | supported via GGUF | merged → GGUF | cross-platform CPU inference |
| **ONNX Runtime** | supported via ONNX | merged → ONNX | hardware-accelerated edge |
| MLX / CoreML | check docs | merged | Apple silicon |

Liquid publishes official GGUF and ONNX conversions of the base checkpoints.
**Do not assume a GGUF conversion of a merged LoRA model exists** — a merge plus
a quantise step is your own artifact and needs its own accuracy check.

### Conversions that are *not* supported or not recommended

- **Quantising during training.** Do not. QLoRA costs accuracy, complicates
  merging, and is not needed at 1.2B. Quantise once, at export.
- **Training in 4-bit** without validating the exact checkpoint + GPU combo. The
  conv blocks are an unverified path.
- **Converting a base model with an adapter applied "on the fly"** into GGUF. If a
  runtime needs GGUF, merge first, convert second, and measure.

---

## 5. Recommended MICO path

```bash
# 1. export the adapter (canonical)
uv run lfm-finetune export-adapter -c configs/agriculture/lfm25_1.2b.yaml --run-dir <run>

# 2. merge (only if the target runtime needs standalone weights)
uv run lfm-finetune merge-adapter  -c configs/agriculture/lfm25_1.2b.yaml --run-dir <run>

# 3. behaviour check before shipping
uv run lfm-finetune inference -c configs/agriculture/lfm25_1.2b.yaml \
  --merged merged/agri_lfm25_1.2b --json -p "..."
# compare against the adapter path; they must match under greedy decoding
```

Then, per runtime:

```bash
# GGUF: merge -> convert -> quantise -> measure
llama-quantize merged/agri_lfm25_1.2b/model-00001-of-00002.safetensors \
                mico-agri-q4_k_m.gguf Q4_K_M

# ONNX: merge -> convert -> measure, with a real MICO prompt set
```

**Always measure after quantisation.** Report the quantisation method and the
perplexity or task-metric delta against the unquantised merged model. A
quantised agricultural assistant that stops asking clarifying questions is worse
than an unquantised one, and nothing in the conversion will warn you.

---

## 6. Sizing

| Model | bf16 merged | Q8 | Q4_K_M | Fits |
|---|---|---|---|---|
| 350M | 0.68 GiB | ~0.36 | ~0.22 | low-end edge, MCU-class gateways |
| 1.2B | 2.4 GiB | ~1.2 | ~0.75 | mid-range on-device |
| 2.6B | 4.9 GiB | ~2.5 | ~1.6 | edge gateway |

Plus KV cache for the context window. At 2048 tokens, budget roughly
`layers × kv_heads × head_dim × 2 × seq × bytes`. For 1.2B at bf16 that is
about 0.25 GiB — plan for it explicitly, since it is often the memory that
breaks an edge deployment after the weights fit comfortably.

---

## 7. Deployment checklist

- [ ] `export_info.json` pins base model id **and** revision
- [ ] tokenizer + `chat_template.jinja` shipped with the artifact
- [ ] inference uses the checkpoint's own chat template, `add_special_tokens=False`
- [ ] generation stops on `eos_token_id` (7 for 65 k-vocab LFM2.5; do not hard-code)
- [ ] adapter path and merged path produce identical greedy output
- [ ] post-quantisation accuracy measured on a real MICO prompt set
- [ ] system prompt identical to training
- [ ] extension-officer escalation path present in responses
- [ ] run `run_metadata.json` archived alongside the artifact
- [ ] `comparison.md` archived: the quality case for the shipped model

---

## 8. What is verified vs assumed

**Verified here:** adapter export and verification; merge with bf16; weight-tying
preservation; merged model reloads; merged ≡ base+adapter under greedy decoding;
tokenizer/template bundled.

**Not verified here (no GPU, no device):** GGUF/ONNX conversion; quantised
accuracy; throughput and latency; KV-cache memory on device; llama.cpp/vLLM
loading. These require the actual target hardware and the official Liquid
inference docs.