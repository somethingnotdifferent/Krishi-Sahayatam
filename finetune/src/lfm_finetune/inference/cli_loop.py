"""Interactive and one-shot inference over base+adapter or a merged model."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.schema import ExperimentConfig
from ..inference.generator import GenerationConfig, Generator
from ..training.tokenization import ChatFormatter
from ..utils.errors import CheckpointError, ModelError
from ..utils.logging import get_logger
from ..utils.seeding import set_seed

log = get_logger("inference.cli")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely and say when you are uncertain."
)


@dataclass
class InferenceEngine:
    """Base model + optional adapter, ready for chat turns.

    Args:
        config: Experiment config (supplies the tokenizer, chat template and
            generation defaults).
        adapter_path: LoRA adapter directory. ``None`` runs the raw base model,
            which is what the baseline evaluation needs.
        merged_path: A previously merged model directory. Mutually exclusive
            with ``adapter_path``.
    """

    config: ExperimentConfig
    model: Any
    tokenizer: Any
    formatter: ChatFormatter
    device: str
    source: str

    @classmethod
    def load(
        cls,
        config: ExperimentConfig,
        *,
        adapter_path: Path | str | None = None,
        merged_path: Path | str | None = None,
        dtype: str | None = None,
    ) -> InferenceEngine:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..model.loader import _resolve_dtype, resolve_device
        from ..utils.hardware import audit

        if adapter_path and merged_path:
            raise ModelError(
                "Pass either adapter_path or merged_path, not both.",
                hint="A merged model already contains the adapter's weights.",
            )

        hardware = audit()
        resolved_dtype = dtype or _resolve_dtype(config.model.dtype, hardware)
        device = resolve_device(config.model.device, resolved_dtype)

        if merged_path:
            source = f"merged:{merged_path}"
            log.info("Loading merged model from %s", merged_path)
            model = AutoModelForCausalLM.from_pretrained(
                str(merged_path), dtype=resolved_dtype, low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(str(merged_path))
            model = model.to(device)
        else:
            source = f"base+adapter:{adapter_path}" if adapter_path else f"base:{config.model.name}"
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.name,
                revision=config.model.revision,
                trust_remote_code=config.model.trust_remote_code,
            )
            if config.tokenization.chat_template:
                template = Path(config.tokenization.chat_template)
                if not template.exists():
                    raise CheckpointError(f"tokenization.chat_template not found: {template}")
                tokenizer.chat_template = template.read_text(encoding="utf-8")
            model = AutoModelForCausalLM.from_pretrained(
                config.model.name,
                revision=config.model.revision,
                dtype=resolved_dtype,
                low_cpu_mem_usage=True,
                attn_implementation=config.model.attn_implementation
                if config.model.attn_implementation != "flash_attention_2"
                else "sdpa",
            )
            model = model.to(device)
            if adapter_path:
                from peft import PeftModel

                try:
                    model = PeftModel.from_pretrained(model, str(adapter_path))
                except Exception as exc:
                    raise CheckpointError(
                        f"Could not load adapter {adapter_path}: {exc}",
                        hint="The adapter must match the base model id and revision it was "
                        "trained against. Check adapter/export_info.json.",
                    ) from exc
                model = model.eval()

        model.eval()
        formatter = ChatFormatter(tokenizer, config.tokenization)
        log.info("Inference engine ready: %s (dtype=%s device=%s)", source, resolved_dtype, device)
        return cls(
            config=config,
            model=model,
            tokenizer=tokenizer,
            formatter=formatter,
            device=str(next(model.parameters()).device),
            source=source,
        )

    def generator(self, gen_config: GenerationConfig | None = None) -> Generator:
        return Generator(self.model, self.tokenizer, self.formatter, device=self.device)

    def chat(self, messages: list[dict[str, Any]], gen_config: GenerationConfig | None = None) -> str:
        gen_config = gen_config or GenerationConfig.from_experiment_config(self.config)
        return self.generator(gen_config).generate_one(messages, gen_config)

    def default_generation_config(self) -> GenerationConfig:
        return GenerationConfig.from_experiment_config(self.config)


BANNER = r"""
  __ _ __ __ | | ___  ___ ___ _ __   ___ _ __
 / _` | '__| | |/ / |/ __/ __| '_ \ / _ \ '_ \
| (_| | |  | |   <| \__ \ (__| | | |  __/ | | |
 \__,_|_|  |_|_|\_\ |___/___|_| |_|\___|_| |_|   Krishi Sahayatam / MICO

 LFM2 / LFM2.5 LoRA inference
 Commands: /help  /system <text>  /reset  /params  /save <path>  /quit
"""


def run_interactive(
    engine: InferenceEngine,
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    stream: bool = False,
) -> int:
    """Minimal REPL. Works over pipes (for scripted testing) as well as a tty."""
    gen_config = engine.default_generation_config()
    set_seed(gen_config.seed)

    active_system = system_prompt
    history: list[dict[str, Any]] = (
        [{"role": "system", "content": active_system}] if active_system else []
    )

    print(BANNER)
    print(f"model source : {engine.source}")
    print(f"decoding     : {gen_config.describe()}")
    print(f"max_new_tokens: {gen_config.max_new_tokens}\n")

    while True:
        try:
            line = input("farmer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        if line.startswith("/"):
            command, _, rest = line.partition(" ")
            command = command.lower()
            if command in ("/quit", "/exit", "/q"):
                return 0
            if command == "/help":
                print(__doc__ or "")
                print(BANNER)
                continue
            if command == "/reset":
                history = [{"role": "system", "content": active_system}] if active_system else []
                print("conversation reset.")
                continue
            if command == "/system":
                active_system = rest.strip() or None
                history = ([{"role": "system", "content": active_system}] if active_system else [])
                print(f"system prompt set to: {active_system!r}")
                continue
            if command == "/params":
                print(f"decoding        : {gen_config.describe()}")
                print(f"max_new_tokens  : {gen_config.max_new_tokens}")
                print(f"repetition_pen.  : {gen_config.repetition_penalty}")
                print(f"max_seq_length  : {engine.config.tokenization.max_seq_length}")
                print(f"append_eos      : {engine.config.tokenization.append_eos_token}")
                continue
            if command == "/save":
                if not rest:
                    print("usage: /save <path.jsonl>")
                    continue
                from ..utils.io import write_jsonl

                write_jsonl(rest, history)
                print(f"saved conversation to {rest}")
                continue
            print(f"unknown command {command!r}; try /help")
            continue

        history.append({"role": "user", "content": line})
        try:
            answer = engine.chat(history, gen_config)
        except Exception as exc:
            print(f"[generation failed] {exc}", file=sys.stderr)
            history.pop()
            continue
        history.append({"role": "assistant", "content": answer})
        print(f"assistant> {answer}\n")


def run_batch(
    engine: InferenceEngine,
    prompts: list[str],
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    output_path: Path | None = None,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """One-shot inference over a list of prompts (for scripted evaluation).

    Args:
        quiet: Suppress the human-readable echo of each prompt/answer. Required
            when the caller asked for JSON on stdout -- otherwise the transcript
            is interleaved into the document and ``json.loads`` fails, which
            breaks exactly the scripted usage this mode exists for.
    """
    gen_config = engine.default_generation_config()
    set_seed(gen_config.seed)
    generator = engine.generator(gen_config)

    base: list[dict[str, Any]] = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    records: list[dict[str, Any]] = []
    for prompt in prompts:
        messages = base + [{"role": "user", "content": prompt}]
        answer = generator.generate_one(messages, gen_config)
        records.append({"prompt": prompt, "generation": answer})
        if not quiet:
            print(f"> {prompt}\n{answer}\n")

    if output_path:
        from ..utils.io import write_jsonl

        write_jsonl(output_path, records)
        log.info("Wrote %d generations to %s", len(records), output_path)
    return records