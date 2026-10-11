"""Generation harness shared by baseline eval, after-training eval and inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..data.schema import Conversation
from ..training.tokenization import ChatFormatter
from ..utils.errors import EvaluationError
from ..utils.logging import get_logger

log = get_logger("inference.generator")


@dataclass
class GenerationConfig:
    """Runtime generation parameters. Values come from ``evaluation`` in YAML."""

    max_new_tokens: int = 320
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    num_beams: int = 1
    batch_size: int = 1
    seed: int = 42

    @classmethod
    def from_experiment_config(cls, config: Any) -> GenerationConfig:
        ev = config.evaluation
        return cls(
            max_new_tokens=ev.max_new_tokens,
            do_sample=ev.do_sample,
            temperature=ev.temperature,
            top_p=ev.top_p,
            top_k=ev.top_k,
            repetition_penalty=ev.repetition_penalty,
            num_beams=ev.num_beams,
            batch_size=ev.batch_size,
            seed=ev.seed,
        )

    def to_kwargs(self) -> dict[str, Any]:
        """Generation kwargs, omitting parameters that are invalid in combination.

        ``temperature``/``top_p``/``top_k`` are meaningless (and rejected by
        transformers in recent versions) when ``do_sample=False``, so they are
        dropped for greedy decoding.
        """
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            kwargs["temperature"] = max(1e-5, self.temperature)
            kwargs["top_p"] = self.top_p
            if self.top_k > 0:
                kwargs["top_k"] = self.top_k
        elif self.num_beams > 1:
            # Beam search is the deterministic alternative to sampling.
            kwargs["early_stopping"] = True
        return kwargs

    def describe(self) -> str:
        if self.do_sample:
            return (f"sampling: temperature={self.temperature} top_p={self.top_p} "
                    f"top_k={self.top_k or 'off'} seed={self.seed}")
        if self.num_beams > 1:
            return f"beam search: num_beams={self.num_beams} (deterministic)"
        return "greedy decoding (deterministic, reproducible)"


class Generator:
    """Thin, well-instrumented wrapper around ``model.generate``."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        formatter: ChatFormatter,
        *,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.device = device

    def generate_one(self, messages: Sequence[dict[str, Any]], gen_config: GenerationConfig) -> str:
        prompt_ids = self.formatter.build_prompt_ids(messages, add_generation_prompt=True)
        if not prompt_ids:
            raise EvaluationError(
                "Empty prompt after chat-template rendering.",
                hint="The conversation has no user turn, or max_seq_length is smaller than "
                "the rendered system prompt.",
            )
        return self._generate_from_ids([prompt_ids], gen_config)[0]

    def generate_batch(
        self, batch_messages: Sequence[Sequence[dict[str, Any]]], gen_config: GenerationConfig
    ) -> list[str]:
        """Generate for several conversations, returning outputs in input order."""
        ids = [self.formatter.build_prompt_ids(m, add_generation_prompt=True) for m in batch_messages]
        return self._generate_from_ids(ids, gen_config)

    def _generate_from_ids(self, prompts: list[list[int]], gen_config: GenerationConfig) -> list[str]:
        import torch

        pad_id = self.formatter.pad_token_id
        max_len = max(len(p) for p in prompts)
        # Left padding for generation: a decoder-only model must have the real
        # prompt ending at the final position for every row in the batch.
        input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(prompts), max_len), dtype=torch.long)
        for i, p in enumerate(prompts):
            input_ids[i, max_len - len(p) :] = torch.tensor(p, dtype=torch.long)
            attention_mask[i, max_len - len(p) :] = 1

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        was_cache = getattr(self.model.config, "use_cache", None)
        self.model.config.use_cache = True
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_config.to_kwargs(),
                    **self.formatter.generation_kwargs(),
                    pad_token_id=pad_id,
                )
        except Exception as exc:
            raise EvaluationError(
                f"Generation failed: {exc}",
                hint="If this is an OOM, lower evaluation.max_new_tokens or "
                "evaluation.batch_size. If it is 'pad_token_id must be set', the "
                "tokenizer_config is missing a pad token.",
            ) from exc
        finally:
            self.model.config.use_cache = was_cache

        results: list[str] = []
        for row in outputs:
            # Generated tokens always begin at position ``max_len``, i.e. after the
            # longest padded prompt. Slicing from the start of *this row's* prompt
            # would silently return the prompt itself whenever the row needed no
            # left padding (which is every row in a batch of one).
            generated = row[max_len:]
            results.append(self.formatter.strip_generated(generated.tolist()))
        return results


@dataclass
class GenerationRun:
    """Outputs of one generation sweep, ready for scoring or inspection."""

    label: str
    samples: list[dict[str, Any]] = field(default_factory=list)
    generation_config: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "duration_seconds": round(self.duration_seconds, 2),
            "generation_config": self.generation_config,
            "num_samples": len(self.samples),
            "samples": self.samples,
        }


def run_generation_sweep(
    generator: Generator,
    eval_items: list[dict[str, Any]],
    gen_config: GenerationConfig,
    *,
    label: str = "model",
) -> GenerationRun:
    """Generate responses for every evaluation item, in batches."""
    import time

    from ..utils.seeding import set_seed

    set_seed(gen_config.seed)

    run = GenerationRun(label=label, generation_config={
        **gen_config.to_kwargs(),
        "seed": gen_config.seed,
        "strategy": gen_config.describe(),
        "batch_size": gen_config.batch_size,
    })
    started = time.time()

    for start in range(0, len(eval_items), gen_config.batch_size):
        chunk = eval_items[start : start + gen_config.batch_size]
        outputs = generator.generate_batch([item["messages"] for item in chunk], gen_config)
        for item, output in zip(chunk, outputs, strict=True):
            run.samples.append({
                "id": item.get("id"),
                "prompt": item.get("prompt"),
                "reference": item.get("reference"),
                "generation": output,
                "metadata": item.get("metadata", {}),
            })
        log.info("[%s] generated %d/%d", label, min(start + len(chunk), len(eval_items)), len(eval_items))

    run.duration_seconds = time.time() - started
    return run


def conversations_to_items(conversations: list[Conversation], *, limit: int | None = None) -> list[dict[str, Any]]:
    from ..evaluation.metrics import build_eval_set

    return build_eval_set(conversations, limit=limit)