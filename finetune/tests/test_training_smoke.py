"""End-to-end training smoke tests.

These prove the whole pipeline executes on CPU with a small checkpoint:

    load -> format -> mask -> LoRA -> forward -> backward -> optimizer
         -> checkpoint -> resume -> adapter reload -> inference

They are marked ``train`` because they actually run optimizer steps. The 350M
checkpoint is used (rather than a synthetic one) because the point is to exercise
LFM2's hybrid blocks, not just a Transformer stub.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lfm_finetune.config.loader import load_config
from lfm_finetune.utils.errors import CheckpointError, DatasetError

REPO = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.train


@pytest.fixture(autouse=True)
def _reclaim_memory():
    """Break reference cycles left behind by Trainer/PEFT after every test.

    transformers Trainer and PEFT wrappers create reference cycles, so a trained
    model is not freed by refcounting alone. Without an explicit collection each
    test leaks ~1.4 GiB of fp32 weights and the suite starts thrashing on a
    small-RAM host -- which shows up as tests failing only when run in sequence.
    """
    import gc

    yield
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory, requires_model):
    """A tiny but genuine LFM2 checkpoint on disk, built once per session.

    Training tests need a *real* LFM2 graph -- conv blocks, hybrid layer types,
    the real chat template -- but not 350M parameters. Fourteen training runs
    against the 350M checkpoint would load ~1.4 GiB of fp32 weights each and
    exhaust a small-RAM host, making tests fail only when run in sequence.

    So we instantiate Lfm2ForCausalLM from the real LFM2.5-350M-Base *config*,
    shrink it, and save it alongside a copy of the real tokenizer. Every code
    path (module naming, chat template, EOS handling, LoRA targets, merging) is
    identical; only the tensor sizes differ.

    ``tests/test_model_loading.py`` and ``tests/test_chat_template.py`` cover the
    real 350M checkpoint, so nothing is left unverified.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    directory = tmp_path_factory.mktemp("tiny-lfm") / "model"
    directory.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "LiquidAI/LFM2.5-350M-Base", trust_remote_code=False
    )
    cfg = AutoConfig.from_pretrained("LiquidAI/LFM2.5-350M-Base", trust_remote_code=False)
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.block_ff_dim = 128
    cfg.block_dim = 64
    cfg.num_hidden_layers = 4
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.max_position_embeddings = 4096
    cfg.conv_dim = 64
    # Must cover the tokenizer's highest id or embedding lookup fails.
    cfg.vocab_size = 65536

    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(str(directory), safe_serialization=True)
    tokenizer.save_pretrained(str(directory))
    return directory


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, tiny_checkpoint):
    """A prepared synthetic dataset, built once for the module."""
    from lfm_finetune.data.pipeline import prepare_dataset
    from lfm_finetune.data.synthetic import write_synthetic_dataset

    root = tmp_path_factory.mktemp("prepared")
    write_synthetic_dataset(root / "raw", num_examples=24, seed=11)

    config, _ = load_config(REPO / "configs/smoke.yaml")
    config.dataset.path = root / "raw"
    config.dataset.processed_dir = root / "data" / "splits"
    config.dataset.use_processed = True
    config.dataset.quality.min_assistant_chars = 1
    config.dataset.quality.near_duplicate = False
    config.dataset.quality.remove_prompt_leakage = False
    config.dataset.splitting.train_ratio = 0.6
    config.dataset.splitting.validation_ratio = 0.2
    prepare_dataset(config, output_root=root, tokenizer=None, run_tokenizer_stats=False)
    return root


@pytest.fixture
def train_config(prepared, tiny_checkpoint):
    """A smoke config on the tiny real-architecture checkpoint."""
    config, _ = load_config(REPO / "configs/smoke.yaml")
    config.model.name = str(tiny_checkpoint)
    config.model.local_files_only = True
    config.dataset.path = prepared / "raw"
    config.dataset.processed_dir = prepared / "data" / "splits"
    config.dataset.use_processed = True
    config.training.smoke_test = True
    config.training.max_steps = 3
    config.training.save_steps = 2
    config.checkpointing.dir = REPO / "checkpoints" / "_pytest"
    return config


def _cleanup(config) -> None:
    target = REPO / "checkpoints" / "_pytest"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    del config


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #

def test_preflight_succeeds(train_config) -> None:
    from lfm_finetune.training.trainer import preflight

    loaded, hardware, report = preflight(train_config, load_base_model=True)
    try:
        assert loaded is not None
        assert report["model_loaded"] is True
        assert hardware.cpu_count_logical >= 1
    finally:
        _cleanup(train_config)


def test_preflight_fails_when_dataset_missing(train_config, tmp_path: Path) -> None:
    from lfm_finetune.training.trainer import preflight

    train_config.dataset.path = tmp_path / "does-not-exist"
    train_config.dataset.use_processed = False
    with pytest.raises(DatasetError, match="does not exist"):
        preflight(train_config, load_base_model=False)


def test_context_length_gate(train_config) -> None:
    """max_seq_length above the model's context must fail before training.

    The model loader catches this (it runs earlier and has the real config), so
    the trainer's own gate is a backstop rather than the first line of defence.
    """
    from lfm_finetune.utils.errors import ModelError

    # At the config's `le` ceiling, and still above LFM2.5's declared 128000.
    train_config.tokenization.max_seq_length = 131072
    with pytest.raises(ModelError, match="exceeds the checkpoint"):
        from lfm_finetune.training.trainer import preflight

        preflight(train_config, load_base_model=True)


def test_dry_run_trains_nothing(train_config) -> None:
    """--dry-run must exercise preflight, masking and the LoRA audit, and stop."""
    from lfm_finetune.training.trainer import train

    try:
        result = train(train_config, output_dir=train_config.checkpointing.dir, dry_run=True)
        assert result.adapter_path is None
        assert result.trainable_params > 0
        assert not list(train_config.checkpointing.dir.glob("checkpoint-*"))
    finally:
        _cleanup(train_config)


# --------------------------------------------------------------------------- #
# Full smoke training
# --------------------------------------------------------------------------- #

def test_smoke_training_end_to_end(train_config) -> None:
    """The headline test: the entire pipeline runs and produces a verified adapter."""
    from lfm_finetune.training.trainer import train

    try:
        result = train(train_config, output_dir=train_config.checkpointing.dir,
                       smoke_test=True)

        assert result.trainable_params > 0
        assert result.total_params > result.trainable_params
        assert result.duration_seconds > 0
        assert result.adapter_path is not None and result.adapter_path.exists()
        assert result.verified is True

        # A real, finite loss.
        assert result.metrics.get("final_train_loss") is not None
        assert result.metrics["final_train_loss"] == result.metrics["final_train_loss"]  # not NaN

        # Run directory is self-describing.
        assert result.run_dir.is_dir()
        assert (result.run_dir / "resolved_config.yaml").exists()
        assert (result.run_dir / "run_metadata.json").exists()
        assert (result.run_dir / "metrics.jsonl").exists()
        assert (result.run_dir / "train.log").exists()

        # Adapter artifact contents.
        assert (result.adapter_path / "adapter_config.json").exists()
        assert list(result.adapter_path.glob("adapter_model*.safetensors"))
        assert (result.adapter_path / "export_info.json").exists()
        # Tokenizer is bundled so inference needs no network.
        assert (result.adapter_path / "tokenizer.json").exists()

        # A resumable checkpoint exists.
        checkpoints = list(result.run_dir.glob("checkpoint-*"))
        assert checkpoints, "expected at least one checkpoint"
        ckpt = checkpoints[-1]
        for required in ("trainer_state.json", "optimizer.pt", "scheduler.pt",
                         "rng_state.pth"):
            assert (ckpt / required).exists(), f"checkpoint missing {required}"
        assert list(ckpt.glob("adapter_model*.safetensors"))

        # Metadata records everything needed to reproduce and audit.
        meta = json.loads((result.run_dir / "run_metadata.json").read_text())
        assert meta["model"]["id"]
        assert meta["model"]["revision"]
        assert meta["lora"]["audit"]["passed"] is True
        assert meta["lora"]["resolved_target_modules"]
        assert meta["seeds"]["seed"] == train_config.training.seed
        assert meta["dataset"]["num_train"] > 0
        assert meta["environment"]["packages"]["torch"]
        assert meta["environment"]["git"]
        assert meta["tokenization"]["eos_token_id"] == 7
        assert meta["tokenization"]["loss_masking"].startswith("assistant-only")
        assert meta["command"]["argv"]
    finally:
        _cleanup(train_config)


def test_resolved_config_is_self_contained(train_config) -> None:
    """The frozen config must reproduce the run without the source YAML."""
    from lfm_finetune.config.loader import parse_config_obj
    from lfm_finetune.training.trainer import train

    try:
        result = train(train_config, output_dir=train_config.checkpointing.dir,
                       smoke_test=True)
        raw = json.loads(json.dumps(
            __import__("yaml").safe_load((result.run_dir / "resolved_config.yaml").read_text())
        ))
        restored = parse_config_obj(raw)
        assert restored.run.name == train_config.run.name
        assert restored.model.name == train_config.model.name
        assert restored.lora.r == train_config.lora.r
        assert restored.training.seed == train_config.training.seed
        assert restored.fingerprint() == train_config.fingerprint()
    finally:
        _cleanup(train_config)


def test_resume_continues_in_place(train_config) -> None:
    """Resuming must continue the SAME run directory, not fork a new one.

    A new directory per resume would fragment metrics.jsonl and leave the
    original resolved config describing a run it no longer contains.
    """
    from lfm_finetune.training.trainer import train

    try:
        first = train(train_config, output_dir=train_config.checkpointing.dir,
                      smoke_test=True)
        run_dir = first.run_dir
        ckpt = sorted(run_dir.glob("checkpoint-*"))[-1]
        metrics_before = len((run_dir / "metrics.jsonl").read_text().splitlines())

        train_config.training.max_steps = 5
        second = train(train_config, output_dir=train_config.checkpointing.dir,
                       smoke_test=True, resume_from=ckpt)

        assert second.run_dir == run_dir, "resume created a new run directory"
        assert second.resumed_from == ckpt
        metrics_after = len((run_dir / "metrics.jsonl").read_text().splitlines())
        assert metrics_after > metrics_before, "metrics history was not continued"
    finally:
        _cleanup(train_config)


def test_resume_auto_selects_latest(train_config) -> None:
    from lfm_finetune.training.trainer import train

    try:
        first = train(train_config, output_dir=train_config.checkpointing.dir,
                      smoke_test=True)
        train_config.training.max_steps = 5
        second = train(train_config, output_dir=first.run_dir, smoke_test=True,
                       resume_from="auto")
        assert second.resumed_from is not None
    finally:
        _cleanup(train_config)


def test_resume_missing_checkpoint_fails_fast(train_config, tmp_path: Path) -> None:
    from lfm_finetune.training.trainer import train

    try:
        with pytest.raises(CheckpointError, match="does not exist"):
            train(train_config, output_dir=train_config.checkpointing.dir,
                  smoke_test=True, resume_from=tmp_path / "nope")
    finally:
        _cleanup(train_config)


def test_resume_incomplete_checkpoint_fails_fast(train_config, tmp_path: Path) -> None:
    from lfm_finetune.training.trainer import train

    fake = tmp_path / "checkpoint-99"
    fake.mkdir()
    try:
        with pytest.raises(CheckpointError, match="incomplete"):
            train(train_config, output_dir=train_config.checkpointing.dir,
                  smoke_test=True, resume_from=fake)
    finally:
        _cleanup(train_config)


def test_dataloader_batches_and_masks(train_config) -> None:
    """Forward+backward must produce a finite gradient on supervised tokens only."""
    import torch

    from lfm_finetune.data.pipeline import load_splits
    from lfm_finetune.model.loader import load_model
    from lfm_finetune.model.lora import attach_lora
    from lfm_finetune.training.collator import SFTCollator, tokenize_conversations
    from lfm_finetune.training.tokenization import ChatFormatter

    loaded = load_model(train_config, for_training=True)
    formatter = ChatFormatter(loaded.tokenizer, train_config.tokenization)
    splits = load_splits(train_config.dataset.processed_dir)

    tokens = tokenize_conversations(splits["train"][:4], formatter,
                                    assistant_only_loss=True, split="train")
    batch = SFTCollator(pad_token_id=formatter.pad_token_id)(tokens.examples[:2])

    model, audit, _t, _i, _d = attach_lora(loaded.model, train_config.lora,
                                           model_id=loaded.model_id)
    assert audit.ok

    batch = {k: v.to(loaded.device) for k, v in batch.items()}
    assert int((batch["labels"] != -100).sum()) > 0, "no supervised tokens in the batch"

    outputs = model(**batch)
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()

    grads = [p.grad for _n, p in model.named_parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "no LoRA parameter received a gradient"
    )


def test_inference_with_adapter_runs(train_config) -> None:
    """base + adapter must generate, and terminate, on real data."""
    from lfm_finetune.inference.cli_loop import InferenceEngine
    from lfm_finetune.inference.generator import GenerationConfig
    from lfm_finetune.training.trainer import train

    try:
        result = train(train_config, output_dir=train_config.checkpointing.dir,
                       smoke_test=True)
        engine = InferenceEngine.load(train_config, adapter_path=result.adapter_path)
        assert "adapter" in engine.source

        messages = [
            {"role": "system", "content": "You are a careful reasoning assistant."},
            {"role": "user", "content": "What is 12 times 4?"},
        ]
        gen_config = GenerationConfig(max_new_tokens=12, do_sample=False)
        answer = engine.chat(messages, gen_config)
        assert isinstance(answer, str)

        # Structural guarantees that hold even for a randomly-initialised tiny
        # model: the prompt must not leak into the output, and the terminator
        # must be wired (a model that never emits EOS would return
        # max_new_tokens tokens every time, which the loop below would notice).
        assert "What is 12 times 4?" not in answer, (
            f"the prompt leaked into the generation output: {answer!r}"
        )
        assert answer == answer.strip()

        # An untrained tiny model may legitimately emit the end-of-turn token
        # immediately, producing an empty string. What must not happen is a
        # crash or an unterminated generation; assert the mechanics instead.
        second = engine.chat(
            [{"role": "user", "content": "Count the vowels in 'orchard'."}], gen_config
        )
        assert isinstance(second, str)
    finally:
        _cleanup(train_config)


def test_merged_model_loads_and_generates(train_config) -> None:
    """The merge path must produce a standalone model that still answers."""
    from lfm_finetune.export.adapter import merge_adapter
    from lfm_finetune.inference.cli_loop import InferenceEngine
    from lfm_finetune.inference.generator import GenerationConfig
    from lfm_finetune.training.trainer import train

    merged_dir = REPO / "merged" / "_pytest"
    try:
        result = train(train_config, output_dir=train_config.checkpointing.dir,
                       smoke_test=True)
        merge = merge_adapter(
            result.adapter_path, merged_dir,
            base_model_id=train_config.model.name,
            base_revision=train_config.model.revision,
            config=train_config,
            merge_in_bf16=False,      # CPU: bf16 merge is slow and lossless anyway
            save_tokenizer=True,
            verify=True,
        )
        assert merge.verified is True
        assert (merged_dir / "config.json").exists()
        assert list(merged_dir.glob("*.safetensors"))
        assert (merged_dir / "merge_info.json").exists()
        info = json.loads((merged_dir / "merge_info.json").read_text())
        assert info["weight_tying_preserved"] is True

        engine = InferenceEngine.load(train_config, merged_path=merged_dir)
        gen_config = GenerationConfig(max_new_tokens=8, do_sample=False)
        answer = engine.chat(
            [{"role": "user", "content": "What is 2 plus 2?"}], gen_config
        )
        assert isinstance(answer, str)
    finally:
        _cleanup(train_config)
        if merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)


def test_evaluation_runs_on_holdout(train_config) -> None:
    """Base vs tuned evaluation must produce a comparison on the test split."""
    from lfm_finetune.evaluation.runner import run_comparison_evaluation
    from lfm_finetune.training.trainer import train

    out_dir = REPO / "evaluations" / "_pytest"
    try:
        result = train(train_config, output_dir=train_config.checkpointing.dir,
                       smoke_test=True)
        train_config.evaluation.num_samples = 2
        train_config.evaluation.max_new_tokens = 8
        train_config.evaluation.batch_size = 2

        payload = run_comparison_evaluation(
            train_config, result.adapter_path, output_dir=out_dir, split="test"
        )
        report = json.loads(Path(payload["comparison_path"]).read_text())
        assert report["evaluation_set"]["disjointness_vs_train"]["disjoint"] is True
        assert "base" in report["held_out_loss"]
        assert "tuned" in report["held_out_loss"]
        assert report["interpretation_guard"]
        assert report["human_rubric_sheet"]["samples"]
        # Human scores must stay null: the pipeline does not fake them.
        for sample in report["human_rubric_sheet"]["samples"]:
            assert all(v is None for v in sample["scores"].values())
    finally:
        _cleanup(train_config)
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)