"""Configuration loading, inheritance, overrides and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from lfm_finetune.config.loader import deep_merge, load_config, parse_set_args
from lfm_finetune.config.schema import ExperimentConfig, parse_config_obj
from lfm_finetune.utils.errors import ConfigError

# --------------------------------------------------------------------------- #
# Basic validation
# --------------------------------------------------------------------------- #

def test_minimal_config_validates(base_config_dict: dict) -> None:
    config = ExperimentConfig.model_validate(base_config_dict)
    assert config.run.name == "test-run"
    assert config.training.preset == "smoke"


def test_unknown_top_level_key_rejected(base_config_dict: dict) -> None:
    base_config_dict["totally_unexpected"] = 1
    with pytest.raises(ValidationError, match="extra_forbidden|not permitted"):
        ExperimentConfig.model_validate(base_config_dict)


def test_unknown_nested_key_rejected(base_config_dict: dict) -> None:
    """A typo inside a section must fail, not be silently ignored.

    Silently dropping `learning_ratee: 1e-4` is how people end up training at an
    unexpected learning rate for a month.
    """
    base_config_dict["training"]["learning_ratee"] = 1e-4
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base_config_dict)


def test_dataset_path_required_without_use_processed(base_config_dict: dict) -> None:
    base_config_dict["dataset"]["path"] = None
    base_config_dict["dataset"]["use_processed"] = False
    with pytest.raises(Exception, match="dataset.path is required"):
        ExperimentConfig.model_validate(base_config_dict)


def test_split_ratios_must_not_exceed_one() -> None:
    from lfm_finetune.config.schema import SplittingConfig

    with pytest.raises(ValidationError):
        SplittingConfig(train_ratio=0.8, validation_ratio=0.5)


def test_lora_rank_vs_alpha_guard(base_config_dict: dict) -> None:
    base_config_dict["lora"] = {"r": 64, "alpha": 8, "target_modules": "auto"}
    with pytest.raises(Exception, match="lora.r=64 is far above"):
        ExperimentConfig.model_validate(base_config_dict)


def test_lm_head_in_modules_to_save_rejected(base_config_dict: dict) -> None:
    """On LFM2.5 base checkpoints lm_head is tied to the embeddings.

    Training it separately breaks the tie, so the config layer refuses it.
    """
    base_config_dict["lora"] = {
        "r": 8, "alpha": 16, "target_modules": "auto", "modules_to_save": ["lm_head"]
    }
    with pytest.raises(Exception, match="ties\n?lm_head|lm_head"):
        ExperimentConfig.model_validate(base_config_dict)


def test_seed_mismatch_rejected(base_config_dict: dict) -> None:
    base_config_dict["dataset"]["splitting"] = {"seed": 7, "train_ratio": 0.8,
                                                 "validation_ratio": 0.1}
    with pytest.raises(Exception, match="dataset_split_seed"):
        ExperimentConfig.model_validate(base_config_dict)


def test_do_sample_with_zero_temperature_rejected(base_config_dict: dict) -> None:
    base_config_dict["evaluation"] = {"do_sample": True, "temperature": 0.0}
    with pytest.raises(Exception, match="do_sample"):
        ExperimentConfig.model_validate(base_config_dict)


def test_load_best_model_requires_eval_and_save(base_config_dict: dict) -> None:
    base_config_dict["training"] = {
        "load_best_model_at_end": True, "eval_strategy": "no", "save_strategy": "no"
    }
    with pytest.raises(Exception, match="load_best_model_at_end"):
        ExperimentConfig.model_validate(base_config_dict)


def test_absurd_effective_batch_rejected(base_config_dict: dict) -> None:
    base_config_dict["training"] = {
        "per_device_train_batch_size": 64, "gradient_accumulation_steps": 16
    }
    with pytest.raises(Exception, match="Effective batch size"):
        ExperimentConfig.model_validate(base_config_dict)


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("preset", ["smoke", "pilot", "balanced", "memory_saver", "full"])
def test_all_presets_validate(preset: str) -> None:
    config = ExperimentConfig.model_validate({
        "run": {"name": "p"},
        "model": {"name": "LiquidAI/LFM2.5-1.2B-Base"},
        "dataset": {"path": "x.jsonl"},
        "training": {"preset": preset},
    })
    assert config.training.preset == preset
    assert config.training.learning_rate > 0


def test_unknown_preset_rejected() -> None:
    with pytest.raises(Exception, match="Unknown training.preset"):
        ExperimentConfig.model_validate({
            "run": {"name": "p"}, "model": {"name": "m"},
            "dataset": {"path": "x"}, "training": {"preset": "turbo"},
        })


def test_explicit_field_beats_preset() -> None:
    config = ExperimentConfig.model_validate({
        "run": {"name": "p"}, "model": {"name": "m"}, "dataset": {"path": "x"},
        "training": {"preset": "balanced", "learning_rate": 1.25e-5},
    })
    assert config.training.learning_rate == pytest.approx(1.25e-5)


def test_preset_only_sets_training_keys() -> None:
    """A preset that leaks a key into the wrong section must fail loudly.

    Regression test: an early version of `balanced` set `max_seq_length` (a
    tokenization key) and pydantic reported the error against the preset name
    rather than the typo.
    """
    from lfm_finetune.config.schema import TrainingConfigModel

    for preset, keys in ExperimentConfig.PRESETS.items():
        illegal = set(keys) - set(TrainingConfigModel.model_fields)
        assert not illegal, f"preset {preset!r} sets non-training keys: {illegal}"


# --------------------------------------------------------------------------- #
# Inheritance
# --------------------------------------------------------------------------- #

def test_extends_chain_merges_deeply(tmp_path: Path) -> None:
    (tmp_path / "parent.yaml").write_text(textwrap.dedent("""
        run: {name: parent, domain: p}
        model: {name: LiquidAI/LFM2.5-1.2B-Base, dtype: bfloat16}
        dataset: {path: parent.jsonl, quality: {min_user_chars: 3, max_turns: 10}}
        lora: {r: 8, alpha: 16}
        tokenization: {max_seq_length: 512}
    """))
    (tmp_path / "child.yaml").write_text(textwrap.dedent("""
        extends: parent.yaml
        run: {name: child}
        model: {dtype: float32}
        lora: {r: 32}
    """))

    config, provenance = load_config(tmp_path / "child.yaml")
    assert config.run.name == "child"
    assert config.run.domain == "p"            # inherited
    assert config.model.dtype == "float32"     # overridden
    assert config.model.name.endswith("1.2B-Base")  # inherited
    assert config.lora.r == 32                 # overridden
    assert config.lora.alpha == 16             # inherited sibling
    assert config.dataset.quality.min_user_chars == 3   # nested inherited
    assert config.dataset.quality.max_turns == 10
    assert config.tokenization.max_seq_length == 512
    assert len(provenance["extends_chain"]) == 2


def test_extends_list_applies_in_order(tmp_path: Path) -> None:
    """Multiple parents merge left to right; the later one wins."""
    (tmp_path / "a.yaml").write_text(
        "run: {name: p}\nmodel: {name: m}\ndataset: {path: d}\n"
        "lora: {r: 8, alpha: 8}\n"
    )
    (tmp_path / "b.yaml").write_text("lora: {r: 16}\n")
    (tmp_path / "c.yaml").write_text("extends: [a.yaml, b.yaml]\n")
    config, _ = load_config(tmp_path / "c.yaml")
    assert config.lora.r == 16     # later parent wins
    assert config.lora.alpha == 8  # only the first parent set it


def test_circular_extends_detected(tmp_path: Path) -> None:
    (tmp_path / "x.yaml").write_text("extends: y.yaml\n")
    (tmp_path / "y.yaml").write_text("extends: x.yaml\n")
    with pytest.raises(ConfigError, match="Circular"):
        load_config(tmp_path / "x.yaml")


def test_missing_extends_reports_candidates(tmp_path: Path) -> None:
    (tmp_path / "child.yaml").write_text("extends: nope.yaml\n")
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "child.yaml")


def test_null_in_child_resets_to_default(tmp_path: Path) -> None:
    """`key: null` in a child means "forget the parent value", not "inherit"."""
    (tmp_path / "parent.yaml").write_text(
        "run: {name: p}\nmodel: {name: m}\ndataset: {path: d}\n"
        "training: {learning_rate: 9e-9, seed: 5}\n"
    )
    (tmp_path / "child.yaml").write_text(
        "extends: parent.yaml\ntraining: {learning_rate: null}\n"
    )
    config, _ = load_config(tmp_path / "child.yaml")
    assert config.training.learning_rate != 9e-9   # parent's value was dropped
    assert config.training.learning_rate == 2e-4   # back to the schema default
    assert config.training.seed == 5               # sibling still inherited


def test_deep_merge_replaces_lists_not_concatenating() -> None:
    """Concatenating target_modules across an inheritance chain would silently
    grow the LoRA target set -- the exact failure this project must avoid."""
    merged = deep_merge(
        {"lora": {"target_modules": ["a", "b"]}},
        {"lora": {"target_modules": ["c"]}},
    )
    assert merged["lora"]["target_modules"] == ["c"]


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #

def test_set_args_parsed_and_applied() -> None:
    overrides = parse_set_args(["training.learning_rate=1e-4", "lora.r=32",
                                 "tokenization.max_seq_length=512",
                                 "model.dtype=float32"])
    assert overrides == {
        "training": {"learning_rate": 0.0001},
        "lora": {"r": 32},
        "tokenization": {"max_seq_length": 512},
        "model": {"dtype": "float32"},
    }


def test_set_args_json_values() -> None:
    out = parse_set_args(["lora.use_rslora=true", "training.eval_strategy=epoch",
                          "lora.rslora=null"])
    assert out["lora"]["use_rslora"] is True
    assert out["training"]["eval_strategy"] == "epoch"
    assert out["lora"]["rslora"] is None


def test_malformed_set_args_rejected() -> None:
    with pytest.raises(ConfigError, match="Malformed --set"):
        parse_set_args(["training.learning_rate"])


def test_env_overrides_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LFM_TRAINING__LEARNING_RATE", "3e-4")
    monkeypatch.setenv("LFM_LORA__R", "64")
    monkeypatch.setenv("LFM_LORA__USE_DORA", "true")
    out = __import__(
        "lfm_finetune.config.schema", fromlist=["env_overrides"]
    ).env_overrides()
    assert out["training"]["learning_rate"] == pytest.approx(3e-4)
    assert out["lora"]["r"] == 64
    assert out["lora"]["use_dora"] is True


def test_overrides_take_priority_over_yaml(tmp_path: Path) -> None:
    (tmp_path / "c.yaml").write_text(
        "run: {name: p}\nmodel: {name: LiquidAI/LFM2.5-1.2B-Base}\n"
        "dataset: {path: x}\nlora: {r: 8, alpha: 16, target_modules: auto}\n"
    )
    config, provenance = load_config(tmp_path / "c.yaml",
                                     overrides={"lora": {"r": 32}})
    assert config.lora.r == 32
    assert "cli" in provenance["overrides"]


# --------------------------------------------------------------------------- #
# Shipped configs
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "rel",
    [
        "configs/base.yaml",
        "configs/lfm_sft.yaml",
        "configs/smoke.yaml",
        "configs/agriculture/_domain.yaml",
        "configs/agriculture/lfm25_350m.yaml",
        "configs/agriculture/lfm25_1.2b.yaml",
        "configs/agriculture/lfm25_2.6b.yaml",
    ],
)
def test_shipped_configs_load(rel: str) -> None:
    """Every config in the repo must be valid -- a broken config in git is a
    broken quickstart."""
    config, provenance = load_config(REPO / rel)
    assert config.run.name
    assert config.model.name
    assert config.tokenization.max_seq_length >= 32
    assert config.lora.target_modules is not None
    assert provenance["fingerprint"]


def test_yaml_no_bare_no_is_boolean() -> None:
    """YAML 1.1 reads a bare `no` as False, which silently breaks eval_strategy."""
    raw = (REPO / "configs/base.yaml").read_text()
    for line in raw.splitlines():
        stripped = line.split("#")[0].strip()
        for key in ("eval_strategy", "save_strategy", "packing", "use_cache"):
            if stripped.startswith(f"{key}:"):
                value = stripped.split(":", 1)[1].strip()
                assert value not in ("no", "yes", "on", "off"), (
                    f"{key}: {value} is unquoted and will be parsed as a boolean"
                )


def test_agriculture_configs_use_domain_inheritance() -> None:
    for rel in ("configs/agriculture/lfm25_350m.yaml",
                "configs/agriculture/lfm25_1.2b.yaml",
                "configs/agriculture/lfm25_2.6b.yaml"):
        config, provenance = load_config(REPO / rel)
        assert config.run.domain == "agriculture"
        assert any("_domain.yaml" in p for p in provenance["extends_chain"]), (
            f"{rel} must inherit from configs/agriculture/_domain.yaml"
        )
        assert config.dataset.system_prompt, f"{rel} must define the domain system prompt"


def test_lfm_configs_never_target_generic_module_names() -> None:
    """`o_proj` / `gate_proj` do not exist on LFM2/LFM2.5.

    Copying a generic PEFT recipe is the single most common way to end up with
    an adapter that matches nothing. Assert the shipped configs do not do it.
    """
    forbidden = {"o_proj", "gate_proj", "up_proj", "down_proj", "q_proj", "v_proj"}
    for path in (REPO / "configs").rglob("*.yaml"):
        raw = path.read_text()
        for name in forbidden:
            assert f"- {name}" not in raw, (
                f"{path.name} appears to target {name!r}, which does not exist on LFM2"
            )


def test_fingerprint_changes_with_config() -> None:
    a = parse_config_obj({"run": {"name": "x"}, "model": {"name": "m"},
                          "dataset": {"path": "d"}}).fingerprint()
    b = parse_config_obj({"run": {"name": "x"}, "model": {"name": "m"},
                          "dataset": {"path": "d"}, "lora": {"r": 32}}).fingerprint()
    assert a != b


def test_to_dict_is_json_serialisable(base_config_dict: dict) -> None:
    import json

    config = ExperimentConfig.model_validate(base_config_dict)
    json.dumps(config.to_dict())  # must not raise
    assert "PRESETS" not in config.to_dict()