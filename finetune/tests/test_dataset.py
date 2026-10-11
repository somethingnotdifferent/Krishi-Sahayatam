"""Dataset ingestion, schema validation, quality gates, dedup and splitting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lfm_finetune.config.loader import load_config
from lfm_finetune.data.dedupe import (
    NearDuplicateIndex,
    answer_key,
    conversation_text,
    exact_key,
    find_cross_split_overlap,
    prompt_key,
    summarise_overlap,
)
from lfm_finetune.data.io import convert_record, detect_format, iter_raw_records
from lfm_finetune.data.pipeline import load_and_clean, prepare_dataset
from lfm_finetune.data.schema import Conversation
from lfm_finetune.data.split import build_plan, split_conversations, subsample
from lfm_finetune.data.validate import QualityConfig, TextNormalizer, validate_conversation
from lfm_finetune.utils.errors import DatasetLeakageError
from lfm_finetune.utils.io import write_jsonl

REPO = Path(__file__).resolve().parent.parent


def conv(user: str = "hello there", assistant: str = "a sufficiently long answer") -> Conversation:
    return Conversation(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ])


# --------------------------------------------------------------------------- #
# Format detection and conversion
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "record,expected",
    [
        ({"messages": [{"role": "user", "content": "u"}]}, "conversational"),
        ({"conversations": [{"from": "human", "value": "u"}]}, "conversational"),
        ({"instruction": "i", "input": "x", "output": "o"}, "instruction"),
        ({"prompt": "p", "completion": "c"}, "instruction"),
        ({"question": "q", "answer": "a"}, "instruction"),
        ({"nothing": "useful"}, None),
    ],
)
def test_detect_format(record: dict, expected: str | None) -> None:
    assert detect_format(record) == expected


def test_convert_conversational() -> None:
    c = convert_record({"messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]})
    assert [m.role for m in c.messages] == ["system", "user", "assistant"]


def test_convert_role_aliases() -> None:
    c = convert_record({"conversations": [
        {"from": "human", "value": "u"},
        {"from": "gpt", "value": "a"},
    ]})
    assert [m.role for m in c.messages] == ["user", "assistant"]


def test_convert_instruction_pairs() -> None:
    for record in (
        {"instruction": "Translate", "input": "hi", "output": "hello"},
        {"prompt": "Capitalise", "completion": "Hi"},
        {"question": "2+2?", "answer": "4"},
    ):
        c = convert_record(record, fmt="instruction")
        assert c.user_texts and c.assistant_texts
        assert c.messages[-1].role == "assistant"


def test_alpaca_input_is_appended_to_instruction() -> None:
    c = convert_record(
        {"instruction": "Summarise", "input": "long text", "output": "short"},
        fmt="instruction",
    )
    assert c.last_user == "Summarise\n\nlong text"


def test_system_prompt_injected_when_absent() -> None:
    c = convert_record({"instruction": "i", "output": "o"}, fmt="instruction",
                       system_prompt="BE SAFE")
    assert c.messages[0].role == "system"
    assert c.messages[0].content == "BE SAFE"


def test_system_prompt_not_duplicated() -> None:
    c = convert_record(
        {"messages": [{"role": "system", "content": "MINE"},
                      {"role": "user", "content": "u"},
                      {"role": "assistant", "content": "a"}]},
        system_prompt="BE SAFE",
    )
    assert [m.role for m in c.messages].count("system") == 1
    assert c.messages[0].content == "MINE"


def test_unknown_format_raises_with_guidance() -> None:
    from lfm_finetune.utils.errors import DatasetError

    with pytest.raises(DatasetError, match="Could not determine record format"):
        convert_record({"unrelated": "x"})


def test_tool_calls_preserved() -> None:
    c = convert_record({"messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get", "arguments": {"city": "Pune"}}}]},
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]})
    assert c.messages[1].tool_calls


def test_multimodal_content_blocks_collapse_to_text() -> None:
    c = convert_record({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "describe this"}]},
        {"role": "assistant", "content": "A field."},
    ]})
    assert c.user_texts == ["describe this"]


# --------------------------------------------------------------------------- #
# Quality gates
# --------------------------------------------------------------------------- #

@pytest.fixture
def quality() -> QualityConfig:
    return QualityConfig()


def test_valid_conversation_accepted(quality: QualityConfig) -> None:
    outcome = validate_conversation(
        conv(), index=0, source=None, quality=quality, normalizer=TextNormalizer()
    )
    assert outcome.rejection is None
    assert outcome.conversation is not None


@pytest.mark.parametrize(
    "messages,reason",
    [
        ([{"role": "user", "content": "why is my crop yellowing"}], "missing_turn"),
        ([{"role": "assistant", "content": "some answer text"}], "missing_turn"),
        ([{"role": "user", "content": "why is my crop yellowing"},
          {"role": "assistant", "content": "hi"}], "empty_assistant_response"),
        # A whitespace-only user turn is removed by drop_empty_messages before the
        # length gates run, so the row surfaces as missing_turn, not as an empty
        # user message. Either way it is rejected -- what matters is that it is.
        ([{"role": "user", "content": "  "},
          {"role": "assistant", "content": "a long enough answer here"}], "missing_turn"),
        ([{"role": "user", "content": "why is my crop yellowing"},
          {"role": "assistant", "content": "TODO: write this"}], "placeholder_text"),
        ([{"role": "system", "content": "s1"},
          {"role": "system", "content": "s2"},
          {"role": "user", "content": "why is my crop yellowing"},
          {"role": "assistant", "content": "a long enough answer here"}],
         "multiple_system_prompts"),
        ([{"role": "user", "content": "why is my crop yellowing"},
          {"role": "assistant", "content": "x" * 20000}], "assistant_response_too_long"),
        # Role structure is fine here, so the length gate is what rejects it.
        ([{"role": "user", "content": "why is my crop yellowing"},
          {"role": "assistant", "content": "a"}], "empty_assistant_response"),
    ],
)
def test_quality_rejections(messages: list[dict], reason: str,
                            quality: QualityConfig) -> None:
    c = Conversation(messages=messages)
    outcome = validate_conversation(
        c, index=0, source=None, quality=quality, normalizer=TextNormalizer()
    )
    assert outcome.rejection is not None, f"expected rejection {reason}"
    assert outcome.rejection.reason == reason


def test_prompt_leakage_detected() -> None:
    """An answer that is verbatim inside the prompt is templated/duplicated data."""
    answer = ("It is very likely that the crop is suffering from a nitrogen deficiency "
              "because the older leaves are uniformly pale.")
    c = Conversation(messages=[
        {"role": "user", "content": f"A farmer said: {answer}"},
        {"role": "assistant", "content": answer},
    ])
    outcome = validate_conversation(
        c, index=0, source=None, quality=QualityConfig(), normalizer=TextNormalizer()
    )
    assert outcome.rejection is not None
    assert outcome.rejection.reason == "prompt_leakage"


def test_normalisation_is_applied_and_flagged() -> None:
    c = Conversation(messages=[
        {"role": "user", "content": "  what   is\r\n2+2?  "},
        {"role": "assistant", "content": "It is 4, computed by addition."},
    ])
    outcome = validate_conversation(
        c, index=0, source=None, quality=QualityConfig(), normalizer=TextNormalizer()
    )
    assert outcome.conversation is not None
    assert outcome.modified is True
    assert "\r" not in outcome.conversation.user_texts[0]


def test_control_characters_stripped() -> None:
    c = Conversation(messages=[
        {"role": "user", "content": "what\x00 is\x07 this"},
        {"role": "assistant", "content": "It is a test of cleaning behaviour."},
    ])
    outcome = validate_conversation(
        c, index=0, source=None, quality=QualityConfig(), normalizer=TextNormalizer()
    )
    assert outcome.conversation is not None
    assert "\x00" not in outcome.conversation.user_texts[0]


def test_nfc_normalisation_makes_lookalikes_equal() -> None:
    """Canonically equivalent sequences must dedup, else they leak across splits."""
    decomposed = "café question about crops"          # e + combining acute
    composed = "café question about crops"            # precomposed
    keys = {
        exact_key(conv(user=decomposed)),
        exact_key(conv(user=composed)),
    }
    assert len(keys) == 1, "NFC normalisation should make these identical"


def test_html_stripped() -> None:
    c = Conversation(messages=[
        {"role": "user", "content": "<p>what is <b>2+2</b>?</p>"},
        {"role": "assistant", "content": "It is 4 after adding both numbers."},
    ])
    outcome = validate_conversation(
        c, index=0, source=None, quality=QualityConfig(), normalizer=TextNormalizer()
    )
    assert outcome.conversation is not None
    assert "<p>" not in outcome.conversation.user_texts[0]


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #

def test_exact_key_identical_for_same_content() -> None:
    assert exact_key(conv()) == exact_key(conv())


def test_prompt_key_ignores_the_answer() -> None:
    assert prompt_key(conv(assistant="first answer")) == prompt_key(conv(assistant="second"))
    assert answer_key(conv(assistant="first")) != answer_key(conv(assistant="second"))


def test_near_duplicate_index_catches_paraphrase() -> None:
    """A wording-level variant must be caught.

    Character 5-gram Jaccard scores similarity, so the threshold has to match the
    degree of difference. A light edit of a long passage lands well above 0.7 but
    below 0.85; the near-duplicate_threshold in the domain config is set to 0.85
    precisely so that templated variants are caught without flagging genuinely
    different examples that share boilerplate.
    """
    index = NearDuplicateIndex(threshold=0.70)
    base = ("the farmer reports dark brown concentric lesions on the lower leaves of the "
            "tomato plants which spread quickly after three days of heavy rainfall")
    assert index.query(base)[0] is False           # first insert
    variant = base.replace("quickly", "rapidly").replace("three", "3")
    assert index.query(variant)[0] is True


def test_near_duplicate_index_respects_strict_threshold() -> None:
    """The same pair must NOT be flagged at a strict threshold -- otherwise the
    gate would be noise rather than signal."""
    strict = NearDuplicateIndex(threshold=0.95)
    base = ("the farmer reports dark brown concentric lesions on the lower leaves of the "
            "tomato plants which spread quickly after three days of heavy rainfall")
    strict.query(base)
    variant = base.replace("quickly", "rapidly")
    assert strict.query(variant)[0] is False


def test_near_duplicate_index_does_not_flag_unrelated() -> None:
    index = NearDuplicateIndex(threshold=0.9)
    index.query("wheat irrigation scheduling during the rabi season in punjab")
    assert index.query("sugarcane planting depth and seed rate for loamy soil")[0] is False


def test_cross_split_overlap_detected() -> None:
    a = conv(user="same question here", assistant="answer number one")
    b = conv(user="same question here", assistant="answer number two")
    findings = find_cross_split_overlap({"train": [a], "test": [b]}, keys=("prompt",))
    assert findings
    assert summarise_overlap(findings)["counts_by_key_type"]["prompt"] == 1


def test_split_overlap_verification_raises() -> None:
    a = conv(user="q", assistant="identical answer text here")
    b = conv(user="q", assistant="identical answer text here")
    from lfm_finetune.data.split import verify_no_cross_split_duplicates

    with pytest.raises(DatasetLeakageError, match="Exact duplicate"):
        verify_no_cross_split_duplicates({"train": [a], "validation": [b]})


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def test_hash_split_is_deterministic() -> None:
    items = [conv(user=f"unique question number {i}") for i in range(200)]
    plan_a = build_plan(train_ratio=0.8, validation_ratio=0.1, seed=42)
    plan_b = build_plan(train_ratio=0.8, validation_ratio=0.1, seed=42)
    a = split_conversations(items, plan_a, exact_key_fn=exact_key)
    b = split_conversations(items, plan_b, exact_key_fn=exact_key)
    assert {id(x) for x in a["train"]} == {id(x) for x in b["train"]}


def test_hash_split_respects_seed() -> None:
    items = [conv(user=f"unique question number {i}") for i in range(300)]
    p42 = split_conversations(items, build_plan(train_ratio=0.8, validation_ratio=0.1, seed=42), exact_key_fn=exact_key)
    p43 = split_conversations(items, build_plan(train_ratio=0.8, validation_ratio=0.1, seed=43), exact_key_fn=exact_key)
    assert [c.id for c in p42["train"]] != [c.id for c in p43["train"]] or \
           [exact_key(c) for c in p42["train"]] != [exact_key(c) for c in p43["train"]]


def test_hash_split_stable_under_insertion() -> None:
    """Content-addressed splitting means growing the corpus must not move
    existing examples between splits."""
    original = [conv(user=f"question {i}") for i in range(200)]
    grown = original + [conv(user=f"new question {i}") for i in range(50)]

    before = split_conversations(original, build_plan(train_ratio=0.8, validation_ratio=0.1, seed=7), exact_key_fn=exact_key)
    after = split_conversations(grown, build_plan(train_ratio=0.8, validation_ratio=0.1, seed=7), exact_key_fn=exact_key)

    train_before = {exact_key(c) for c in before["train"]}
    train_after = {exact_key(c) for c in after["train"]}
    assert train_before <= train_after, "existing training examples changed split"


def test_empty_validation_split_raises() -> None:
    items = [conv(user=f"q{i}") for i in range(3)]
    with pytest.raises(DatasetLeakageError, match="empty"):
        split_conversations(items, build_plan(train_ratio=0.99, validation_ratio=0.0, seed=1), exact_key_fn=exact_key)


def test_ratios_summing_over_one_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds 1.0"):
        build_plan(train_ratio=0.9, validation_ratio=0.3, seed=1)


def test_subsample_is_deterministic() -> None:
    items = [conv(user=f"q{i}") for i in range(100)]
    a = subsample(items, 10, seed=3)
    b = subsample(items, 10, seed=3)
    assert [exact_key(x) for x in a] == [exact_key(x) for x in b]
    assert len(a) == 10
    assert a == sorted(a, key=lambda c: items.index(c))  # original order preserved


# --------------------------------------------------------------------------- #
# End-to-end preparation
# --------------------------------------------------------------------------- #

def test_prepare_writes_all_three_splits_and_report(synthetic_raw: Path) -> None:
    config, _ = load_config(REPO / "configs/smoke.yaml")
    config.dataset.path = synthetic_raw / "raw"
    config.dataset.processed_dir = synthetic_raw / "splits"
    config.dataset.quality.min_assistant_chars = 1
    config.dataset.quality.near_duplicate = False
    config.dataset.quality.remove_prompt_leakage = False
    config.dataset.splitting.train_ratio = 0.6
    config.dataset.splitting.validation_ratio = 0.2

    result = prepare_dataset(config, output_root=synthetic_raw, tokenizer=None,
                             run_tokenizer_stats=False)

    for split in ("train", "validation", "test"):
        assert (synthetic_raw / "data" / "splits" / f"{split}.jsonl").exists()
    assert result.report_path and result.report_path.exists()

    report = json.loads(result.report_path.read_text())
    assert report["counts"]["accepted"] > 0
    assert report["counts"]["split_train"] > 0
    # Nothing is ever dropped silently: the counters and the quarantine file exist.
    assert (synthetic_raw / "data" / "validated" / "rejected.jsonl").exists()
    assert (synthetic_raw / "data" / "validated" / "accepted.jsonl").exists()
    assert report["dataset_hash"] == result.dataset_hash
    assert report["leakage"]["hard"]["hard_leakage"] is False


def test_prepare_deduplicates_before_splitting(synthetic_raw: Path) -> None:
    """Two identical rows must not land in different splits."""
    raw = synthetic_raw / "raw"
    source = raw / "synthetic.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    rows.append(dict(rows[0]))                      # exact duplicate
    write_jsonl(source, rows)

    config, _ = load_config(REPO / "configs/smoke.yaml")
    config.dataset.path = raw
    config.dataset.quality.min_assistant_chars = 1
    config.dataset.quality.near_duplicate = False
    config.dataset.quality.remove_prompt_leakage = False
    result = prepare_dataset(config, output_root=synthetic_raw, tokenizer=None,
                             run_tokenizer_stats=False)

    assert result.counts["deduplicated_exact"] >= 1
    seen: set[str] = set()
    for split in result.splits.values():
        for c in split:
            assert exact_key(c) not in seen
            seen.add(exact_key(c))


def test_prepare_rejects_everything_when_data_is_garbage(tmp_path: Path) -> None:
    from lfm_finetune.utils.errors import DatasetError

    raw = tmp_path / "raw"
    raw.mkdir()
    write_jsonl(raw / "bad.jsonl", [{"nonsense": 1}, {"also": "bad"}])

    config, _ = load_config(REPO / "configs/smoke.yaml")
    config.dataset.path = raw
    with pytest.raises(DatasetError, match="rejected"):
        prepare_dataset(config, output_root=tmp_path, tokenizer=None,
                        run_tokenizer_stats=False)


def test_load_and_clean_counts_are_consistent(synthetic_raw: Path) -> None:
    config, _ = load_config(REPO / "configs/smoke.yaml")
    config.dataset.path = synthetic_raw / "raw"
    config.dataset.quality.min_assistant_chars = 1
    config.dataset.quality.near_duplicate = False
    config.dataset.quality.remove_prompt_leakage = False

    conversations, counts, rejections, summary, _dup, _near = load_and_clean(
        config, raw_path=synthetic_raw / "raw"
    )
    assert counts["accepted"] == len(conversations)
    assert counts["accepted"] + counts["rejected"] == counts["raw_rows"]
    assert len(rejections) >= counts["rejected"]
    assert sum(v for k, v in summary.items()
               if k not in ("duplicate_exact", "duplicate_near")) >= counts["rejected"]


def test_iter_raw_records_skips_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_jsonl(raw / "data.jsonl", [{"instruction": "i", "output": "o"}])
    (raw / "manifest.json").write_text('{"license": "CC0-1.0"}')
    records = [r for _, r in iter_raw_records(raw)]
    assert len(records) == 1
    assert "instruction" in records[0]


def test_malformed_json_line_raises_with_line_number(tmp_path: Path) -> None:
    from lfm_finetune.utils.errors import DatasetError

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"instruction": "a", "output": "b"}\n{not json}\n')
    with pytest.raises(DatasetError, match="line 2"):
        list(iter_raw_records(bad))


# --------------------------------------------------------------------------- #
# Synthetic fixture contract
# --------------------------------------------------------------------------- #

def test_synthetic_dataset_is_deterministic() -> None:
    from lfm_finetune.data.synthetic import generate_synthetic

    a = generate_synthetic(20, seed=99)
    b = generate_synthetic(20, seed=99)
    assert a == b


def test_synthetic_answers_are_computed_correctly() -> None:
    from lfm_finetune.data.synthetic import generate_synthetic

    for record in generate_synthetic(40, seed=5):
        question = record["messages"][-2]["content"]
        answer = record["messages"][-1]["content"]
        if " + " in question and "What is" in question:
            a, b = question.replace("What is", "").replace("?", "").split("+")
            assert answer == str(int(a.strip()) + int(b.strip()))


def test_synthetic_agriculture_placeholders_are_marked() -> None:
    from lfm_finetune.data.synthetic import placeholder_agriculture_records

    records = placeholder_agriculture_records()
    assert records
    for record in records:
        assert record["metadata"]["placeholder"] is True
        assert record["metadata"]["synthetic"] is True


def test_synthetic_manifest_declares_license(tmp_path: Path) -> None:
    from lfm_finetune.data.synthetic import write_synthetic_dataset

    manifest = write_synthetic_dataset(tmp_path / "raw", num_examples=8, seed=1)
    assert manifest["license"] == "CC0-1.0"
    assert manifest["synthetic"] is True
    assert "SYNTHETIC" in manifest["provenance_notes"].upper()

# --------------------------------------------------------------------------- #
# Near-duplicate detection: the shared-system-prompt trap
# --------------------------------------------------------------------------- #

def test_shared_system_prompt_does_not_create_false_near_duplicates() -> None:
    """Regression test for a real bug.

    Every record in a domain corpus shares the same long system prompt (the MICO
    behaviour contract is ~350 characters). When that text was included in the
    near-duplicate similarity, it contributed ~350 identical character shingles
    to every pair, pushing genuinely distinct Q&A records above the threshold and
    silently deleting them from the corpus.

    Templated-but-distinct records must survive; genuine paraphrases must not.
    """
    system = (
        "You are MICO, an agricultural advisory assistant for smallholder farmers in India. "
        "You name the two or three most plausible causes before suggesting anything, you ask "
        "for the details that would separate them, you state your uncertainty plainly, and "
        "you refer the farmer to a local agriculture extension officer when a problem cannot "
        "be identified or is spreading quickly."
    )
    answers = [
        "Rings with a paler centre on the oldest leaves point two ways: a fungal leaf problem "
        "that usually follows wet weather, or a potassium shortfall on older foliage.",
        "Dry tips during tillering here show up for three reasons: saline soil, a potassium "
        "shortage, or wind scorch from the afternoon blast.",
        "Pale new growth usually means nitrogen or sulphur is short, and occasionally iron when "
        "the soil is alkaline, so the position of the pale leaves matters.",
    ]
    queries = ["my tomato plants have brown rings", "rice leaf tips drying", "sugarcane pale leaves"]

    index = NearDuplicateIndex(threshold=0.85)
    for query, answer in zip(queries, answers, strict=True):
        record = Conversation(messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ])
        assert index.query(conversation_text(record))[0] is False, (
            "distinct records sharing a system prompt must not be called near-duplicates"
        )


def test_real_paraphrase_still_caught_despite_shared_system_prompt() -> None:
    """The fix must not blind the detector: a restated answer is still a duplicate."""
    system = "You are MICO, an agricultural advisory assistant for smallholder farmers in India. " * 6
    base_answer = (
        "Rings with a paler centre on the oldest leaves point two ways: a fungal leaf problem "
        "that usually follows wet weather, or a potassium shortfall on older foliage."
    )
    paraphrase = base_answer.replace("usually", "generally").replace("oldest", "older")

    index = NearDuplicateIndex(threshold=0.85)
    first = Conversation(messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": "my tomato plants have brown rings"},
        {"role": "assistant", "content": base_answer},
    ])
    second = Conversation(messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": "my tomato plants have brown rings"},
        {"role": "assistant", "content": paraphrase},
    ])
    assert index.query(conversation_text(first))[0] is False
    assert index.query(conversation_text(second))[0] is True


def test_conversation_text_can_include_system_when_wanted() -> None:
    c = Conversation(messages=[
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ])
    assert "SYS" not in conversation_text(c)
    assert "SYS" in conversation_text(c, include_system=True)
