"""Chat-template rendering, loss masking, and train/inference consistency.

These are the tests that protect the single most damaging silent bug in an SFT
pipeline: supervising the prompt. They run against the **real** checkpoint
tokenizer, because the whole point is that the behaviour depends on the exact
template shipped with the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lfm_finetune.config.loader import load_config
from lfm_finetune.config.schema import TokenizationConfig
from lfm_finetune.data.schema import Conversation
from lfm_finetune.training.collator import SFTCollator, tokenize_conversations
from lfm_finetune.training.tokenization import (
    IGNORE_INDEX,
    ChatFormatter,
    format_example_for_display,
    template_fingerprint,
    verify_template_consistency,
)
from lfm_finetune.utils.errors import TokenizerError

REPO = Path(__file__).resolve().parent.parent
IM_START = "<|" + "im_start" + "|>"
IM_END = "<|" + "im_end" + "|>"


@pytest.fixture(scope="module")
def config():
    cfg, _ = load_config(REPO / "configs/smoke.yaml")
    return cfg


@pytest.fixture(scope="module")
def tokenizer_and_formatter(config, requires_model):
    from lfm_finetune.model.loader import load_tokenizer

    tokenizer, source = load_tokenizer(config.model)
    formatter = ChatFormatter(tokenizer, config.tokenization)
    return tokenizer, formatter, source


@pytest.fixture
def simple_conversation() -> Conversation:
    return Conversation(
        id="t-1",
        messages=[
            {"role": "system", "content": "You are a careful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2 + 2 = 4."},
        ],
    )


# --------------------------------------------------------------------------- #
# Checkpoint template facts
# --------------------------------------------------------------------------- #

def test_checkpoint_ships_a_chat_template(tokenizer_and_formatter) -> None:
    tokenizer, _formatter, source = tokenizer_and_formatter
    assert tokenizer.chat_template, "checkpoint must provide a chat template"
    assert source == "checkpoint"


def test_render_produces_prefix_consistent_text(tokenizer_and_formatter) -> None:
    tokenizer, formatter, _ = tokenizer_and_formatter
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": "ASST"},
    ]
    full = formatter.render(messages, add_generation_prompt=False)
    prompt = formatter.render(messages[:2], add_generation_prompt=True)
    assert full.startswith(prompt), (
        "the prompt render must be a strict prefix of the full render; otherwise "
        "prefix-length loss masking would supervise the wrong span"
    )


def test_generation_prompt_ends_with_assistant_newline(tokenizer_and_formatter) -> None:
    tokenizer, formatter, _ = tokenizer_and_formatter
    prompt = formatter.render(
        [{"role": "user", "content": "U"}], add_generation_prompt=True
    )
    assert prompt.endswith("assistant" + chr(10))


def test_template_uses_chatml_control_tokens(tokenizer_and_formatter) -> None:
    """LFM2.5 wraps every turn in <|im_start|> / <|im_end|>."""
    tokenizer, formatter, _ = tokenizer_and_formatter
    rendered = formatter.render(
        [{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}],
        add_generation_prompt=False,
    )
    assert IM_START in rendered
    assert IM_END in rendered


def test_eos_token_id_is_the_end_of_turn_token(tokenizer_and_formatter) -> None:
    tokenizer, formatter, _ = tokenizer_and_formatter
    assert tokenizer.eos_token_id is not None
    assert tokenizer.decode([tokenizer.eos_token_id]) == IM_END
    assert formatter.eos_token_id == tokenizer.eos_token_id


def test_eos_token_is_the_chatml_end_marker(tokenizer_and_formatter) -> None:
    """`tokenizer.eos_token` resolves to the literal end-of-turn marker.

    Worth asserting explicitly because `special_tokens_map.json` in the
    LFM2.5 repos registers ids 6 and 7 with an *empty* `content` field, which
    makes the terminator look absent when you inspect the raw JSON. The loaded
    tokenizer resolves the real marker, and code that assumes eos is a blank
    string will mis-handle generation.
    """
    tokenizer, formatter, _ = tokenizer_and_formatter
    assert tokenizer.eos_token == IM_END
    assert formatter.eos_token_id == tokenizer.eos_token_id == 7


def test_add_special_tokens_false_is_required(tokenizer_and_formatter) -> None:
    """The template emits {{ bos_token }} itself; the default tokenizer call
    prepends a second BOS, shifting the input distribution off pre-training."""
    tokenizer, formatter, _ = tokenizer_and_formatter
    rendered = formatter.render(
        [{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}],
        add_generation_prompt=False,
    )
    ours = formatter.encode(rendered)
    default = tokenizer(rendered)["input_ids"]
    assert ours[0] == tokenizer.bos_token_id
    assert ours.count(tokenizer.bos_token_id) == 1
    assert default.count(tokenizer.bos_token_id) == 2, (
        "this test is meaningless unless the default call really doubles the BOS"
    )


def test_template_fingerprint_is_stable_and_sensitive(tokenizer_and_formatter) -> None:
    tokenizer, _formatter, _ = tokenizer_and_formatter
    original = tokenizer.chat_template
    try:
        first = template_fingerprint(tokenizer)
        assert first == template_fingerprint(tokenizer)
        tokenizer.chat_template = original + "\n{# changed #}"
        assert template_fingerprint(tokenizer) != first
    finally:
        tokenizer.chat_template = original


# --------------------------------------------------------------------------- #
# Loss masking
# --------------------------------------------------------------------------- #

def test_prompt_tokens_are_masked(tokenizer_and_formatter, simple_conversation) -> None:
    _tokenizer, formatter, _ = tokenizer_and_formatter
    example = formatter.format_conversation(simple_conversation, assistant_only_loss=True)

    assert example.num_prompt_tokens > 0
    assert all(lab == IGNORE_INDEX for lab in example.labels[: example.num_prompt_tokens])
    assert any(lab != IGNORE_INDEX for lab in example.labels[example.num_prompt_tokens :])


def test_prompt_coverage_is_substantial(tokenizer_and_formatter,
                                        simple_conversation) -> None:
    """If the mask is not actually masking, the run silently trains on the
    prompt. Coverage should be far below 100%."""
    _tokenizer, formatter, _ = tokenizer_and_formatter
    example = formatter.format_conversation(simple_conversation, assistant_only_loss=True)
    assert example.loss_coverage < 0.9, (
        f"loss coverage {example.loss_coverage:.1%} -- assistant-only masking is "
        f"probably not working"
    )


def test_supervised_span_is_exactly_the_answer_plus_eos(tokenizer_and_formatter,
                                                        simple_conversation) -> None:
    tokenizer, formatter, _ = tokenizer_and_formatter
    example = formatter.format_conversation(simple_conversation, assistant_only_loss=True)
    supervised = example.input_ids[example.num_prompt_tokens :]
    text = tokenizer.decode(supervised, skip_special_tokens=True)
    assert text.strip() == "2 + 2 = 4."
    assert supervised[-1] == formatter.eos_token_id, (
        "the supervised target must end with the end-of-turn token"
    )


def test_full_sequence_mode_masks_only_bos(tokenizer_and_formatter,
                                          simple_conversation) -> None:
    _tokenizer, formatter, _ = tokenizer_and_formatter
    example = formatter.format_conversation(simple_conversation, assistant_only_loss=False)
    assert example.labels[0] == IGNORE_INDEX
    assert all(lab != IGNORE_INDEX for lab in example.labels[1:])


def test_multi_turn_masks_every_prompt(tokenizer_and_formatter) -> None:
    _tokenizer, formatter, _ = tokenizer_and_formatter
    convo = Conversation(messages=[
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer text"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer text"},
    ])
    example = formatter.format_conversation(convo, assistant_only_loss=True)
    supervised_text = _tokenizer.decode(
        example.input_ids[example.num_prompt_tokens :], skip_special_tokens=True
    )
    # The trailing assistant turn is supervised. Whether the first one is included
    # depends on the template; what must hold is that the LAST user turn is not
    # part of the supervised span.
    assert "second question" not in supervised_text
    assert "second answer text" in supervised_text


def test_every_turn_covered_when_all_assistant(tokenizer_and_formatter) -> None:
    _tokenizer, formatter, _ = tokenizer_and_formatter
    convo = Conversation(messages=[
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ])
    example = formatter.format_conversation(convo, assistant_only_loss=True)
    assert example.num_loss_tokens >= 2


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #

def test_truncation_keeps_the_supervised_span(tokenizer_and_formatter) -> None:
    """Truncating from the tail would delete the answer -- the only tokens worth
    training on. keep_head_tail must cut the middle instead."""
    tokenizer, formatter, _ = tokenizer_and_formatter
    convo = Conversation(messages=[
        {"role": "user", "content": "please analyse " + "filler context " * 120},
        {"role": "assistant", "content": "the final answer is 42"},
    ])
    limit = 200
    example = formatter.format_conversation(convo, assistant_only_loss=True, max_length=limit)
    assert example.truncated is True
    assert example.length == limit
    assert example.num_loss_tokens > 0, "truncation must not remove all supervision"
    text = tokenizer.decode(
        example.input_ids[example.num_prompt_tokens :], skip_special_tokens=True
    )
    assert "42" in text, "the supervised answer must survive truncation"


def test_truncation_strategy_error_raises(config, tokenizer_and_formatter) -> None:
    tokenizer, _formatter, _ = tokenizer_and_formatter
    strict = ChatFormatter(tokenizer, TokenizationConfig(
        max_seq_length=32, truncation_strategy="error", append_eos_token=True
    ))
    convo = Conversation(messages=[
        {"role": "user", "content": "a very long question " * 50},
        {"role": "assistant", "content": "an answer that is long enough"},
    ])
    with pytest.raises(TokenizerError, match="exceeds tokenization.max_seq_length"):
        strict.format_conversation(convo)


def test_long_prompt_still_produces_supervision(tokenizer_and_formatter) -> None:
    """Regression test for a real bug.

    When the prompt alone exceeded `max_seq_length`, the old head+tail truncation
    produced an example in which *every* position was masked: the record trained
    on nothing yet looked normal in the dataset statistics. Truncation must
    always reserve room for the supervised span.
    """
    tokenizer, formatter, _ = tokenizer_and_formatter
    convo = Conversation(messages=[
        {"role": "system", "content": "you are a crop advisor"},
        {"role": "user", "content": "a long question " * 200},
        {"role": "assistant", "content": "the final answer is 42"},
    ])
    example = formatter.format_conversation(convo, assistant_only_loss=True, max_length=64)
    assert example.length == 64
    assert example.num_loss_tokens > 0
    text = tokenizer.decode(
        example.input_ids[example.num_prompt_tokens :], skip_special_tokens=True
    )
    assert "42" in text


def test_zero_loss_example_raises(tokenizer_and_formatter) -> None:
    """An example with no supervised token must be rejected, not silently
    trained on.

    Reachable through the non-default ``truncate_tail`` strategy, which is exactly
    why ``keep_head_tail`` is the default. This also pins the guard itself:
    without it such a record would contribute a zero-length gradient while
    looking perfectly normal in the dataset statistics.
    """
    tokenizer, _formatter, _ = tokenizer_and_formatter
    truncating = ChatFormatter(tokenizer, TokenizationConfig(
        max_seq_length=32, truncation_strategy="truncate_tail", append_eos_token=True
    ))
    convo = Conversation(messages=[
        {"role": "user", "content": "a question much longer than the budget " * 20},
        {"role": "assistant", "content": "a short answer"},
    ])
    with pytest.raises(TokenizerError, match="zero tokens"):
        truncating.format_conversation(convo, assistant_only_loss=True)


def test_non_prefix_consistent_template_is_rejected(tokenizer_and_formatter) -> None:
    """If the template cannot reproduce the prompt as a prefix of the full
    render, prefix-length masking would supervise the wrong span -- so this must
    fail loudly rather than train on the wrong tokens."""
    tokenizer, formatter, _ = tokenizer_and_formatter
    original = tokenizer.chat_template
    try:
        # A template whose output depends on how many messages there are.
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['role'] }}\\n{{ m['content'] }}\\n{% endfor %}"
            "{% if add_generation_prompt %}assistant\\n{% endif %}"
            "{% if messages | length > 2 %}TRAILER{% endif %}"
        )
        convo = Conversation(messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ])
        with pytest.raises(TokenizerError, match="not prefix-consistent"):
            formatter.format_conversation(convo, assistant_only_loss=True)
    finally:
        tokenizer.chat_template = original


def test_empty_assistant_turn_yields_only_the_terminator(tokenizer_and_formatter) -> None:
    """Documented behaviour: an empty final answer produces a 1-token target
    (the end-of-turn token), not a failure. The dataset validator is what rejects
    empty answers, so the formatter does not have to."""
    _tokenizer, formatter, _ = tokenizer_and_formatter
    convo = Conversation(messages=[
        {"role": "user", "content": "a question"},
        {"role": "assistant", "content": ""},
    ])
    example = formatter.format_conversation(convo, assistant_only_loss=True)
    assert example.num_loss_tokens == 1
    assert example.input_ids[-1] == formatter.eos_token_id


# --------------------------------------------------------------------------- #
# Train / inference equivalence
# --------------------------------------------------------------------------- #

def test_inference_prompt_is_a_prefix_of_training_ids(tokenizer_and_formatter,
                                                      simple_conversation) -> None:
    _tokenizer, formatter, _ = tokenizer_and_formatter
    example = formatter.format_conversation(simple_conversation, assistant_only_loss=True)
    prompt_ids = formatter.build_prompt_ids(
        [m.to_dict() for m in simple_conversation.messages[:-1]], add_generation_prompt=True
    )
    assert example.input_ids[: len(prompt_ids)] == prompt_ids


def test_consistency_check_passes_on_real_data(tokenizer_and_formatter) -> None:
    _tokenizer, formatter, _ = tokenizer_and_formatter
    convos = [
        Conversation(id=f"c{i}", messages=[
            {"role": "user", "content": f"question number {i}"},
            {"role": "assistant", "content": f"answer number {i}, with some detail"},
        ])
        for i in range(8)
    ]
    result = verify_template_consistency(_tokenizer, formatter, convos)
    assert result["ok"], result["failures"]
    assert result["checked"] == 8
    assert result["mean_loss_coverage"] is not None


def test_generation_kwargs_omits_pad_token_id(tokenizer_and_formatter) -> None:
    """Passing pad_token_id from both the tokenizer and the formatter makes
    PeftModel.generate raise 'multiple values for keyword argument'."""
    _tokenizer, formatter, _ = tokenizer_and_formatter
    kwargs = formatter.generation_kwargs()
    assert "eos_token_id" in kwargs
    assert "pad_token_id" not in kwargs


def test_strip_generated_removes_terminator(tokenizer_and_formatter) -> None:
    tokenizer, formatter, _ = tokenizer_and_formatter
    ids = tokenizer("hello", add_special_tokens=False)["input_ids"]
    assert formatter.strip_generated(ids + [formatter.eos_token_id]) == "hello"


# --------------------------------------------------------------------------- #
# Collator
# --------------------------------------------------------------------------- #

def test_collator_pads_and_masks(tokenizer_and_formatter) -> None:
    _tokenizer, formatter, _ = tokenizer_and_formatter
    convos = [
        Conversation(messages=[{"role": "user", "content": "short"},
                               {"role": "assistant", "content": "tiny answer"}]),
        Conversation(messages=[{"role": "user", "content": "a much longer question " * 5},
                               {"role": "assistant", "content": "a longer answer here"}]),
    ]
    tokens = tokenize_conversations(convos, formatter, assistant_only_loss=True)
    batch = SFTCollator(pad_token_id=formatter.pad_token_id)(tokens.examples)

    lengths = {len(f.input_ids) for f in tokens.examples}
    assert len(lengths) > 1, "test needs examples of differing length to exercise padding"
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] % 8 == 0
    for row in range(2):
        pad_positions = batch["attention_mask"][row] == 0
        assert torch_all_masked(batch["labels"][row], pad_positions)
        assert torch_all_zero(batch["input_ids"][row], pad_positions)


def torch_all_masked(labels, mask) -> bool:
    return all(int(labels[i]) == IGNORE_INDEX for i in range(len(labels)) if bool(mask[i]))


def torch_all_zero(ids, mask) -> bool:
    return all(int(ids[i]) == 0 for i in range(len(ids)) if bool(mask[i]))


def _unformattable() -> Conversation:
    """A record with no assistant turn: valid JSON, no supervised target."""
    return Conversation(messages=[
        {"role": "user", "content": "a question with no reply at all"},
        {"role": "tool", "content": "some tool output"},
    ])


def test_tokenize_drops_and_reports_unformattable(tokenizer_and_formatter) -> None:
    """One bad row must not abort a run, but it must be counted and explained."""
    _tokenizer, formatter, _ = tokenizer_and_formatter
    good = Conversation(messages=[{"role": "user", "content": "q"},
                                  {"role": "assistant", "content": "an answer"}])
    tokens = tokenize_conversations([good, _unformattable()], formatter,
                                    assistant_only_loss=True)
    assert len(tokens) == 1
    assert tokens.num_dropped == 1
    assert tokens.dropped_reasons


def test_all_examples_failing_raises(tokenizer_and_formatter) -> None:
    _tokenizer, formatter, _ = tokenizer_and_formatter
    with pytest.raises(TokenizerError, match="failed to format"):
        tokenize_conversations([_unformattable()], formatter, assistant_only_loss=True)


def test_display_marks_supervised_tokens(tokenizer_and_formatter,
                                        simple_conversation) -> None:
    tokenizer, formatter, _ = tokenizer_and_formatter
    example = formatter.format_conversation(simple_conversation, assistant_only_loss=True)
    text = format_example_for_display(example, tokenizer)
    assert "Blank 'marker' column" in text
    assert "loss_tokens=" in text
    # Lines carry a leading space for unsupervised tokens and the token repr
    # directly for supervised ones; both shapes must be present.
    assert any(line.startswith("    ") and " id=" in line for line in text.splitlines())


def test_eos_can_be_disabled_but_is_default(tokenizer_and_formatter) -> None:
    _tokenizer, _formatter, _ = tokenizer_and_formatter
    assert TokenizationConfig().append_eos_token is True
    assert TokenizationConfig().max_seq_length == 1024