"""Check prompt selection and token preservation for reference generation."""

from pathlib import Path
from runpy import run_path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


select_prompts = run_path(str(Path(__file__).resolve().parents[2]
                              / "scripts/cans_generate_reference_corpus.py"))["select_prompts"]


@pytest.fixture
def tokenizer():
    vocabulary = {"<unk>": 0, "<bos>": 1, **{str(i): i + 2 for i in range(20)}}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<bos>")


def settings(**overrides):
    return {"num_prompts": 4, "selection_pool_size": 5, "prompt_tokens": 2,
            "selection_seed": 42, "add_bos": False, **overrides}


def test_selection_filters_rows_and_keeps_the_same_prefix(tokenizer):
    texts = ["", " = Heading = ", "0"] + [f"{i} {i + 1} {i + 2}" for i in range(8)]
    complete = select_prompts(texts, tokenizer, settings())
    short = select_prompts(texts, tokenizer, settings(num_prompts=2))
    assert short == complete[:2]
    assert len({row["source_row"] for row in complete}) == 4
    for row in complete:
        assert row["source_row"] >= 3
        expected = tokenizer.encode(texts[row["source_row"]], add_special_tokens=False)[:2]
        assert row["prompt_token_ids"] == expected
        assert row["prompt"] == tokenizer.decode(expected)


def test_bos_is_explicit_and_preserves_context_tokens(tokenizer):
    texts = [f"{i} {i + 1}" for i in range(8)]
    raw = select_prompts(texts, tokenizer, settings())
    prefixed = select_prompts(texts, tokenizer, settings(add_bos=True))
    for before, after in zip(raw, prefixed, strict=True):
        assert after["source_row"] == before["source_row"]
        assert after["prompt"] == before["prompt"]
        assert after["prompt_token_ids"] == [tokenizer.bos_token_id] + before["prompt_token_ids"]
    tokenizer.bos_token = None
    with pytest.raises(ValueError, match="no BOS"):
        select_prompts(texts, tokenizer, settings(add_bos=True))


def test_selection_rejects_insufficient_source_data(tokenizer):
    with pytest.raises(ValueError, match="eligible rows"):
        select_prompts(["1 2", "= Heading =", ""], tokenizer, settings())
    with pytest.raises(ValueError, match="num_prompts"):
        select_prompts([], tokenizer, settings(num_prompts=6))
