"""Check mathematical invariants and the released reproduction artifacts."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
import yaml

from cans.data_processing import config as load_config, padded_batch, read_jsonl, sha256
from cans.gradient_fisher_extraction import extract_statistics, response_gradients
from cans.model_interventions import fold_scales, neuron_alphas
from cans.scaling_solver import cans_scales, grad_only_scales


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=("qwen", "llama"))
def model(request):
    torch.manual_seed(19)
    settings = dict(vocab_size=41, hidden_size=16, intermediate_size=24,
                    num_hidden_layers=2, num_attention_heads=2,
                    num_key_value_heads=1, attention_dropout=0.0)
    if request.param == "qwen":
        model_config = Qwen3Config(**settings, head_dim=8)
        model_class = Qwen3ForCausalLM
    else:
        model_config = LlamaConfig(**settings)
        model_class = LlamaForCausalLM
    model_config._attn_implementation = "eager"
    model = model_class(model_config).eval()
    model.requires_grad_(False)
    return model


def test_custom_model_configuration(tmp_path):
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / "another_model.yaml").write_text("model: example/another-model\n")
    domain = folder / "domain.yaml"
    domain.write_text("model_config: another_model.yaml\ndata: data/domain.jsonl\n")
    root, model_config, domain_config = load_config(domain)
    assert root == tmp_path
    assert model_config["model"] == "example/another-model"
    assert domain_config["data"] == "data/domain.jsonl"
    assert load_config(folder / "another_model.yaml")[1] == model_config


def test_global_alpha_matches_weight_folding(model):
    ids = torch.tensor([[1, 5, 7, 3], [0, 0, 2, 4]])
    mask = (ids != 0).long()
    scale = torch.linspace(0.6, 1.4, 48).reshape(2, 24)
    folded = deepcopy(model)
    fold_scales(folded, scale)
    with neuron_alphas(model, scale.expand(2, -1, -1), mask):
        expected = model(ids, attention_mask=mask).logits
    actual = folded(ids, attention_mask=mask).logits
    torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="Reload"):
        fold_scales(folded, scale)


def test_independent_gradients_and_finite_difference(model):
    rows = [{"input_ids": [1, 3, 4, 5], "target_indices": [2, 3]},
            {"input_ids": [2, 6, 9], "target_indices": [2]}]
    together = response_gradients(model, rows, 0)
    separate = torch.cat([response_gradients(model, [row], 0) for row in rows])
    torch.testing.assert_close(together, separate, atol=1e-7, rtol=1e-4)
    ids, mask, _ = padded_batch(rows[:1], 0, "cpu")
    alpha = torch.ones(1, 2, 24)
    scores = []
    for delta in (-0.01, 0.01):
        alpha[0, 0, 3] = 1 + delta
        with neuron_alphas(model, alpha, mask):
            logits = model(ids, attention_mask=mask).logits.log_softmax(-1)
            scores.append(sum(logits[0, i - 1, ids[0, i]] for i in [2, 3]))
    numerical = (scores[1] - scores[0]) / 0.02
    torch.testing.assert_close(together[0, 0, 3], numerical, atol=6e-5, rtol=0.03)


def test_fisher_squares_after_response_sum(model):
    row = {"input_ids": [1, 4, 3, 9], "target_indices": [2, 3]}
    whole = response_gradients(model, [row], 0)
    token_gradients = [response_gradients(model, [{**row, "target_indices": [i]}], 0)
                       for i in row["target_indices"]]
    torch.testing.assert_close(whole, sum(token_gradients), atol=1e-7, rtol=1e-4)
    fisher = extract_statistics(model, [row], 0, fisher=True)
    torch.testing.assert_close(fisher, whole[0].double().square())
    assert not torch.allclose(fisher, sum(g[0].double().square() for g in token_gradients),
                              atol=1e-10, rtol=1e-3)


def test_zero_response_and_resume_binding(model, tmp_path):
    rows = [{"input_ids": [1, 2], "target_indices": []},
            {"input_ids": [1, 2, 3], "target_indices": [2]}]
    checkpoint = tmp_path / "checkpoint.pt"
    value = extract_statistics(model, rows, 0, fisher=True, batch_size=2,
                               checkpoint=checkpoint, binding={"data": "A"})
    expected = response_gradients(model, rows[1:], 0)[0].double().square() / 2
    torch.testing.assert_close(value, expected, atol=1e-9, rtol=1e-4)
    with pytest.raises(ValueError, match="different inputs"):
        extract_statistics(model, rows, 0, fisher=True, checkpoint=checkpoint,
                           binding={"data": "B"}, resume=True)


def test_closed_form_optimality_and_clipping():
    g = torch.tensor([[1.0, -2.0, 0.3]], dtype=torch.float64)
    f = torch.tensor([[0.2, 0.7, 1.0]], dtype=torch.float64)
    scales, info = cans_scales(g, f, 0.1, damping=0.3)
    metric = f + 0.3
    assert info["pre_cap_cost"] == pytest.approx(0.1)
    optimum = float((g * (scales - 1)).sum())
    torch.manual_seed(5)
    for _ in range(20):
        direction = torch.randn_like(g)
        direction *= (0.2 / (metric * direction.square()).sum()).sqrt()
        assert float((g * direction).sum()) <= optimum + 1e-10
    capped, info = cans_scales(g, f, 0.1, damping=0.3, cap=0.05)
    assert (capped - 1).abs().max() <= 0.05 + 1e-12
    assert info["post_cap_cost"] <= info["pre_cap_cost"]
    base, _ = cans_scales(torch.zeros_like(g), f, 0.1, damping=0.3)
    assert torch.equal(base, torch.ones_like(g))


def test_grad_only_ranking():
    g = torch.tensor([[0.1, -3.0, 0.4, 0.4]])
    result = grad_only_scales(g, 1, 0.5)
    assert result.tolist() == [[1, 1, 0.5, 1]]
    with pytest.raises(ValueError):
        grad_only_scales(g, 4, 0)


def test_release_hashes_and_spans():
    manifest = json.loads((ROOT / "samples/manifest.json").read_text())
    for item in manifest["files"]:
        assert sha256(ROOT / item["path"]) == item["sha256"], item["path"]
    for domain, count in [("safety", 256), ("toxicity", 500), ("hallucination", 108)]:
        rows = read_jsonl(ROOT / f"samples/{domain}.jsonl")
        assert len(rows) == count
        assert len({row["id"] for row in rows}) == count
        for row in rows:
            assert row["response"][row["span_start"]:row["span_end"]] == row["span"]
    reference = read_jsonl(ROOT / "samples/reference.jsonl")
    assert len(reference) == 4096
    assert len({row["prompt_index"] for row in reference}) == 1024
    assert all(len(row["prompt_token_ids"]) == 128 for row in reference)


def test_all_released_scales_match_solver():
    f = torch.load(ROOT / "cans/artifacts/fisher.pt", weights_only=True)["fisher"]
    for domain in ("safety", "toxicity", "hallucination"):
        config = yaml.safe_load((ROOT / f"configs/{domain}.yaml").read_text())
        artifact = torch.load(ROOT / config["artifact"], weights_only=True)
        scales, info = cans_scales(config["objective_sign"] * artifact["gradient"], f, **config["cans"])
        torch.testing.assert_close(scales, artifact["scales"], atol=1e-12, rtol=1e-12)
        assert info["damping"] == float(f.double().median())
        assert info["pre_cap_cost"] == pytest.approx(config["cans"]["epsilon"])
        assert info["capped_fraction"] <= 0.15
