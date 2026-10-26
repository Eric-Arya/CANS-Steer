"""SN-Tune importance and sparse tuning, ported from our experiment implementation."""
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from types import SimpleNamespace
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb, repeat_kv
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb
STRUCTURES = ("fwd_up", "fwd_down", "q", "k", "v")
def ranked_top(values: torch.Tensor, count: int) -> list[int]:
    flat = values.detach().float().reshape(-1)
    count = min(count, flat.numel())
    return torch.argsort(flat, descending=True, stable=True)[:count].cpu().tolist()


class LayerScorer:
    def __init__(
        self,
        layer_index: int,
        attention: torch.nn.Module,
        mlp: torch.nn.Module,
        args,
        model_type: str,
    ):
        self.layer_index = layer_index
        self.attention = attention
        self.mlp = mlp
        self.top_attn = args.top_attn
        self.top_ffn = args.top_ffn
        self.q_chunk = args.q_chunk
        self.model_type = model_type
        self.num_heads = int(
            getattr(attention, "num_heads", attention.config.num_attention_heads)
        )
        self.num_key_value_heads = int(
            getattr(
                attention,
                "num_key_value_heads",
                attention.config.num_key_value_heads,
            )
        )
        self.position_embeddings = None
        self.attention_mask = None
        self.query_states = None
        self.key_states = None
        self.candidates: dict[str, list[int]] = {}
        query_capture = attention.q_norm if model_type == "qwen3" else attention.q_proj
        key_capture = attention.k_norm if model_type == "qwen3" else attention.k_proj
        self.handles = [
            attention.register_forward_pre_hook(self.attention_pre_hook, with_kwargs=True),
            query_capture.register_forward_hook(self.query_hook),
            key_capture.register_forward_hook(self.key_hook),
            attention.o_proj.register_forward_pre_hook(self.output_projection_pre_hook),
            mlp.down_proj.register_forward_pre_hook(self.down_projection_pre_hook),
        ]

    def attention_pre_hook(self, _module, _args, kwargs) -> None:
        self.position_embeddings = kwargs["position_embeddings"]
        self.attention_mask = kwargs.get("attention_mask")
        self.query_states = None
        self.key_states = None
        self.candidates = {}

    def query_hook(self, _module, _inputs, output) -> None:
        if output.ndim == 3:
            output = output.view(
                output.shape[0],
                output.shape[1],
                self.num_heads,
                self.attention.head_dim,
            )
        self.query_states = output

    def key_hook(self, _module, _inputs, output) -> None:
        if output.ndim == 3:
            output = output.view(
                output.shape[0],
                output.shape[1],
                self.num_key_value_heads,
                self.attention.head_dim,
            )
        self.key_states = output

    @torch.inference_mode()
    def output_projection_pre_hook(self, module, inputs) -> None:
        if self.query_states is None or self.key_states is None or self.position_embeddings is None:
            raise RuntimeError(f"Layer {self.layer_index}: incomplete attention capture")
        query = self.query_states.transpose(1, 2)
        key = self.key_states.transpose(1, 2)
        cos, sin = self.position_embeddings
        apply_rotary = (
            qwen3_apply_rotary_pos_emb
            if self.model_type == "qwen3"
            else llama_apply_rotary_pos_emb
        )
        query, key = apply_rotary(query, key, cos, sin)
        key = repeat_kv(key, self.attention.num_key_value_groups)

        logits = torch.matmul(query, key.transpose(2, 3)) * self.attention.scaling
        if self.attention_mask is not None:
            logits = logits + self.attention_mask[:, :, :, : key.shape[-2]]
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)

        scores = []
        for start in range(0, query.shape[-1], self.q_chunk):
            stop = min(query.shape[-1], start + self.q_chunk)
            contribution = (
                query[:, :, :, start:stop].transpose(-2, -1).unsqueeze(-1)
                * key[:, :, :, start:stop].transpose(-2, -1).unsqueeze(-2)
                * self.attention.scaling
            )
            without = torch.softmax(
                logits.unsqueeze(2) - contribution,
                dim=-1,
                dtype=torch.float32,
            ).to(query.dtype)
            score = (without - probabilities.unsqueeze(2)).pow(2).sum(dim=(-2, -1))
            scores.append(score)
        query_score = torch.cat(scores, dim=-1)
        if query_score.numel() != self.attention.q_proj.out_features:
            raise RuntimeError(
                f"Layer {self.layer_index}: Q score width {query_score.numel()} does not "
                f"match q_proj rows {self.attention.q_proj.out_features}"
            )
        query_candidates = ranked_top(query_score, self.top_attn)
        self.candidates["q"] = query_candidates
        # The paper proxy and current Llama detector treat expanded Q/K symmetrically.
        self.candidates["k"] = list(query_candidates)

        pooled_value = inputs[0]
        value_score = torch.linalg.vector_norm(pooled_value.float(), dim=1)
        value_score *= torch.linalg.vector_norm(module.weight.float(), dim=0)
        self.candidates["v"] = ranked_top(value_score, self.top_attn)

    @torch.inference_mode()
    def down_projection_pre_hook(self, module, inputs) -> None:
        intermediate = inputs[0]
        ffn_score = torch.linalg.vector_norm(intermediate.float(), dim=1)
        ffn_score *= torch.linalg.vector_norm(module.weight.float(), dim=0)
        candidates = ranked_top(ffn_score, self.top_ffn)
        self.candidates["fwd_up"] = candidates
        self.candidates["fwd_down"] = list(candidates)

    def result(self) -> dict[str, list[int]]:
        missing = set(STRUCTURES).difference(self.candidates)
        if missing:
            raise RuntimeError(f"Layer {self.layer_index}: missing scores for {sorted(missing)}")
        return {name: self.candidates[name] for name in STRUCTURES}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def aggregate_ranked(per_prompt: list[dict[str, Any]], num_layers: int):
    ranked: dict[str, dict[str, list[int]]] = {name: {} for name in STRUCTURES}
    for name in STRUCTURES:
        for layer in range(num_layers):
            prefixes = [row["layers"][str(layer)][name] for row in per_prompt]
            common = set(prefixes[0]).intersection(*(set(values) for values in prefixes[1:]))
            rank_sum = {
                index: sum(values.index(index) for values in prefixes) for index in common
            }
            ranked[name][str(layer)] = sorted(common, key=lambda index: (rank_sum[index], index))
    return ranked


def expanded_to_physical(index: int, head_dim: int, kv_repeat: int) -> int:
    expanded_head, coordinate = divmod(index, head_dim)
    return (expanded_head // kv_repeat) * head_dim + coordinate


def cap_selection(ranked, cap: int, head_dim: int, kv_repeat: int):
    selected: dict[str, dict[str, list[int]]] = {}
    for name in STRUCTURES:
        selected[name] = {}
        for layer, values in ranked[name].items():
            ordered = list(dict.fromkeys(values))
            if name in {"k", "v"}:
                ordered = list(
                    dict.fromkeys(
                        expanded_to_physical(index, head_dim, kv_repeat) for index in ordered
                    )
                )
            selected[name][layer] = sorted(ordered[:cap])
    return selected


class SparseNeuronLinear(nn.Module):
    """Frozen linear layer plus trainable FP32 row or column deltas."""

    def __init__(self, base: nn.Linear, indices: set[int], axis: str):
        super().__init__()
        if axis not in {"row", "column"}:
            raise ValueError(f"Unsupported sparse axis: {axis}")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.axis = axis
        ordered = torch.tensor(sorted(indices), dtype=torch.long, device=base.weight.device)
        self.register_buffer("indices", ordered, persistent=False)
        shape = (
            (len(indices), base.in_features)
            if axis == "row"
            else (base.out_features, len(indices))
        )
        self.delta = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32, device=base.weight.device),
            requires_grad=bool(indices),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if self.axis == "row":
            delta_output = F.linear(inputs, self.delta.to(inputs.dtype))
            return output.index_add(-1, self.indices, delta_output)
        selected_inputs = inputs.index_select(-1, self.indices)
        return output + F.linear(selected_inputs, self.delta.to(inputs.dtype))

    def merge(self) -> nn.Linear:
        weight = self.base.weight.detach().float()
        delta = self.delta.detach().float()
        if self.axis == "row":
            weight.index_add_(0, self.indices, delta)
        else:
            weight.index_add_(1, self.indices, delta)
        self.base.weight = nn.Parameter(weight, requires_grad=False)
        if self.base.bias is not None:
            self.base.bias = nn.Parameter(self.base.bias.detach().float(), requires_grad=False)
        return self.base


def install_sparse_deltas(model, selected: list[dict[int, list[int]]]) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    total = 0
    for layer_index, layer in enumerate(model.model.layers):
        specs = (
            (layer.mlp, "up_proj", selected[0], "row"),
            (layer.mlp, "down_proj", selected[1], "column"),
            (layer.self_attn, "q_proj", selected[2], "row"),
            (layer.self_attn, "k_proj", selected[3], "row"),
            (layer.self_attn, "v_proj", selected[4], "row"),
        )
        for parent, attribute, values, axis in specs:
            indices = set(values[str(layer_index)])
            wrapped = SparseNeuronLinear(getattr(parent, attribute), indices, axis)
            setattr(parent, attribute, wrapped)
            total += wrapped.delta.numel()
    return total


@torch.inference_mode()
def identify(model, tokenizer, rows, cap=25, top_attn=100, top_ffn=200, q_chunk=128):
    args = SimpleNamespace(top_attn=top_attn, top_ffn=top_ffn, q_chunk=q_chunk)
    scorers = [LayerScorer(i, layer.self_attn, layer.mlp, args, model.config.model_type)
               for i, layer in enumerate(model.model.layers)]
    records = []
    try:
        for row in rows:
            encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=True).to(model.device)
            model(**encoded, use_cache=False)
            records.append({"id": row["id"], "layers": {str(i): s.result() for i, s in enumerate(scorers)}})
    finally:
        for scorer in scorers:
            scorer.close()
    ranked = aggregate_ranked(records, len(scorers))
    cfg = model.config
    selection = cap_selection(ranked, cap, getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
                              cfg.num_attention_heads // cfg.num_key_value_heads)
    return {"selection": selection, "ranked": ranked, "per_prompt": records}


def sparse_state(model):
    return {name: {"axis": module.axis, "indices": module.indices.cpu(), "delta": module.delta.detach().cpu()}
            for name, module in model.named_modules() if isinstance(module, SparseNeuronLinear)}


def apply_sparse_state(model, state, strength=1.):
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, item in state.items():
            module = modules[name]
            dim = 0 if item["axis"] == "row" else 1
            weight = module.weight.float()
            weight.index_add_(dim, item["indices"].to(weight.device), item["delta"].to(weight.device) * strength)
            module.weight.copy_(weight.to(module.weight.dtype))
