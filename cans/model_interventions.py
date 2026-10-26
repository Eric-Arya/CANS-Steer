"""Model loading and global MLP-intermediate neuron interventions."""

from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_processing import sha256


def validate_model_source(name, expected_config_sha256):
    """Reject a local override whose config differs from the released model."""
    path = Path(name)
    if path.is_dir() and sha256(path / "config.json") != expected_config_sha256:
        raise ValueError("Local model config does not match the released base model")


def load_model(name, *, device="cuda:0", extraction=False, revision=None):
    """Use FP32 for statistics and BF16 for evaluation."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, torch_dtype=torch.float32 if extraction else torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.requires_grad_(False)
    model.config.use_cache = not extraction
    return model, tokenizer


def down_projections(model):
    """Return the gated MLP output projections in layer order."""
    try:
        return [layer.mlp.down_proj for layer in model.model.layers]
    except AttributeError as exc:
        raise ValueError("Expected a Llama/Qwen-style gated MLP") from exc


@contextmanager
def neuron_alphas(model, alpha, attention_mask):
    """Give each example its own alpha, shared over all non-padding tokens."""
    projections = down_projections(model)
    expected = (attention_mask.shape[0], len(projections), projections[0].in_features)
    if tuple(alpha.shape) != expected:
        raise ValueError(f"Expected alpha shape {expected}, got {tuple(alpha.shape)}")
    handles = []
    try:
        for index, module in enumerate(projections):
            def hook(_module, inputs, layer=index):
                scale = torch.where(attention_mask.bool().unsqueeze(-1),
                                    alpha[:, layer, :].unsqueeze(1), 1.0)
                return (inputs[0] * scale, *inputs[1:])
            handles.append(module.register_forward_pre_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def fold_scales(model, scales):
    """Fold one multiplier into each down-projection column, exactly once."""
    if getattr(model, "_cans_folded", False):
        raise ValueError("Reload the base model before applying another intervention")
    projections = down_projections(model)
    if tuple(scales.shape) != (len(projections), projections[0].in_features):
        raise ValueError("Scale tensor does not match the model's neuron coordinates")
    if not scales.isfinite().all():
        raise ValueError("Scale tensor contains non-finite values")
    with torch.no_grad():
        for module, scale in zip(projections, scales):
            module.weight.mul_(scale.to(module.weight).unsqueeze(0))
    model._cans_folded = True


def fold_vllm_scales(model, scales):
    """Apply the same column operation inside a single-GPU vLLM worker."""
    if getattr(model, "_cans_folded", False):
        raise ValueError("The vLLM model already has a CANS intervention")
    modules = dict(model.named_modules())
    projections = [m for n, m in modules.items() if n.endswith(".mlp.down_proj")]
    if (not projections or tuple(scales.shape) != (len(projections), projections[0].weight.shape[1])
            or not scales.isfinite().all()):
        raise ValueError("Expected finite scales for every model neuron")
    for layer, scale in enumerate(scales):
        matches = [m for n, m in modules.items() if n.endswith(f"layers.{layer}.mlp.down_proj")]
        if len(matches) != 1 or matches[0].weight.shape[1] != len(scale):
            raise ValueError("Expected an unquantized, single-GPU down projection")
        weight = matches[0].weight
        if weight.dtype != torch.bfloat16:
            raise ValueError("Evaluation weights must use BF16")
        with torch.no_grad():
            weight.mul_(scale.to(weight).unsqueeze(0))
    model._cans_folded = True
