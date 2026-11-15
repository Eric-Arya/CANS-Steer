"""Config and corpus I/O, target-token alignment, batching, and tensor storage."""

import hashlib
import json
from pathlib import Path

import torch
import yaml


TARGET_TOKENIZATION_POLICY = "target-boundary-prefix-plus-suffix_bos-if-absent_v2"


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_fisher_binding(curvature, common, reference, artifact_path=None):
    """Bind whole-response Fisher statistics to the actual model/reference input.

    Release artifacts carry embedded metadata. Research extraction artifacts may
    carry the same identities in a sibling ``metadata.json`` instead.
    """
    metadata = curvature.get("metadata")
    metadata_path = None
    if not isinstance(metadata, dict):
        metadata_path = Path(artifact_path).with_name("metadata.json") if artifact_path else None
        if metadata_path is None or not metadata_path.is_file():
            raise ValueError("Fisher requires embedded or sibling extraction metadata")
        metadata = json.loads(metadata_path.read_text())
        recorded_hash = metadata.get("artifact_sha256")
        if recorded_hash is not None and recorded_hash != sha256(artifact_path):
            raise ValueError("Fisher extraction metadata belongs to another artifact")
    if metadata.get("model_config_sha256") != common["model_config_sha256"]:
        raise ValueError("Fisher belongs to another model")
    if metadata.get("estimator") != "whole-response-logprob-gradient-square":
        raise ValueError("Expected matching whole-response Fisher")
    scope = metadata.get("scope", metadata.get("extraction_scope"))
    if scope != "global/all-token":
        raise ValueError("Fisher requires global/all-token extraction")
    model_dtype = metadata.get("model_dtype", metadata.get("model_computation_dtype"))
    if model_dtype != "float32":
        raise ValueError("Fisher requires FP32 model computation")
    if metadata.get("accumulator_dtype", "float64") != "float64":
        raise ValueError("Fisher requires FP64 accumulation")
    revision = metadata.get("revision")
    if revision is not None and common.get("revision") is not None and revision != common["revision"]:
        raise ValueError("Fisher belongs to another model revision")

    reference = Path(reference)
    reference_hash = sha256(reference)
    hashes = [metadata[key] for key in ("data_sha256", "continuations_sha256") if key in metadata]
    if not hashes or any(value != reference_hash for value in hashes):
        raise ValueError("Fisher belongs to another reference corpus")
    count = len(read_jsonl(reference))
    expected = common.get("reference_generation", {})
    configured_count = expected.get("num_prompts", 1) * expected.get("continuations_per_prompt", count)
    counts = [metadata[key] for key in ("count", "continuation_count") if key in metadata]
    if not counts or any(value != count for value in counts) or count != configured_count:
        raise ValueError("Fisher/reference continuation count differs from configuration")

    shape = tuple(common["shape"])
    if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("Expected a positive [layers, intermediate_size] model shape")
    tensor = curvature.get("fisher")
    if not isinstance(tensor, torch.Tensor) or tensor.dtype not in (torch.float32, torch.float64):
        raise ValueError("Fisher must be an FP32 or FP64 tensor")
    if metadata.get("shape") is not None and tuple(metadata["shape"]) != shape:
        raise ValueError("Fisher metadata shape differs from model coordinates")
    flat_layer_major = (tensor.ndim == 1 and tensor.numel() == shape[0]*shape[1]
                        and str(metadata.get("coordinate_order", "")).startswith("layer-major"))
    if tuple(tensor.shape) != shape and not flat_layer_major:
        raise ValueError("Fisher tensor shape differs from model coordinates")
    if not tensor.isfinite().all() or (tensor < 0).any():
        raise ValueError("Fisher must be finite and nonnegative")
    companion = curvature.get("fisher_fp64")
    if companion is not None:
        if (not isinstance(companion, torch.Tensor) or companion.dtype != torch.float64
                or companion.shape != tensor.shape or not companion.isfinite().all()
                or (companion < 0).any() or not torch.equal(companion.float(), tensor.float())):
            raise ValueError("Invalid or inconsistent FP64 Fisher companion")
    return dict(model_config_sha256=common["model_config_sha256"], reference_sha256=reference_hash,
                count=count, shape=list(shape), scope=scope, estimator=metadata["estimator"], model_dtype=model_dtype,
                metadata_sha256=sha256(metadata_path) if metadata_path else None)


def config(path):
    """Load a model or domain config relative to its project directory."""
    path = Path(path).resolve()
    root = path.parent.parent
    domain = yaml.safe_load(path.read_text())
    if "model" in domain:
        return root, model_config(path), {}
    model_path = path.parent / domain.get("model_config", "model.yaml")
    common = model_config(model_path)
    return root, common, domain


def evaluation_config(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1]/"configs/evaluation.yaml"
    return yaml.safe_load(path.read_text())


def model_config(path):
    path = Path(path)
    common = yaml.safe_load(path.read_text())
    shared = evaluation_config(path.parent/common["evaluation_config"] if common.get("evaluation_config") else None)
    common.update(seed=shared["runtime"]["seed"], max_model_len=shared["runtime"]["max_model_len"],
                  evaluation=shared["tasks"], evaluation_runtime=shared["runtime"],
                  evaluation_protocol=shared)
    return common


def save_tensor(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def encode_span(tokenizer, row):
    """Tokenize the complete response and select tokens overlapping its exact span."""
    response, span = row["response"], row["span"]
    start, end = row["span_start"], row["span_end"]
    if not (0 <= start < end <= len(response)) or response[start:end] != span:
        raise ValueError(f"Invalid exact response span: {row['id']}")
    if row["format"] == "chat":
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
    elif row["format"] == "raw":
        prefix = row["prompt"]
    else:
        raise ValueError("format must be raw or chat")
    # Causality makes tokens after the target gradient-neutral. Keep them in the
    # input so alpha coverage always remains the complete non-padding sequence.
    text = prefix + response[:end]
    bos = getattr(tokenizer, "bos_token", None)
    encoded = tokenizer(text, add_special_tokens=not bool(bos and text.startswith(bos)),
                        return_offsets_mapping=True)
    # Preserve the target-boundary tokenization used by the frozen extraction.
    # Append the unscored suffix separately; it cannot change earlier logits.
    suffix_ids = tokenizer.encode(response[end:], add_special_tokens=False)
    start, end = len(prefix) + start, len(prefix) + end
    targets = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
               if b > a and b > start and a < end]
    if not targets or targets[0] == 0:
        raise ValueError(f"No predictable target tokens: {row['id']}")
    return {"id": row["id"], "input_ids": encoded["input_ids"] + suffix_ids, "target_indices": targets}


def padded_batch(rows, pad_id, device):
    """Left-pad complete sequences; prediction masks refer to shifted targets."""
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    target_mask = torch.zeros((len(rows), width - 1), dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        padding = width - len(row["input_ids"])
        ids[i, padding:] = torch.tensor(row["input_ids"], device=device)
        attention[i, padding:] = 1
        for target in row["target_indices"]:
            if not 0 < target < len(row["input_ids"]):
                raise ValueError("Target index is outside the predictable sequence")
            target_mask[i, padding + target - 1] = True
    return ids, attention, target_mask
