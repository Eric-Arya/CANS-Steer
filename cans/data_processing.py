"""Config and corpus I/O, target-token alignment, batching, and tensor storage."""

import hashlib
import json
from pathlib import Path

import torch
import yaml


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


def config(path):
    """Load a model or domain config relative to its project directory."""
    path = Path(path).resolve()
    root = path.parent.parent
    domain = yaml.safe_load(path.read_text())
    if "model" in domain:
        return root, domain, {}
    model_path = path.parent / domain.get("model_config", "model.yaml")
    common = yaml.safe_load(model_path.read_text())
    return root, common, domain


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
    encoded = tokenizer(prefix + response[:end], add_special_tokens=True, return_offsets_mapping=True)
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
