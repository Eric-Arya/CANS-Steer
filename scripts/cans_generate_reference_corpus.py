"""Sample raw WikiText continuations and freeze their token IDs for Fisher."""

import argparse
from hashlib import sha256 as hash_bytes
from importlib.metadata import version
import json
import os
from pathlib import Path
import random

import torch
from transformers import AutoTokenizer

from cans.data_processing import config, sha256, write_jsonl
from cans.model_interventions import validate_model_source


def select_prompts(texts, tokenizer, settings):
    """Take a fixed prefix of a seeded sample of eligible WikiText rows."""
    count, pool_size = settings["num_prompts"], settings["selection_pool_size"]
    width = settings["prompt_tokens"]
    if not 0 < count <= pool_size or width < 1:
        raise ValueError("Require 0 < num_prompts <= selection_pool_size and prompt_tokens > 0")
    if not isinstance(settings["add_bos"], bool):
        raise ValueError("add_bos must be true or false")
    if settings["add_bos"] and tokenizer.bos_token_id is None:
        raise ValueError("The tokenizer has no BOS token; set add_bos to false")
    eligible = []
    for source_row, text in enumerate(texts):
        text = (text or "").strip()
        if not text or (text.startswith("=") and text.endswith("=")):
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= width:
            eligible.append((source_row, tokens[:width]))
    if len(eligible) < pool_size:
        raise ValueError(f"Need {pool_size} eligible rows, found {len(eligible)}")
    # Keep the draw size fixed so shorter runs use the same source prompts.
    selected = random.Random(settings["selection_seed"]).sample(eligible, pool_size)[:count]
    prefix = [tokenizer.bos_token_id] if settings["add_bos"] else []
    return [{"prompt_index": index, "source_row": source_row,
             "prompt": tokenizer.decode(tokens, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False),
             "prompt_token_ids": prefix + tokens}
            for index, (source_row, tokens) in enumerate(selected)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--model", help="Local copy of the model selected by the config")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="WikiText train Parquet file with a text column")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, help="Override the configured prompt count")
    parser.add_argument("--batch-size", type=int, help="Override the generation batch size")
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(".metadata.json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError(f"Use a fresh output path: {args.output}")
    _, common, _ = config(args.config)
    settings = dict(common["reference_generation"])
    if args.num_prompts is not None:
        settings["num_prompts"] = args.num_prompts
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if any(settings[key] < 1 for key in ("batch_size", "continuations_per_prompt", "max_new_tokens")):
        raise ValueError("Batch size, continuation count, and response length must be positive")
    model = args.model or common["model"]
    validate_model_source(model, common["model_config_sha256"])
    tokenizer = AutoTokenizer.from_pretrained(model, revision=common.get("revision"))
    import pyarrow.parquet as pq

    texts = pq.read_table(args.dataset, columns=["text"]).column("text").to_pylist()
    prompts = select_prompts(texts, tokenizer, settings)
    if max(len(row["prompt_token_ids"]) for row in prompts) + settings["max_new_tokens"] > common["max_model_len"]:
        raise ValueError("Prompt and response lengths exceed max_model_len")
    os.environ["VLLM_USE_V1"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from vllm import LLM, SamplingParams

    # Reference sampling uses FP32, as in the frozen corpus; evaluation uses BF16.
    llm = LLM(model=model, revision=common.get("revision"), dtype="float32",
              tensor_parallel_size=1, enforce_eager=True,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              max_model_len=common["max_model_len"], max_num_seqs=settings["batch_size"],
              gpu_memory_utilization=0.8, seed=settings["generation_seed"])
    results = []
    continuations = settings["continuations_per_prompt"]
    # Group requests by continuation, preserving the released sampling order.
    for continuation_index in range(continuations):
        for start in range(0, len(prompts), settings["batch_size"]):
            batch = prompts[start:start + settings["batch_size"]]
            ids = [row["prompt_index"] * continuations + continuation_index for row in batch]
            seeds = [settings["generation_seed"] + index for index in ids]
            parameters = [SamplingParams(temperature=1.0, top_p=1.0, top_k=-1,
                          max_tokens=settings["max_new_tokens"], seed=seed) for seed in seeds]
            outputs = llm.generate([{"prompt_token_ids": row["prompt_token_ids"]} for row in batch],
                                   parameters, use_tqdm=False)
            for row, index, seed, output in zip(batch, ids, seeds, outputs, strict=True):
                response_ids = list(output.outputs[0].token_ids)
                # Keep every sampled token, including EOS; never re-tokenize the text.
                results.append({**row, "id": index, "continuation_index": continuation_index,
                                "response_token_ids": response_ids,
                                "response": tokenizer.decode(response_ids, skip_special_tokens=True,
                                                             clean_up_tokenization_spaces=False),
                                "generation_seed": seed})
            print(f"Generated {len(results)}/{len(prompts) * continuations} continuations", flush=True)
    results.sort(key=lambda row: row["id"])
    write_jsonl(args.output, results)
    metadata = {
        "model": common["model"], "revision": common.get("revision"),
        "model_config_sha256": common["model_config_sha256"],
        "tokenizer_sha256": hash_bytes(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "dataset": str(args.dataset), "dataset_sha256": sha256(args.dataset),
        "settings": settings, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
        "seed_rule": "generation_seed + prompt_index * continuations_per_prompt + continuation_index",
        "format": "raw", "dtype": "float32", "backend": "vllm-v0-eager",
        "count": len(results), "response_tokens": sum(len(row["response_token_ids"]) for row in results),
        "output_sha256": sha256(args.output),
        "packages": {name: version(name) for name in ("torch", "transformers", "vllm", "pyarrow")},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
