# Frozen Qwen3-4B samples

This directory contains the released Qwen3-4B inputs. Paths in the YAML
configurations are relative to the repository root. Llama configurations expect
separate model-specific inputs, which are not included in this release.

## CANS corpora

| File | Rows | Contents |
| --- | ---: | --- |
| `safety.jsonl` | 256 | Safe responses and exact first-refusal-cue spans |
| `toxicity.jsonl` | 500 | Toxic responses and reviewed exact toxic spans |
| `hallucination.jsonl` | 108 | Truthful responses and reviewed short exact answer spans |
| `reference.jsonl` | 4,096 | Four frozen WikiText continuations for each of 1,024 prompts, with token IDs |

Domain rows retain the complete `prompt` and `response`, a non-empty exact
`span`, and character offsets `span_start` and `span_end`. The span identifies
the target; it does not truncate the complete response or global alpha coverage.

## Baseline corpora

Each domain has a directory under `baselines/`.

| File | Safety rows | Toxicity rows | Hallucination rows |
| --- | ---: | ---: | ---: |
| `pairs.jsonl` | 256 | 500 | 108 |
| `caa_pairs.jsonl` | 256 | 500 | 108 |
| `sft.jsonl` | 256 | 500 | 108 |
| `sn_tune.jsonl` | 256 | 500 | 108 |
| `sn_identification.jsonl` | 200 | 200 | 108 |
| `dap_identification.jsonl` | 200 | 200 | 200 |
| `aura.jsonl` | 1,280 | 2,500 | 216 |
| `h_neurons.jsonl` | 512 | 1,000 | 216 |
| `caa_selection.jsonl` | 200 | 100 | 100 |

Safety DPO and CAA retain identical character-matched response pairs; CAA
keeps raw prompts, and DPO stores the Qwen3 chat prefix. Both responses in
each pair are hard-truncated to the shorter Unicode character length.

Pair files contain `prompt`, `chosen`, and `rejected`. Single-response corpora
contain `prompt` and `response`; class-based methods also retain labels.
H-Neurons rows include exact spans, complete input token IDs, target token
ranges, and other feature regions. `h_split.json` records training and held-out
row IDs; `heldout_prompts.jsonl` contains the corresponding unique held-out
prompts. `provenance/` preserves source and review records and frozen selections.

Toxicity additionally includes `aura_jigsaw.jsonl` with 2,000 clean and 500 toxic
comments, and `provenance/master_pool.jsonl` with the on-policy candidate pool.
See [baseline reproduction](../baselines/README.md) for the method-specific input
formats and supervision settings.

## Benchmark inputs

| File under `benchmarks/` | Prompts |
| --- | ---: |
| `harmbench.jsonl` | 400 |
| `rtp.jsonl` | 600 |
| `truthfulqa.jsonl` | 400 |
| `ifeval.jsonl` | 541 |
| `bbh.jsonl` | 540 |
| `mmlu.jsonl` | 570 |

Benchmark rows retain frozen IDs, formatted prompts, and benchmark-specific
scoring fields. `ifeval_scorer.zip` bundles the official IFEval scorer and its
license; the setup script extracts it into `external/`.

## Integrity and provenance

`manifest.json` records checksums for CANS corpora, benchmark inputs, the
bundled scorer, and released tensors, along with source identities and evaluator
revisions. `baselines/manifest.json` records baseline input checksums and sources.
Historical source paths identify provenance; execution reads the released paths
specified in the configurations.

Verify from the repository root:

```bash
python scripts/prepare_release.py --verify-only
```

Datasets and bundled third-party components retain their original licenses.
See [attribution](../README.md#attribution) and the
[baseline sources](../baselines/README.md).
