# Script index

Run commands from the repository root after installing the locked environment:

```bash
uv sync --extra eval --extra baselines --extra test --frozen
source .venv/bin/activate
```

All entry-point scripts live directly in this directory. CANS-specific scripts use the
`cans_` prefix; shared and baseline scripts use descriptive names.

Tests live in `tests/`; run `pytest -q` from the repository root.

Every entry point accepts `--help`. Model-dependent commands accept `--model`
for a local copy of the configured model. Write run outputs under `outputs/`.

## Script entry points

| Script | Purpose |
| --- | --- |
| `prepare_release.py` | Verify all CANS and baseline files, spans, and grouped splits; restore missing inputs; optionally prepare evaluators |
| `cans_generate_reference_corpus.py` | Sample and freeze WikiText reference continuations |
| `extract.py` | Extract Grad/Fisher statistics, SN-Tune/DAP rankings, AurA scales, CAA vectors, or H-Neurons features |
| `cans_compute_neuron_scales.py` | Solve CANS or Grad-only neuron scales |
| `generate_responses.py` | Generate benchmark responses for CANS, Grad-only, all baselines, or the unmodified model |
| `evaluate_responses.py` | Score generated responses with the matching frozen benchmark inputs |

## Extraction

Use `extract.py --method` with one of `grad`, `fisher`, `sn_tune`, `dap`,
`aura`, `aura_jigsaw`, `caa`, or `h_neurons`.

```bash
python scripts/extract.py --method grad \
  --config configs/safety.yaml --output outputs/safety_gradient.pt
python scripts/extract.py --method fisher --output outputs/fisher.pt
python scripts/extract.py --method caa \
  --config configs/baselines/qwen3_safety.yaml --output outputs/safety_caa.pt
```

Grad and Fisher retain FP32 model computation, global/all-token alpha coverage,
and `--resume`. Baseline methods retain their recorded extraction precision and
settings. DAP requires `--guide` pointing to its matching IA3-DPO artifact.
Extraction writes the same artifact schemas consumed by scaling, training,
selection, and generation.

## Generation

`generate_responses.py` is the only benchmark-generation entry point:

- Pass `--scales` for CANS or Grad-only. These use vLLM.
- Pass a baseline `--config` and `--artifact` for SFT, SN-Tune, DPO, DAP, AurA,
  Jigsaw-AurA, CAA, or H-Neurons. These use Transformers cached decoding.
- Omit both intervention arguments for the base model. A baseline config uses
  Transformers; a model or CANS config uses vLLM.

`--scales` and `--artifact` are mutually exclusive. DAP additionally requires
`--guide`. `--strength` controls SN-Tune/CAA strength or the H-Neurons multiplier;
`--layer` and `--candidate` select unfinalized CAA/H-Neurons candidates.
`--input` accepts frozen selection inputs; omitting it uses the complete benchmark.
All generation uses BF16 and the original per-benchmark sampling settings.

## Evaluation

BBH scoring automatically reads the model from the response file's sibling
`.metadata.json`, written by `generate_responses.py`. Qwen3 uses the pinned
`bbh_cot_zeroshot` `flexible-extract` filter and case-insensitive matching;
Llama uses `bbh_cot_fewshot` `get-answer` and case-sensitive exact matching.
The scoring summary records the selected protocol and filter. For responses
without model metadata, pass `--bbh-protocol qwen3` or `--bbh-protocol llama`:

```bash
python scripts/evaluate_responses.py --benchmark bbh --bbh-protocol llama \
  --benchmark-data /path/to/llama/benchmarks \
  --responses outputs/llama_bbh.jsonl --output outputs/llama_bbh_summary.json
```

Supply the matching model's frozen benchmark inputs. Llama BBH inputs must
contain the paper's three worked demonstrations; choosing a scoring protocol
does not change the prompts used to generate existing responses.

## Training and selection

| Script | Supported operations |
| --- | --- |
| `train_sft_sn_tune_dpo.py` | LoRA SFT, SN-Tune, and the IA3-DPO guide for DAP |
| `select_caa_layer_h_neurons.py` | CAA layer selection and H-Neurons classifier fitting/selection |

These entry points call the top-level [`baselines/`](../baselines/README.md)
package. The shared training implementation is
`baselines/sft_sn_tune_dpo_training.py`.

SN-Tune extracts coordinates before training. DAP trains its DPO guide before
extracting a ranking. AurA proceeds from extraction to generation. CAA and
H-Neurons use the recorded domain-selection inputs. See the
[baseline reproduction guide](../baselines/README.md) for each method's commands.

## Release verification and evaluator setup

```bash
python scripts/prepare_release.py --verify-only
python scripts/prepare_release.py --evaluators
```

The verification command checks both manifests, exact H-Neurons answer spans,
prompt-grouped train/held-out splits, and selection-record hashes. Use
`--output outputs/release_audit.json` to save its JSON report. To restore a
missing manifest-listed input, omit `--verify-only` and supply `--base-url`.
The evaluator option prepares the pinned evaluator source and language resources
under `external/`.
