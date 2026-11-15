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
| `cans_evaluate_approximation.py` | Evaluate diagonal CANS updates with the full-Fisher quadratic and measured KL; JSON/CSV only |
| `generate_responses.py` | Generate benchmark responses for CANS, Grad-only, all baselines, or the unmodified model |
| `evaluate_responses.py` | Score generated responses and aggregate five complete capability benchmarks |

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

### Extraction identity and resume

Target-gradient extraction adds BOS only when
it is absent from the formatted prompt; native Llama-3/3.1 chat templates
already contain it. Resume requires matching data, settings and tokenization.
When solving coefficients with `cans_compute_neuron_scales.py`, omit
`--gradient` and `--fisher` to reuse the included statistics. Pass the resulting
coefficients to `generate_responses.py`.
Fisher inputs must bind the configured model and exact reference file, including
its SHA-256, continuation count, extraction scope, precision and coordinates.
Changing `reference_data` requires Fisher statistics extracted from that reference;
a compatible tensor shape alone does not establish compatibility.

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

Shared generation and scoring settings are centralized in
[`configs/evaluation.yaml`](../configs/evaluation.yaml). All model configurations
inherit its runtime, benchmark limits, sampling settings and judge rules.
MBPP and MGSM use the model-specific prompting and context policies described
in this guide; the other released benchmarks retain the shared 4,096-token context.

### BBH

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

### IFEval

For IFEval responses without generation metadata, supply the source model:

```bash
python scripts/evaluate_responses.py --benchmark ifeval \
  --model meta-llama/Llama-2-7b-hf --benchmark-data /path/to/llama2/benchmarks \
  --responses outputs/llama2_ifeval.jsonl --output outputs/llama2_ifeval_summary.json
```

For Llama-2 and its derived checkpoints, IFEval scoring uses only the first
response before a literal `### Instruction:` delimiter. The same preprocessing
is used for strict and loose scoring; original generated text remains stored.
Other models retain their existing IFEval scoring protocol. The scoring CLI
reads the model identity from generation metadata. For external saved responses
without that metadata, provide `--model` with a recognized model ID or a local
checkpoint containing `config.json`. Missing or conflicting model identity is
rejected before scoring; `--model` does not change the saved generation.

### TruthfulQA judges

TruthfulQA's primary `score` is **True×Info (%)**: the fraction of responses
accepted by both the official AllenAI truth and information judges. It is a
same-response conjunction, not the product of the two aggregate percentages.
Higher is better. The summary also reports `true_percent`, `info_percent` and
`true_info_failure_percent = 100 - true_info_percent`. Use `--judge-model` for
a local truth judge and `--info-judge-model` for a local information judge.
Both judges evaluate the same extracted current-question answer interval.
Official remote judges load the revisions pinned in `configs/evaluation.yaml`.
For custom remote checkpoints, `--judge-revision` and `--info-judge-revision`
select explicit versions. Local judge overrides use local files and record their
resolved paths and configuration hashes instead of applying a remote revision.

### MBPP and MGSM protocols

Both benchmarks use `generate_responses.py` and `evaluate_responses.py`.
MBPP uses all 500 test
problems (IDs 11–510), one greedy completion, and 512 response tokens. Native
chat uses demonstrations 2/3/4 and three visible assertions. Llama-2 uses raw
prompts, demonstration 2, and one visible assertion. Scoring executes the original
setup and all three assertions, with no code repair or best-of selection.

MGSM uses the same seed-42 selection of 50 aligned questions in each of 11
languages. Native chat uses six ordered official native-language CoT examples,
2,048 response tokens, and EOS/token-budget stopping. Llama-2 uses raw prompts
within its native 4,096 context, with te=1, bn=3, th=4 and six examples for other
languages; all official native next-question labels also stop generation. Only
Telugu may reduce the response budget to the remaining context. Prompts and
examples are never truncated. Numeric scoring extracts the last signed number,
normalizes Unicode digits, and removes grouping commas.

Both use greedy BF16 decoding and seed 112. Their context capacity is 8,192
for instruction-tuned models and 4,096 for Llama-2. Aggregate capability requires
complete IFEval-541, BBH-540, MMLU-570, MBPP-500 and MGSM-550 summaries from
the same model and intervention; partial sets do not receive an average.

The [main README](../README.md#mbpp-mgsm-and-aggregate-capability) provides
generation, scoring, and five-benchmark aggregation commands. Frozen dataset
sources, revisions, and licenses are recorded in
[`capability_sources.json`](../samples/benchmarks/capability_sources.json).

## Full-Fisher approximation diagnostic

The script solves diagonal-Fisher CANS updates, evaluates the undamped full
sample-Fisher quadratic along each update, and compares it with measured
full-vocabulary `KL(base || edited)` on the frozen reference continuations.
It retains all cross-neuron terms by accumulating `(sequence_score @ delta)^2`,
without constructing a dense Fisher matrix. Damping is used for selecting the
update and its budget, and is excluded from the KL prediction.

The two-model epsilon presets live in `configs/evaluation.yaml` under
`approximation.epsilon_presets`. They match the ten epsilon values per model
in the paper's toxicity approximation figure. Damping is computed as the median over all coordinates of the supplied Fisher.

Bundled data, statistics, and coefficients target Qwen3 only. The Llama-2
epsilon preset is a configuration example, not a ready-to-run numerical
reproduction package. To use `--config configs/llama2_toxicity.yaml`, supply
your own model-matched `--gradient`, `--fisher`, and `--reference` inputs. The full reference
contains the configured 1,024 prompts and four frozen continuations per prompt.
Score extraction uses FP32; actual KL uses BF16 model weights and FP64
probabilities and reduction. Response-token KL is summed per continuation,
then averaged over continuations. `--resume` continues a matching run.
The output contains JSON/CSV diagnostics and approximation errors; no figure
is generated. An optional `--score-index` binds existing FP32 score chunks to
the model, reference tokens and extraction manifest; omit it to extract scores.
See `--help` for separate stages and per-point KL jobs.

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
