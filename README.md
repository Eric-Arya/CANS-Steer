<h1 align="center">CANS</h1>
<p align="center"><strong>Capability-Preserving Neuron Steering for Large Language Models</strong></p>

[![arXiv](https://img.shields.io/badge/arXiv-2610.35530-b31b1b.svg)](https://arxiv.org/abs/2610.35530)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This is the **official code and data release** for the paper
**[CANS: Capability-Preserving Neuron Steering for Large Language Models](https://arxiv.org/abs/2610.35530)**.

[Paper](https://arxiv.org/abs/2610.35530) · [PDF](https://arxiv.org/pdf/2610.35530) ·
[Installation](#installation) · [Quick start](#quick-start) ·
[Reproduction](#main-pipeline) · [Citation](#citation)

This repository provides code for CANS, a lightweight and generalizable
neuron steering method that balances target-behavior control with capability preservation through
closed-form scaling for all neurons. Its coefficients fold into MLP weights,
adding no parameters or inference-time operations.

We release **all CANS and baselines implementation code** that support multiple domains and models
and **all annotated corpora used for Qwen3-4B needed to reproduce the CANS and baseline results**
across safety, toxicity, and hallucination,
including WikiText as the general reference input. **We welcome reproduction and discussion.**

![Figure 1: Overview of CANS](fig1.png)

## Release contents

- **CANS implementation** for extracting target gradients and Fisher statistics,
  computing neuron steering coefficients, and generating and evaluating responses.
- **Baseline implementations** of LoRA SFT, SN-Tune, DAP, AurA, CAA, H-Neurons,
  and CANS-Grad-Only.
- **Complete Qwen3-4B reproduction inputs and artifacts** across safety, toxicity,
  and hallucination: annotated CANS and baseline corpora, frozen WikiText reference
  continuations, benchmark inputs, provenance manifests, shared Fisher statistics,
  domain gradients, and ready-to-use CANS steering coefficients.

See the [data inventory](samples/README.md) and
[artifact inventory](cans/artifacts/README.md) for file contents and formats.

## Method overview

CANS estimates a target-behavior gradient and a shared whole-response Fisher
diagonal, then solves for neuron scaling coefficients under a capability budget.
The coefficients are folded into the model's MLP weights before generation.

```text
Frozen domain responses ──> Target gradients ──┐
                                             ├──> Neuron scales ──> Generation ──> Evaluation
Frozen WikiText responses ──> Fisher diagonal ─┘
```

Let $g$ be the target log-probability gradient and $h = s g$, where
$s = +1$ for safety/hallucination and $s = -1$ for toxicity. With Fisher diagonal
$F$ and damping $\lambda$, CANS uses $D = \mathrm{diag}(F + \lambda)$:

$$
\delta = \sqrt{\frac{2\epsilon}{h^\top D^{-1}h}}\,D^{-1}h,
\qquad \mathrm{scale} = 1 + \delta.
$$

Both the direction and the quadratic budget $\tfrac12\delta^\top D\delta$
use the damped metric. For a zero gradient, the update is zero. Damping defaults
to the median over all Fisher coordinates, including zeros.
**CANS-Grad-Only** uses $\mathrm{scale} = 1 + s c g$ across all coordinates,
retaining gradient signs without Fisher division, damping, ranking, or
normalization. Current presets apply both methods without clipping
(`cap: null`). Parameters are defined in the domain YAMLs under `configs/`.

## Repository layout

| Location | Contents |
| --- | --- |
| [`scripts/`](scripts/README.md) | Shared extraction, generation, evaluation, training, and selection; CANS-specific scripts use the `cans_` prefix |
| [`cans/`](cans/) | Reusable CANS implementation |
| [`baselines/`](baselines/README.md) | Baseline method implementations and shared training |
| [`configs/`](configs/) | Model, domain, extraction, training, and evaluation settings |
| [`samples/`](samples/README.md) | Frozen Qwen3 corpora, benchmark inputs, and provenance manifests |
| [`cans/artifacts/`](cans/artifacts/README.md) | Released Fisher statistics, target gradients, and example coefficients |
| [`scripts/tests/`](scripts/tests/) | Method invariants and release verification |

See the [script index](scripts/README.md) for entry points and dependencies.
Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

## Installation

Use Linux, Python 3.10 or 3.11, Git, and an NVIDIA GPU with BF16 support.

```bash
git clone https://github.com/Eric-Arya/CANS-Steer.git
cd CANS-Steer
```

With [uv](https://docs.astral.sh/uv/) installed, run from the repository root:

```bash
uv sync --extra eval --extra baselines --extra test --frozen
source .venv/bin/activate
python scripts/prepare_release.py --verify-only
pytest -q
```

The example settings target an 80 GB GPU. Select the model in
[`configs/model.yaml`](configs/model.yaml), or pass `--model /path/to/model`
for a matching local copy.

## Quick start

Generate and evaluate safety responses using the included CANS coefficients:

```bash
python scripts/prepare_release.py --evaluators

CUDA_VISIBLE_DEVICES=0 python scripts/generate_responses.py \
  --benchmark harmbench --scales cans/artifacts/safety.pt \
  --output outputs/safety_cans.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_responses.py \
  --benchmark harmbench --responses outputs/safety_cans.jsonl \
  --output outputs/safety_summary.json
```

| Domain | Benchmark | Coefficients |
| --- | --- | --- |
| Safety | `harmbench` | `cans/artifacts/safety.pt` |
| Toxicity | `rtp` | `cans/artifacts/toxicity.pt` |
| Hallucination | `truthfulqa` | `cans/artifacts/hallucination.pt` |

Both scripts also accept `ifeval`, `bbh`, `mmlu`, `mbpp`, and `mgsm`.
Omit `--scales` for the base model; use `--judge-model` for a local classifier.

## Evaluation metrics

The table summarizes the frozen evaluation sets and primary percentage scores
configured in [`configs/evaluation.yaml`](configs/evaluation.yaml).

| Benchmark | Evaluation size | Primary metric | Better |
| --- | --- | --- | --- |
| HarmBench | 400 prompts | Attack success rate (ASR) | Lower |
| RealToxicityPrompts (`rtp`) | 600 prompts × 25 completions | Prompts with at least one toxic completion (%) | Lower |
| TruthfulQA | 400 questions | True×Info: answers accepted by both judges (%) | Higher |
| IFEval | 541 prompts | Strict prompt accuracy (%) | Higher |
| BBH | 540 questions | Answer accuracy (%) | Higher |
| MMLU | 570 questions | Constrained A–D accuracy (%) | Higher |
| MBPP | 500 problems | Pass@1 (%) | Higher |
| MGSM | 550 questions, 50 per language | Macro-average accuracy across 11 languages (%) | Higher |

**Capability** is the equal-weight mean of IFEval, BBH, MMLU, MBPP, and MGSM.
Aggregation requires all five complete results from the same model and
intervention. TruthfulQA True×Info counts answers accepted by both judges on the
same response; it is not the product of the two marginal percentages.

Generation/scoring settings, model-specific prompts, stopping rules, and judge
versions are documented in the [evaluation protocol](scripts/README.md#evaluation)
and loaded from the shared YAML.

## Main pipeline

The released corpora are ready to use. To generate a new reference corpus, see
[other models and reference data](#other-models-and-reference-data). Otherwise,
start with extraction, then solve the scales and run the quick-start evaluation:

```bash
# Shared Fisher
CUDA_VISIBLE_DEVICES=0 python scripts/extract.py --method fisher \
  --output outputs/fisher.pt

# Domain gradients
CUDA_VISIBLE_DEVICES=0 python scripts/extract.py --method grad \
  --config configs/safety.yaml --output outputs/safety_gradient.pt

# CANS coefficients
python scripts/cans_compute_neuron_scales.py \
  --config configs/safety.yaml \
  --gradient outputs/safety_gradient.pt --fisher outputs/fisher.pt \
  --output outputs/safety_cans.pt
```

Replace `safety` with `toxicity` or `hallucination` as needed. Omit `--gradient`
and `--fisher` to reuse the included statistics. Extraction uses FP32 and
supports `--resume`; evaluation uses BF16. See
[extraction identity and resume](scripts/README.md#extraction-identity-and-resume)
for BOS handling and model/reference compatibility checks.

The domain YAMLs record the strongest-domain epsilon presets and use no
coefficient clipping (`cap: null`). Damping is computed from the median over
all coordinates of the model-matched WikiText Fisher. Artifacts record the resolved damping and their source statistics.

For **CANS-Grad-Only**, use the coefficient recorded in the domain YAML, or
pass `--coefficient` to evaluate another strength:

```bash
python scripts/cans_compute_neuron_scales.py \
  --config configs/safety.yaml --method grad-only \
  --output outputs/safety_grad_only.pt
```

## MBPP, MGSM, and aggregate capability

```bash
python scripts/generate_responses.py --benchmark mbpp \
  --scales cans/artifacts/safety.pt --output outputs/safety_mbpp.jsonl
python scripts/evaluate_responses.py --benchmark mbpp \
  --responses outputs/safety_mbpp.jsonl --output outputs/safety_mbpp_summary.json
```

Replace `mbpp` with `mgsm` for multilingual math. Model-specific demonstrations,
response budgets, stopping rules, and scoring are described in the
[MBPP and MGSM protocols](scripts/README.md#mbpp-and-mgsm-protocols).
After generating and scoring all five capability benchmarks, aggregate them:

```bash
python scripts/evaluate_responses.py --aggregate \
  outputs/safety_ifeval_summary.json outputs/safety_bbh_summary.json \
  outputs/safety_mmlu_summary.json outputs/safety_mbpp_summary.json \
  outputs/safety_mgsm_summary.json --output outputs/safety_capability.json
```

## Full-Fisher approximation diagnostic

```bash
python scripts/cans_evaluate_approximation.py --config configs/toxicity.yaml \
  --output outputs/toxicity_approximation
```

This script compares the undamped full sample-Fisher quadratic with measured
`KL(base || edited)` along diagonal-Fisher CANS updates. It uses the paper's
ten-point epsilon preset for the configured model and writes JSON/CSV results.
See the [diagnostic protocol](scripts/README.md#full-fisher-approximation-diagnostic)
for reference inputs, precision, caching, and resume options.

## Baselines

We also release our implementations of **LoRA SFT, SN-Tune, DAP (with its
IA3-DPO guide), AurA, CAA, H-Neurons, and CANS-Grad-Only**, together with
**Qwen3-4B baseline corpora** for all three domains. This includes contrastive
pairs, exact answer spans, frozen classifier splits and identification prompts,
the selected toxicity examples and the Jigsaw AurA corpus.

```bash
uv sync --extra eval --extra baselines --extra test --frozen
python scripts/prepare_release.py --verify-only
```

See [baseline reproduction instructions](baselines/README.md) for training,
neuron identification, selection, generation, and evaluation. Configurations
cover the six model/domain settings; Llama data is supplied separately by the
user. All baseline evaluation uses BF16. Baseline code lives in
`baselines/`, configurations in `configs/baselines/`, and Qwen3 data in
`samples/baselines/`.

## Other models and reference data

Compatible Qwen and Llama models with `mlp.down_proj` are supported on one GPU
with unquantized weights. Copy `configs/` into a new project and update the model
ID, revision, config hash, shape, and data paths. Provide corpora and benchmark
prompts for that model using the included JSONL schema, then pass the new
configuration with `--config`.

To generate reference continuations from the WikiText-2-raw-v1
[training Parquet file](https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/cans_generate_reference_corpus.py \
  --config configs/model.yaml \
  --dataset /path/to/train-00000-of-00001.parquet \
  --output outputs/reference.jsonl
```

Configure sampling and BOS in `reference_generation`; set `reference_data` to
the generated JSONL before estimating Fisher. Paths in YAML files are relative
to the project directory containing `configs/`. The generator saves exact token
IDs and sampling metadata. For custom benchmark inputs, set `benchmark_data`
in the model config and pass that directory as `--benchmark-data` when scoring.

## Data and reproducibility

The release contains Qwen3-4B corpora for all three domains and frozen WikiText
continuations as the general reference input, with response token IDs. Baseline
inputs include paired responses, reviewed exact spans, identification prompts, frozen classifier
splits, selection manifests, selected toxicity response pairs, and Jigsaw AurA data.
See the [sample inventory](samples/README.md) for counts and file schemas.

`samples/manifest.json` and `samples/baselines/manifest.json` record file hashes
and source provenance. The verification command above checks all released
files, including baseline exact spans and grouped splits.
Grad and Fisher extraction use FP32 with global/all-token alpha coverage.
Model evaluation uses BF16. Model revisions, generation settings, and actual
method settings are recorded in the supplied configurations and artifacts.

## Citation

If you use CANS or the released code and data in your research, please cite:

```bibtex
@article{wu2026cans,
  title   = {{CANS}: Capability-Preserving Neuron Steering for Large Language Models},
  author  = {Wu, Yucheng and Xie, Chengyi and Wang, Chenpeng and Pan, Liangming},
  journal = {arXiv preprint arXiv:2610.35530},
  year    = {2026},
  doi     = {10.48550/arXiv.2610.35530},
  url     = {https://arxiv.org/abs/2610.35530}
}
```

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

## Questions and contributions

We welcome reproduction reports, questions, and contributions through
[GitHub Issues](https://github.com/Eric-Arya/CANS-Steer/issues) and pull requests.
For reproduction issues, include the model revision, configuration, command,
environment details, and relevant logs or evaluation outputs.

## Attribution

CANS code uses the MIT License. Datasets, model weights, and third-party
evaluators retain their original licenses. The bundled IFEval scorer preserves
its Apache-2.0 license. Sources:
[Circuit Breakers](https://github.com/GraySwanAI/circuit-breakers),
[RealToxicityPrompts](https://allenai.org/data/real-toxicity-prompts),
[TruthfulQA](https://github.com/sylinrl/TruthfulQA),
[WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
[HarmBench](https://github.com/centerforaisafety/HarmBench),
[IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval),
[MBPP](https://github.com/google-research/google-research/tree/master/mbpp),
[MGSM](https://github.com/google-research/url-nlp/tree/main/mgsm),
and [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

MBPP and MGSM source revisions, evaluation splits, and bundled license/attribution
files are listed in [`capability_sources.json`](samples/benchmarks/capability_sources.json).
