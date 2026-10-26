# Baseline implementations

The release includes SFT (LoRA), SN-Tune, DAP with its MLP-only IA3-DPO
guide, AurA, CAA, H-Neurons, and CANS-Grad-Only. The implementation supports
Llama and Qwen3 decoder models. Frozen baseline data is provided for Qwen3-4B
only; the Llama configurations require the corresponding model-specific inputs.

See the paper appendix for details on data consistency across methods and
fairness guarantees.

## Source layout

This package contains the baseline implementations for Llama and Qwen3.
Extraction, generation, training, and selection entry points are in
[`scripts/`](../scripts/README.md).
Settings are in [`configs/baselines/`](../configs/baselines/), and frozen Qwen3 inputs
in [`samples/baselines/`](../samples/baselines/).

| Module | Responsibility |
| --- | --- |
| `sft_sn_tune_dpo_training.py` | LoRA SFT, sparse SN-Tune, and MLP-only IA3-DPO training; AdamW checkpoints and resumption |
| `sn_tune.py` | SN-Tune neuron identification, sparse parameter updates, and intervention |
| `dap.py` | DAP activation-contrast ranking, IA3 guide parameters, and activation patching |
| `aura.py` | AurA activation extraction, AUROC scaling, and intervention |
| `caa.py` | CAA vector extraction and intervention |
| `h_neurons.py` | H-Neurons feature extraction, grouped splits, classifier fitting, and adaptive lambda selection |
| `common.py` | Baseline configuration, tokenization, response log probabilities, and provenance |

The training module shares optimizer, batching, and checkpoint logic across the
three training methods. Its `sft_rows`, `sn_rows`, and `preference_rows` functions
prepare their respective training inputs; `train` selects the configured method.
The implementation reuses model and data utilities from [`cans/`](../cans/).

## Install and verify

```bash
uv sync --extra eval --extra baselines --extra test --frozen
source .venv/bin/activate
python scripts/prepare_release.py --verify-only
pytest -q
```

Configuration files are in `configs/baselines/`. Paths in those files are
relative to the repository root. Use `--model /path/to/model` to use a local
copy matching the model config hash. No neighboring research repositories,
modified Transformers installation, or private filesystem paths are required.

## Included Qwen3 data

| Input | Safety | Toxicity | Hallucination |
| --- | ---: | ---: | ---: |
| DPO / CAA pairs | 256 | 500 | 108 |
| SFT / SN-Tune desired responses | 256 | 500 | 108 |
| H-Neurons desired + undesired responses | 256 + 256 | 500 + 500 | 108 + 108 |
| AurA desired + undesired responses | 1,024 + 256 | 2,000 + 500 | 173 + 43 |
| SN-Tune identification prompts | 200 | 200 | 108 |
| DAP identification prompts | 200 | 200 | 200 |
| CAA / H-Neurons domain selection prompts | 200 | 100 | 100 |

Toxicity also includes the Jigsaw AurA corpus (2,000 clean + 500 toxic),
and the raw on-policy master pool with all candidate completions and classifier
scores. `samples/baselines/manifest.json` records source identities and hashes.
Exact H-Neurons spans, complete response token IDs, feature regions, frozen
classifier splits, and saved selection records are included.

Safety DPO and CAA use character-matched response pairs: both responses are
hard-truncated to the shorter Python string length, without stripping or
normalization. The Qwen3 safety DPO prompts are already chat-formatted;
`prompt_is_formatted: true` preserves their stored prefix without applying
another chat template. Safety SFT supervises the first refusal cue; toxicity
and hallucination SFT supervise the complete desired response. SN-Tune uses
its own desired-response corpus. Raw SN-Tune training preserves the experiment's
`prompt + ". " + response` serialization, full-sequence loss, and 512-token cap.
Hallucination SN-Tune uses native chat. Qwen3 thinking is disabled.

## Train SFT, SN-Tune, and the DAP guide

```bash
# LoRA SFT: rank 8, alpha 16, all attention/MLP projections, learning rate 5e-5.
python scripts/train_sft_sn_tune_dpo.py --config configs/baselines/qwen3_safety.yaml \
  --method sft --until-epoch 3 --output outputs/safety_sft

# Resume restores the trainable parameters, AdamW moments, scheduler, and RNG.
python scripts/train_sft_sn_tune_dpo.py --config configs/baselines/qwen3_safety.yaml \
  --method sft --resume outputs/safety_sft/epoch_3 --until-epoch 6 \
  --output outputs/safety_sft

# SN-Tune: identify coordinates, then train only the corresponding matrix slices.
python scripts/extract.py --config configs/baselines/qwen3_safety.yaml \
  --method sn_tune --output outputs/safety_sn_coordinates.pt
python scripts/train_sft_sn_tune_dpo.py --config configs/baselines/qwen3_safety.yaml \
  --method sn_tune --selection outputs/safety_sn_coordinates.pt \
  --output outputs/safety_sn

# MLP-only IA3-DPO: beta 0.1, learning rate 1e-3, effective batch 64, 3 epochs.
python scripts/train_sft_sn_tune_dpo.py --config configs/baselines/qwen3_safety.yaml \
  --method dpo --output outputs/safety_dpo
python scripts/extract.py --config configs/baselines/qwen3_safety.yaml \
  --method dap --guide outputs/safety_dpo/epoch_3/intervention.pt \
  --output outputs/safety_dap.pt
```

SFT checkpoints are saved every three epochs. For a new stopping-criterion
run, evaluate each checkpoint on the complete domain and capability benchmarks,
then continue only while average capability has not fallen below the matching
CANS point and the undesirable-behavior rate has not fallen below that point.
The supplied Qwen3 configurations end at the reported epochs: 30 for safety
and toxicity, 15 for hallucination. Each checkpoint includes the AdamW state
in `training_state.pt`; its epoch and step are recorded in `metadata.json`.

SN-Tune retains at most 25 coordinates per layer and parameter structure,
after mapping GQA coordinates to physical rows. It uses 20 epochs, learning
rate 1e-6, cosine scheduling, warmup ratio 0.05, and gradient clipping 0.3.
The intervention strengths 1, 2, 3, and 4 multiply the learned weight change.

DAP ranks RMS activation changes over the same base-generated continuation
tokens in both streams. Generation follows the decode/split/re-tokenize
procedure used in the experiment. Exactly the top 5% of eligible FFN
coordinates are selected, excluding the final layer. The guide advances its
own KV cache; activations are patched after IA3 scaling during prefill and
decoding. Use the exact guide artifact that produced the ranking.

## AurA and CAA

```bash
python scripts/extract.py --config configs/baselines/qwen3_toxicity.yaml \
  --method aura --output outputs/toxicity_aura.pt
python scripts/extract.py --config configs/baselines/qwen3_toxicity.yaml \
  --method aura_jigsaw --output outputs/toxicity_aura_jigsaw.pt
python scripts/extract.py --config configs/baselines/qwen3_safety.yaml \
  --method caa --output outputs/safety_caa_vectors.pt
```

AurA max-pools each gate/up/down projection output coordinate and computes
its AUROC against response labels. AUROC above 0.5 gives multiplier
`2 - 2 * AUROC`; other coordinates retain multiplier 1. Safety uses the first
128 response tokens, while keeping the complete input. Toxicity and Jigsaw
use standalone raw completion/comment text, truncated to 128 tokens.
Hallucination uses its complete canonically serialized response. AurA
multipliers apply to all token positions.

CAA uses balanced A/B assignments and domain-specific choice templates.
Candidate layers span 30–50% of model depth in steps of two. Evaluate each
candidate at strength 1 on `caa_selection.jsonl`, and select the layer with
minimum undesirable-behavior rate. The selected Qwen3 layers are 17, 15,
and 11 for safety, toxicity, and hallucination. Final strengths are
0.5, 1, 1.5, and 2. The vector is added at the final prompt token and each
cached decoding step.

## H-Neurons

```bash
python scripts/extract.py --config configs/baselines/qwen3_safety.yaml \
  --method h_neurons --output outputs/safety_h_features.pt
python scripts/select_caa_layer_h_neurons.py --config configs/baselines/qwen3_safety.yaml \
  --method h_neurons --artifact outputs/safety_h_features.pt --scan \
  --output outputs/safety_h_candidates.pt
```

To refit the recorded Qwen3 setting directly on its frozen training split:

```bash
python scripts/select_caa_layer_h_neurons.py --config configs/baselines/qwen3_safety.yaml \
  --method h_neurons --artifact outputs/safety_h_features.pt --fit-selected \
  --output outputs/safety_h_selected.pt
```

CETT features are the mean absolute neuron contribution over each exact
answer span, divided by the FFN output norm. The classifier uses L1 logistic
regression and the 3:1 negative:positive training-feature expansion. Frozen
seed-42 splits are source-prompt grouped and label-balanced. Held-out
classifier accuracy uses target features only.

Lambda adapts to observed positive-weight neuron counts, retaining four
candidates approximately spanning 0.01–0.35 per mille of eligible neurons.
Evaluate their multiplier-0 interventions on HarmBench-200, RTP-100, or
TruthfulQA-100. Select jointly using held-out classifier accuracy and domain
performance; general-capability scores do not participate. Qwen3 selected
counts are 29, 105, and 127. Final multipliers are 0, 0.25, 0.5, and 0.75.
Llama selected counts are 193, 125, and 68; there is no fixed K=9 exception.

Selection commands consume a JSON file with this schema; `domain_score`
is the undesirable-behavior percentage (lower is better):

```json
{
  "selection_data_sha256": "SHA256 of the configured caa_selection.jsonl",
  "candidates": [{"candidate": 0, "domain_score": 16.5}]
}
```

Include all candidate evaluations. For H-Neurons, `candidate` is the zero-based
scan index. For CAA, it is the layer index. Record a reason addressing both
classifier accuracy and domain performance for H-Neurons:

```bash
python scripts/select_caa_layer_h_neurons.py --config configs/baselines/qwen3_safety.yaml \
  --method h_neurons --artifact outputs/safety_h_candidates.pt \
  --candidate 0 --domain-scores outputs/safety_h_domain_scores.json \
  --reason "Record the reviewed classifier and domain measurements here" \
  --output outputs/safety_h_selected.pt
```

## Generate and score

```bash
python scripts/generate_responses.py \
  --config configs/baselines/qwen3_safety.yaml --benchmark harmbench \
  --artifact outputs/safety_sn/epoch_20/intervention.pt --strength 2 \
  --output outputs/safety_sn_strength2.jsonl
python scripts/evaluate_responses.py --benchmark harmbench \
  --responses outputs/safety_sn_strength2.jsonl \
  --output outputs/safety_sn_strength2_summary.json
```

The shared `scripts/generate_responses.py` entry point uses Transformers
cached decoding for baseline artifacts and vLLM for CANS/Grad-only scales.
Both use BF16 weights, the same frozen formatted prompts, and shared evaluators. Saved metadata identifies the backend, configuration,
input hash, and intervention hash. RTP uses 25 samples per prompt; other
benchmarks use greedy decoding. Final evaluations use complete datasets.

For DAP, pass both `--artifact outputs/safety_dap.pt` and the matching
`--guide outputs/safety_dpo/epoch_3/intervention.pt`. For CAA candidates,
pass `--layer`; for an H-Neurons scan artifact, pass `--candidate`. `--strength`
is a CAA/SN-Tune strength or an H-Neurons activation multiplier.

For domain selection only, pass
`--input samples/baselines/safety/caa_selection.jsonl` to generation and the same
path as `--benchmark-file` to scoring. The selection inputs retain benchmark
metadata and formatting, including category-specific HarmBench judge fields.

CANS-Grad-Only uses the existing `cans_compute_neuron_scales.py --method grad-only`
entry point, followed by `generate_responses.py`. Its K is 8,000 for safety
and toxicity and 1,000 for hallucination; strengths are 0.1 through 0.5.

## Source attribution

SN-Tune's importance calculation, GQA mapping, and sparse parameter updates
are ported from our experiment implementation. DAP follows the
activation-contrasting method and MLP-only IA3-DPO guide. AurA and CAA preserve
their projection-pooling/scaling and contrastive-direction definitions.
H-Neurons follows the CETT feature and sparse classifier definitions.
Original methods and source repositories:

- [SN-Tune](https://github.com/zhaoyiran924/Safety-Neuron): *Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neurons*.
- [DAP](https://github.com/THU-KEG/SafetyNeuron): *Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons*.
- [AurA](https://github.com/apple/ml-aura): *Whispering Experts: Neural Interventions for Toxicity Mitigation in Language Models*.
- [CAA](https://github.com/nrimsky/CAA): *Steering Llama 2 via Contrastive Activation Addition*.
- [H-Neurons](https://github.com/thunlp/H-Neurons): CETT feature construction and L1 classifier.

Dataset and model licenses remain those of their respective sources. Source
file identities and hashes are recorded in the baseline data manifest.
