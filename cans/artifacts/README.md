# Released CANS artifacts

| File | Contents |
| --- | --- |
| `fisher.pt` | Shared whole-response Fisher diagonal |
| `safety.pt` | Safety target gradient and example CANS scales |
| `toxicity.pt` | Toxicity target gradient and example CANS scales |
| `hallucination.pt` | Hallucination target gradient and example CANS scales |

These tensors target Qwen3-4B. Their hashes and source provenance are recorded
in [`samples/manifest.json`](../../samples/manifest.json). The example scales can
be used directly with `scripts/generate_responses.py`; extraction and
scaling commands are in the [main README](../../README.md#main-pipeline).

Save newly computed tensors and trained baseline checkpoints under `outputs/`.
