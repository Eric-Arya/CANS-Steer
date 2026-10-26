"""FP32 target gradients and whole-response diagonal Fisher statistics."""

import torch
import torch.nn.functional as F

from .data_processing import padded_batch, save_tensor
from .model_interventions import down_projections, neuron_alphas


def response_gradients(model, rows, pad_id, *, average_tokens=False):
    """Differentiate each example separately using independent alpha rows."""
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("Extraction requires FP32 model computation")
    device = model.get_input_embeddings().weight.device
    ids, attention, targets = padded_batch(rows, pad_id, device)
    projections = down_projections(model)
    alpha = torch.ones(len(rows), len(projections), projections[0].in_features,
                       device=device, requires_grad=True)
    with neuron_alphas(model, alpha, attention):
        position_ids = (attention.cumsum(-1) - 1).clamp_min(0)
        logits = model(input_ids=ids, attention_mask=attention, position_ids=position_ids,
                       use_cache=False).logits[:, :-1]
        batch_ids, positions = targets.nonzero(as_tuple=True)
        if not len(positions):
            return torch.zeros_like(alpha).detach()
        # Materialize vocabulary probabilities only for scored positions.
        scores = -F.cross_entropy(logits[batch_ids, positions].float(),
                                  ids[batch_ids, positions + 1], reduction="none")
        if average_tokens:
            scores = scores / targets.sum(-1)[batch_ids]
        gradient = torch.autograd.grad(scores.sum(), alpha)[0]
    if not gradient.isfinite().all():
        raise FloatingPointError("Non-finite response gradient")
    return gradient.detach()


def extract_statistics(model, rows, pad_id, *, fisher=False, batch_size=1,
                       checkpoint=None, binding=None, resume=False,
                       quantize_per_example=False):
    """Average target gradients or squared sequence-score gradients, with resume."""
    if not rows or batch_size < 1:
        raise ValueError("Need nonempty inputs and a positive batch size")
    modules = down_projections(model)
    dtype = torch.float64 if fisher else torch.float32
    total = torch.zeros(len(modules), modules[0].in_features, dtype=dtype)
    completed = 0
    if resume and checkpoint and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["binding"] != binding:
            raise ValueError("Checkpoint belongs to different inputs or settings")
        total, completed = saved["sum"], saved["completed"]
    for start in range(completed, len(rows), batch_size):
        values = response_gradients(model, rows[start:start + batch_size], pad_id,
                                    average_tokens=not fisher)
        if fisher:
            values = values.double().square()
        elif quantize_per_example:
            # Round per-example storage to FP16; accumulate the mean in FP32.
            values = values.half().float()
        total += values.to(dtype).sum(0).cpu()
        completed = min(start + batch_size, len(rows))
        if completed % 32 < batch_size or completed == len(rows):
            if checkpoint:
                save_tensor(checkpoint, {"sum": total, "completed": completed, "binding": binding})
            print(f"Processed {completed}/{len(rows)} examples", flush=True)
    return total / len(rows)
