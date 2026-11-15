"""Compare diagonal-CANS updates' full-Fisher quadratic with measured reference KL."""

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path

import torch

from cans.data_processing import config, padded_batch, read_jsonl, save_tensor, sha256, validate_fisher_binding
from cans.gradient_fisher_extraction import response_gradients
from cans.model_interventions import fold_scales, load_model, validate_model_source
from cans.scaling_solver import cans_scales


def frozen_json(path, value):
    """Create one immutable binding safely across simultaneous point workers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError(f"Frozen inputs or settings changed: {path}")
        else:
            temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(value, indent=2) + '\n')
            temporary.replace(path)


def score_sources(index_path, paths, model_config_sha256, reference_sha256):
    """Accept cached sequence scores only through a trusted model/token-bound index."""
    if index_path is None:
        if paths:
            raise ValueError('--score-chunks requires a trusted --score-index')
        return [], None
    index_path = Path(index_path).resolve()
    index = json.loads(index_path.read_text())
    if (index.get('model_config_sha256') != model_config_sha256
            or index.get('reference_sha256') != reference_sha256):
        raise ValueError('Score index belongs to another model or frozen reference corpus')
    manifest = index.get('source_manifest_sha256')
    if not isinstance(manifest, str) or len(manifest) != 64 or not index.get('chunks'):
        raise ValueError('Score index requires a source-manifest SHA256 and chunk records')
    records = []
    for item in index['chunks']:
        path = (index_path.parent/Path(item['path'])).resolve()
        if sha256(path) != item['sha256']:
            raise ValueError('Score-index chunk checksum differs')
        records.append(dict(path=str(path), sha256=item['sha256']))
    if len({r['path'] for r in records}) != len(records):
        raise ValueError('Score index repeats a chunk')
    if paths and {str(Path(p).resolve()) for p in paths} != {r['path'] for r in records}:
        raise ValueError('--score-chunks does not match the trusted score index')
    return records, dict(path=str(index_path), sha256=sha256(index_path),
                         source_manifest_sha256=manifest)


def reference_rows(path, expected, context):
    original = read_jsonl(path)
    if len(original) != expected:
        raise ValueError(f"Expected all {expected} frozen reference continuations")
    rows = []
    for index, row in enumerate(original):
        prompt = row['prompt_token_ids']
        response = row.get('response_token_ids', row.get('fisher_token_ids'))
        if not prompt or response is None or len(prompt)+len(response) > context:
            raise ValueError('Invalid frozen token IDs or context overflow; never truncate')
        candidate = row.get('candidate_index', index)
        rows.append(dict(id=candidate, candidate_index=candidate, input_ids=prompt+response,
                         target_indices=list(range(len(prompt), len(prompt)+len(response)))))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Reference continuation IDs must be unique')
    return rows


def project_scores(scores, deltas):
    """Sum neuron contributions before squaring, preserving every cross term."""
    return scores.double().flatten(1) @ deltas.double().flatten(1).T


def kl_batch(base, edited, rows, pad_id, chunk):
    """Full-vocabulary KL(base || edited), summed over scored tokens per response."""
    device = base.get_input_embeddings().weight.device
    ids, mask, targets = padded_batch(rows, pad_id, device)
    position_ids = (mask.cumsum(-1)-1).clamp_min(0)
    with torch.inference_mode():
        p = base(ids, attention_mask=mask, position_ids=position_ids, use_cache=False).logits[:, :-1]
        q = edited(ids, attention_mask=mask, position_ids=position_ids, use_cache=False).logits[:, :-1]
        batch_ids, positions = targets.nonzero(as_tuple=True)
        total = torch.zeros(len(rows), dtype=torch.float64, device=device)
        for start in range(0, len(positions), chunk):
            bi, pi = batch_ids[start:start+chunk], positions[start:start+chunk]
            logp = p[bi, pi].double().log_softmax(-1)
            logq = q[bi, pi].double().log_softmax(-1)
            total.scatter_add_(0, bi, (logp.exp()*(logp-logq)).sum(-1))
    if not total.isfinite().all() or (total < -1e-10).any():
        raise FloatingPointError('Invalid full-vocabulary KL')
    return total.clamp_min(0).cpu()


def checkpoint(path, binding, width, resume):
    if path.exists():
        if not resume:
            raise FileExistsError(f'Use --resume to reuse {path}')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['binding'] != binding:
            raise ValueError('Checkpoint belongs to different model, data, coefficients or settings')
        values = saved['values']
        if values.ndim != 2 or values.shape[1] != width or not values.isfinite().all():
            raise ValueError('Invalid diagnostic checkpoint')
        return values
    return torch.empty((0, width), dtype=torch.float64)


def fisher_diagnostic(args, common, rows, deltas, binding):
    path = args.output/'projected_scores.pt'
    values = checkpoint(path, binding, len(deltas), args.resume)
    if len(values) > len(rows):
        raise ValueError('Checkpoint has too many reference responses')
    digest = path.with_suffix('.json')
    if len(values) == len(rows) and digest.exists():
        if json.loads(digest.read_text())['sha256'] != sha256(path):
            raise ValueError('Completed projected-score cache changed')
    if len(values) < len(rows) and binding['score_chunks']:
        projected, indices = [], []
        for source in binding['score_chunks']:
            path_source = Path(source['path'])
            if sha256(path_source) != source['sha256']:
                raise ValueError('Source sequence-score cache changed')
            saved = torch.load(path_source, map_location='cpu', weights_only=True)
            if saved.get('manifest_sha256') != binding['score_index']['source_manifest_sha256']:
                raise ValueError('Score chunk belongs to another extraction manifest')
            scores, ids = saved['gradients'], saved['candidate_indices']
            if scores.dtype != torch.float32 or tuple(scores.shape[1:]) != tuple(deltas.shape[1:]) or len(scores) != len(ids):
                raise ValueError('Expected FP32 whole-response score gradients with matching coordinates')
            if not scores.isfinite().all(): raise ValueError('Non-finite cached sequence scores')
            projected.append(project_scores(scores, deltas)); indices.extend(ids)
        expected = [r['candidate_index'] for r in rows]
        if len(indices) != len(set(indices)) or set(indices) != set(expected):
            raise ValueError('Score-cache continuation IDs differ from frozen references')
        order = {candidate:i for i,candidate in enumerate(indices)}
        values = torch.cat(projected)[[order[i] for i in expected]]
        save_tensor(path, dict(binding=binding, values=values))
    if len(values) < len(rows):
        model, tok = load_model(args.model or common['model'], device=args.device,
                                extraction=True, revision=common.get('revision'))
        size = common['fisher_batch_size']
        for start in range(len(values), len(rows), size):
            scores = response_gradients(model, rows[start:start+size], tok.pad_token_id)
            projected = project_scores(scores.cpu(), deltas)
            values = torch.cat((values, projected))
            save_tensor(path, dict(binding=binding, values=values))
            print(f'Full Fisher projections: {len(values)}/{len(rows)}', flush=True)
        del model, tok, scores
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    frozen_json(digest, dict(sha256=sha256(path), count=len(values)))
    return (values.square().mean(0)/2).tolist()


def measured_kl(args, common, rows, deltas, binding, settings):
    paths = [args.output/f'kl_{i:02d}.pt' for i in range(len(deltas))]
    selected = range(len(deltas)) if args.point_index is None else [args.point_index]
    cached = {i:checkpoint(paths[i], dict(binding, point_index=i), 1, args.resume) for i in selected}
    if all(len(values) == len(rows) for values in cached.values()):
        return {i:float(values.mean()) for i,values in cached.items()}
    if any(len(values) > len(rows) for values in cached.values()):
        raise ValueError('KL checkpoint has too many reference responses')
    base, tok = load_model(args.model or common['model'], device=args.device,
                            revision=common.get('revision'))
    try:
        for index in selected:
            delta, path, values = deltas[index], paths[index], cached[index]
            if len(values) == len(rows): continue
            edited, _ = load_model(args.model or common['model'], device=args.device,
                                   revision=common.get('revision'))
            try:
                fold_scales(edited, 1+delta)
                size = settings['kl_batch_size']
                for start in range(len(values), len(rows), size):
                    measured = kl_batch(base, edited, rows[start:start+size], tok.pad_token_id,
                                        settings['vocabulary_chunk_tokens']).reshape(-1, 1)
                    values = torch.cat((values, measured))
                    save_tensor(path, dict(binding=dict(binding, point_index=index), values=values))
                    print(f'KL point {index+1}/{len(deltas)}: {len(values)}/{len(rows)}', flush=True)
                cached[index] = values
            finally:
                del edited
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    finally:
        del base, tok
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {i:float(values.mean()) for i,values in cached.items()}


def approximation_metrics(predicted, actual):
    p, a = torch.tensor(predicted, dtype=torch.float64), torch.tensor(actual, dtype=torch.float64)
    residual, total = float((p-a).square().sum()), float((a-a.mean()).square().sum())
    nonzero = a != 0
    return dict(r_squared_identity=1-residual/total if total else None,
                mape_percent=float(((p[nonzero]-a[nonzero]).abs()/a[nonzero].abs()).mean()*100) if nonzero.any() else None,
                mape_zero_kl_points_excluded=int((~nonzero).sum()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/toxicity.yaml'))
    parser.add_argument('--model', help='Matching local base model or configured model ID')
    parser.add_argument('--gradient', type=Path)
    parser.add_argument('--fisher', type=Path)
    parser.add_argument('--reference', type=Path, help='Complete frozen model-matched tokenized reference JSONL')
    parser.add_argument('--score-chunks', type=Path, nargs='+', help='Authenticated FP32 sequence-score chunks; omit to extract from the model')
    parser.add_argument('--score-index', type=Path, help='Trusted JSON binding cached chunks to model_config_sha256, reference_sha256 and source_manifest_sha256')
    parser.add_argument('--epsilons', type=float, nargs='+', help='Override the model-specific paper epsilon preset')
    parser.add_argument('--damping', type=float, help='Explicit override; otherwise use the domain setting/median')
    parser.add_argument('--stage', choices=('fisher', 'kl', 'all'), default='all')
    parser.add_argument('--point-index', type=int, help='Run only this zero-based KL point; keeps all configured epsilon bindings')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    # Keep FP64 coefficient reductions identical across independent point workers.
    torch.set_num_threads(1)
    root, common, domain = config(args.config)
    if not domain or domain['cans'].get('cap') is not None:
        raise ValueError('Use a domain config with cap: null')
    settings = common['evaluation_protocol']['approximation']
    if settings['model_dtype'] != 'bfloat16' or settings['probability_dtype'] != 'float64':
        raise ValueError('Actual KL requires BF16 weights and FP64 probabilities')
    if min(settings['kl_batch_size'], settings['vocabulary_chunk_tokens'], common['fisher_batch_size']) < 1:
        raise ValueError('Diagnostic batch/chunk sizes must be positive')
    model_key = 'qwen3_4b' if 'qwen3' in common['model'].lower() else 'llama2' if 'llama-2' in common['model'].lower() else None
    if args.epsilons is None and domain['domain'] != 'toxicity':
        raise ValueError('Paper epsilon presets describe toxicity; supply --epsilons for another domain')
    epsilon = args.epsilons or settings['epsilon_presets'].get(model_key)
    if not epsilon or len(set(epsilon)) != len(epsilon):
        raise ValueError('Supply distinct epsilon values for this model')
    if args.point_index is not None and (args.stage != 'kl' or not 0 <= args.point_index < len(epsilon)):
        raise ValueError('--point-index requires stage kl and a valid zero-based epsilon index')
    gradient = args.gradient or root/domain['artifact']
    fisher = args.fisher or root/common['fisher_artifact']
    reference = args.reference or root/common['reference_data']
    validate_model_source(args.model or common['model'], common['model_config_sha256'])
    g_obj = torch.load(gradient, map_location='cpu', weights_only=True)
    f_obj = torch.load(fisher, map_location='cpu', weights_only=True)
    fisher_binding = validate_fisher_binding(f_obj, common, reference, fisher)
    g = g_obj.get('gradient', g_obj.get('mean_g'))
    f = f_obj['fisher'].reshape(common['shape'])
    if g is None or list(g.shape) != common['shape'] or g.dtype != torch.float32:
        raise ValueError('Expected matching FP32 target gradients')
    for obj in (g_obj, f_obj):
        recorded = obj.get('metadata', {}).get('model_config_sha256')
        if recorded is not None and recorded != common['model_config_sha256']:
            raise ValueError('Statistics belong to another model')
    rows = reference_rows(reference, common['reference_generation']['num_prompts'] *
                          common['reference_generation']['continuations_per_prompt'], common['max_model_len'])
    damping = args.damping if args.damping is not None else domain['cans'].get('damping')
    solved = [cans_scales(domain['objective_sign']*g, f, e, damping=damping, cap=None) for e in epsilon]
    deltas = torch.stack([scale-1 for scale, _ in solved])
    shared = args.config.resolve().parent/'evaluation.yaml'
    prior = args.output/'protocol.json'
    if prior.exists() and not args.score_chunks and not args.score_index:
        previous = json.loads(prior.read_text())
        cached_sources, score_index = previous.get('score_chunks', []), previous.get('score_index')
        if score_index and sha256(Path(score_index['path'])) != score_index['sha256']:
            raise ValueError('Trusted score index changed after projection')
    else:
        cached_sources, score_index = score_sources(args.score_index, args.score_chunks,
            common['model_config_sha256'], sha256(reference))
    binding = dict(schema='cans_full_fisher_approximation_v1', model=str(Path(args.model).resolve()) if args.model and Path(args.model).is_dir() else args.model or common['model'],
        revision=common.get('revision'), model_config_sha256=common['model_config_sha256'],
        config_sha256=sha256(args.config), evaluation_sha256=sha256(shared),
        gradient_sha256=sha256(gradient), fisher_sha256=sha256(fisher), reference_sha256=sha256(reference),
        fisher_binding=fisher_binding,
        script_sha256=sha256(Path(__file__)), score_chunks=cached_sources, score_index=score_index,
        delta_sha256=hashlib.sha256(deltas.numpy().tobytes()).hexdigest(),
        epsilons=epsilon, damping=solved[0][1]['damping'], cap=None,
        objective_sign=domain['objective_sign'], reference_count=len(rows), settings=settings,
        definitions=dict(full_fisher='mean((sequence_logprob_gradient dot delta)**2)/2; no damping',
                         actual_kl='KL(base || edited); sum sampled-response token KL then mean continuations',
                         fisher_precision='FP32 model and sequence scores; FP64 projection and accumulation',
                         actual_kl_precision='BF16 base/edited weights; FP64 full-vocabulary probabilities'))
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_json(args.output/'protocol.json', binding)
    torch.manual_seed(settings['seed'])
    q_full = fisher_diagnostic(args, common, rows, deltas, binding) if args.stage in ('fisher', 'all') else None
    actual = measured_kl(args, common, rows, deltas, binding, settings) if args.stage in ('kl', 'all') else {}
    # Separate stages can be resumed and combined without recomputing model passes.
    if q_full is None and (args.output/'projected_scores.pt').exists():
        cached = checkpoint(args.output/'projected_scores.pt', binding, len(deltas), args.resume)
        if len(cached) == len(rows):
            if json.loads((args.output/'projected_scores.json').read_text())['sha256'] != sha256(args.output/'projected_scores.pt'):
                raise ValueError('Completed projected-score cache changed')
            q_full = (cached.square().mean(0)/2).tolist()
    if args.point_index is None:
        for i in range(len(deltas)):
            if i not in actual and (args.output/f'kl_{i:02d}.pt').exists():
                cached = checkpoint(args.output/f'kl_{i:02d}.pt', dict(binding, point_index=i), 1, args.resume)
                if len(cached) == len(rows): actual[i] = float(cached.mean())
    points = [dict(epsilon=e, damping=info['damping'], q_diagonal=float((f.double()*delta.square()).sum()/2),
        damped_diagonal_budget=info['post_cap_cost'], q_full=None if q_full is None else q_full[i],
        actual_kl=actual.get(i),
        relative_error_percent=None if i not in actual or q_full is None or actual[i] == 0 else 100*(q_full[i]-actual[i])/actual[i])
        for i, (e, (_, info), delta) in enumerate(zip(epsilon, solved, deltas, strict=True))]
    result = dict(protocol=binding, complete=q_full is not None and len(actual) == len(deltas), points=points)
    if result['complete']: result['metrics'] = approximation_metrics(q_full, [actual[i] for i in range(len(deltas))])
    suffix = '' if args.point_index is None else f'_{args.point_index:02d}'
    (args.output/f'summary{suffix}.json').write_text(json.dumps(result, indent=2)+'\n')
    with (args.output/f'points{suffix}.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0])); writer.writeheader(); writer.writerows(points)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
