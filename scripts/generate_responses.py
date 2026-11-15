"""Generate BF16 benchmark responses for CANS, Grad-only, and all baselines."""

import argparse
from contextlib import nullcontext
from functools import partial
import json
import os
from pathlib import Path

import torch
import yaml

from cans.data_processing import config, evaluation_config, read_jsonl, sha256, write_jsonl
from cans.capability_benchmarks import EXTENSIONS, prepare_prompts, resolve_task, response_budget
from cans.model_interventions import fold_scales, fold_vllm_scales, load_model, validate_model_source
from baselines.common import load_config as baseline_config, prefix

BASELINE_METHODS = ('sft', 'sn_tune', 'dpo', 'dap', 'aura', 'aura_jigsaw', 'caa', 'h_neurons')


def load_generation_config(path):
    document = yaml.safe_load(Path(path).read_text())
    is_baseline = any(method in document for method in BASELINE_METHODS)
    root, common, spec = baseline_config(path) if is_baseline else config(path)
    return root, common, spec, is_baseline


def encode_prompt(tokenizer, text):
    """Native chat templates may already supply BOS; never add it twice."""
    bos = getattr(tokenizer, "bos_token", None)
    return tokenizer.encode(text, add_special_tokens=not bool(bos and text.startswith(bos)))


def generate_vllm(args, common, rows, settings):
    common, settings = resolve_task(common, args.benchmark, settings)
    runtime = common.get("evaluation_runtime", evaluation_config()["runtime"])
    os.environ["VLLM_USE_V1"] = runtime["vllm_version_mode"]
    torch.backends.cuda.matmul.allow_tf32 = runtime["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["cudnn_allow_tf32"]
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model or common["model"], revision=common.get("revision"),
              dtype=runtime["dtype"], tensor_parallel_size=runtime["tensor_parallel_size"],
              enforce_eager=runtime["enforce_eager"],
              enable_prefix_caching=runtime["enable_prefix_caching"],
              enable_chunked_prefill=runtime["enable_chunked_prefill"],
              max_model_len=common["max_model_len"],
              gpu_memory_utilization=runtime["gpu_memory_utilization"], max_num_seqs=settings["batch_size"],
              seed=common["seed"])
    tokenizer = llm.get_tokenizer()
    rows = prepare_prompts(tokenizer, rows, args.benchmark, settings)
    if args.scales:
        artifact = torch.load(args.scales, map_location="cpu", weights_only=True)
        if artifact["metadata"]["model"] != common["model"]:
            raise ValueError("Scaling artifact belongs to another base model")
        llm.apply_model(partial(fold_vllm_scales, scales=artifact["scales"]))
    sampling = {"temperature": settings.get("temperature", 0), "max_tokens": settings["max_new_tokens"],
                "seed": common["seed"], "stop": settings.get("stop")}
    if args.benchmark == "mmlu":
        choices = [tokenizer.encode(letter, add_special_tokens=False) for letter in settings.get("choices", "ABCD")]
        if any(len(tokens) != 1 for tokens in choices):
            raise ValueError("MMLU choices must be single tokens")
        sampling.update(allowed_token_ids=[tokens[0] for tokens in choices], logprobs=4)
    results = []
    for start in range(0, len(rows), settings["batch_size"]):
        batch = rows[start:start + settings["batch_size"]]
        prompts = [{"prompt_token_ids": encode_prompt(tokenizer, row["formatted_prompt"])} for row in batch]
        if any(len(p["prompt_token_ids"]) >= common["max_model_len"] for p in prompts):
            raise ValueError("Frozen prompt leaves no response capacity; do not truncate it")
        if args.benchmark == "rtp":
            parameters = [SamplingParams(temperature=settings.get("temperature", .6),
                           top_p=settings.get("top_p", .9), top_k=settings.get("top_k", 50),
                           n=settings.get("samples_per_prompt", 25),
                           max_tokens=settings["max_new_tokens"], seed=settings["seed"] + start + i) for i in range(len(batch))]
        elif args.benchmark in EXTENSIONS:
            parameters = [SamplingParams(**dict(sampling, max_tokens=response_budget(
                row, len(prompt['prompt_token_ids']), args.benchmark, common, settings)))
                for row, prompt in zip(batch, prompts, strict=True)]
        else:
            parameters = SamplingParams(**sampling)
        outputs = llm.generate(prompts, parameters, use_tqdm=False)
        for row, output in zip(batch, outputs, strict=True):
            responses = [candidate.text for candidate in output.outputs]
            effective = min(settings["max_new_tokens"], common["max_model_len"] - len(output.prompt_token_ids))
            result = {**row, "responses": responses, "prompt_token_count": len(output.prompt_token_ids),
                      "effective_response_tokens": effective,
                      "generated_token_ids": [list(candidate.token_ids) for candidate in output.outputs],
                      "context_limited": effective < settings['max_new_tokens'],
                      "hit_token_limit": any(len(candidate.token_ids) >= effective for candidate in output.outputs),
                      "finish_reasons": [candidate.finish_reason for candidate in output.outputs]}
            if any(len(ids) > effective for ids in result["generated_token_ids"]):
                raise ValueError("Generation exceeds the configured prompt-plus-response capacity")
            if args.benchmark == "mmlu":
                result["prediction"] = [tokens[0] for tokens in choices].index(output.outputs[0].token_ids[0])
            results.append(result)
        print(f"Generated {len(results)}/{len(rows)} prompts", flush=True)
        if getattr(args, "progress_callback", None):
            args.progress_callback(results[-len(batch):])
    return results


def generate_transformers(a, common, spec, rows, settings):
    common, settings = resolve_task(common, a.benchmark, settings)
    model,tok=load_model(a.model or common['model'],device=a.device,revision=common.get('revision'))
    rows = prepare_prompts(tok, rows, a.benchmark, settings)
    context=nullcontext();method='base'
    if a.artifact:
        obj=torch.load(a.artifact,map_location='cpu',weights_only=True);method=obj['method']
        if obj['metadata']['model']!=common['model'] or obj['metadata']['domain']!=spec['domain']:
            raise ValueError('Baseline model/domain mismatch')
        if method=='sft':
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,a.artifact.parent/obj['adapter']).merge_and_unload().to(dtype=torch.bfloat16).eval()
        elif method=='sn_tune':
            from baselines.sn_tune import apply_sparse_state
            apply_sparse_state(model,obj['sparse_deltas'],a.strength)
        elif method in ('aura','aura_jigsaw'):
            from baselines.aura import intervention
            context=intervention(model,obj['multipliers'])
        elif method=='caa':
            from baselines.caa import intervention
            if 'vector' in obj:layer,vector=obj['layer'],obj['vector']
            else:
                if a.layer is None:raise ValueError('--layer is required for unselected CAA vectors')
                layer,vector=a.layer,obj['vectors'][a.layer]
            context=intervention(model,vector,layer,a.strength)
        elif method=='h_neurons':
            if 'candidates' in obj:
                if a.candidate is None:raise ValueError('--candidate is required for H scan artifacts')
                obj=obj['candidates'][a.candidate]
            scales=torch.ones(obj['shape']);scales.flatten()[obj['selected']]=a.strength
            fold_scales(model,scales)
        elif method=='dpo':
            from baselines.dap import apply_ia3
            apply_ia3(model,obj['ia3_scales'])
        elif method=='dap':
            from baselines.dap import apply_ia3,intervention
            if not a.guide:raise ValueError('--guide required for DAP')
            guide_obj=torch.load(a.guide,map_location='cpu',weights_only=True)
            if guide_obj['metadata']['model']!=common['model'] or guide_obj['metadata']['domain']!=spec['domain']:raise ValueError('Wrong DAP guide')
            expected=obj['metadata'].get('guide_sha256')
            if expected and expected!=sha256(a.guide):raise ValueError('DAP guide differs from ranking guide')
            guide,_=load_model(a.model or common['model'],device=a.device,revision=common.get('revision'))
            apply_ia3(guide,guide_obj['ia3_scales']);context=intervention(model,guide,obj['selected'])
        else:raise ValueError(f'Unsupported baseline artifact {method}')
    results=[]
    with context,torch.inference_mode():
        for i,row in enumerate(rows):
            text=row.get('formatted_prompt')
            if text is None:
                raw=row['prompt']
                text=prefix(tok,raw,'raw' if a.benchmark=='rtp' else 'chat')
            ids=torch.tensor([encode_prompt(tok,text)],device=model.device)
            available = common.get('max_model_len', model.config.max_position_embeddings) - ids.shape[1]
            if available < 1:raise ValueError('Frozen prompt leaves no response capacity; do not truncate it')
            effective = (response_budget(row, ids.shape[1], a.benchmark, common, settings)
                         if a.benchmark in EXTENSIONS else min(settings['max_new_tokens'], available))
            seed=settings.get('seed',common['seed'])+i if a.benchmark=='rtp' else common['seed']
            torch.manual_seed(seed)
            kw={'max_new_tokens':effective,'pad_token_id':tok.pad_token_id,'use_cache':True}
            if a.benchmark=='rtp':kw.update(do_sample=True,temperature=settings.get('temperature',.6),
                top_p=settings.get('top_p',.9),top_k=settings.get('top_k',50),
                num_return_sequences=settings.get('samples_per_prompt',25))
            else:kw.update(do_sample=False)
            if settings.get('stop'):kw.update(stop_strings=settings['stop'],tokenizer=tok)
            if a.benchmark=='mmlu':
                choices=[tok.encode(c,add_special_tokens=False) for c in settings.get('choices','ABCD')]
                if any(len(x)!=1 for x in choices):raise ValueError('MMLU choices must be single tokens')
                allowed=[x[0] for x in choices]
                kw['prefix_allowed_tokens_fn']=lambda batch_id,input_ids:allowed
            generated=model.generate(ids,attention_mask=torch.ones_like(ids),**kw)
            tokens=[r[ids.shape[1]:].tolist() for r in generated]
            texts=[tok.decode(r,skip_special_tokens=True) for r in tokens]
            for n,text in enumerate(texts):
                for stop in settings.get('stop',[]):text=text.split(stop,1)[0]
                texts[n]=text
            result={**row,'formatted_prompt':tok.decode(ids[0],skip_special_tokens=False),
                    'responses':texts,'generated_token_ids':tokens,
                    'prompt_token_count':ids.shape[1],'effective_response_tokens':effective,
                    'context_limited':effective < settings['max_new_tokens'],
                    'hit_token_limit':any(len(t)>=effective for t in tokens)}
            # Preserve supplied prompt bytes for scorer identity checks.
            if 'formatted_prompt' in row:result['formatted_prompt']=row['formatted_prompt']
            if a.benchmark=='mmlu':result['prediction']=allowed.index(tokens[0][0])
            results.append(result)
            if getattr(a, 'progress_callback', None):a.progress_callback([result])
            if (i+1)%10==0:print(f'{i+1}/{len(rows)}',flush=True)
    return results,method


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/model.yaml'))
    parser.add_argument('--model', help='Matching local model path or Hugging Face model ID')
    parser.add_argument('--benchmark', required=True,
                        choices=['harmbench', 'rtp', 'truthfulqa', 'ifeval', 'bbh', 'mmlu', 'mbpp', 'mgsm'])
    intervention = parser.add_mutually_exclusive_group()
    intervention.add_argument('--scales', type=Path, help='CANS or Grad-only neuron scales (vLLM)')
    intervention.add_argument('--artifact', type=Path, help='Baseline intervention artifact (Transformers)')
    parser.add_argument('--guide', type=Path, help='Matching IA3-DPO artifact for DAP')
    parser.add_argument('--strength', type=float, default=1., help='SN/CAA strength or H-Neurons multiplier')
    parser.add_argument('--layer', type=int, help='CAA candidate layer for an unselected vector artifact')
    parser.add_argument('--candidate', type=int, help='H-Neurons candidate index for a scan artifact')
    parser.add_argument('--input', type=Path, help='Frozen selection inputs; omit for the complete benchmark')
    parser.add_argument('--device', default='cuda:0', help='Transformers device; use CUDA_VISIBLE_DEVICES for vLLM')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root, common, spec, is_baseline = load_generation_config(args.config)
    if args.artifact and not is_baseline:
        parser.error('--artifact requires a baseline config such as configs/baselines/qwen3_safety.yaml')
    if args.guide and not args.artifact:
        parser.error('--guide requires --artifact')
    use_transformers = is_baseline and not args.scales
    if not use_transformers and (args.layer is not None or args.candidate is not None):
        parser.error('--layer and --candidate apply to baseline artifacts')
    if args.output.exists():
        raise FileExistsError(f'Use a fresh output path: {args.output}')
    common, settings = resolve_task(common, args.benchmark, common['evaluation'][args.benchmark])
    source = args.input or (root / settings['input'] if settings.get('input') else root / common.get('benchmark_data', 'samples/benchmarks') / f'{args.benchmark}.jsonl')
    rows = read_jsonl(source)
    if not args.input and len(rows) != settings['count']:
        raise ValueError('Frozen benchmark count mismatch')
    if not rows:
        raise ValueError('Benchmark input is empty')
    validate_model_source(args.model or common['model'], common['model_config_sha256'])
    if use_transformers:
        results, method = generate_transformers(args, common, spec, rows, settings)
        backend = 'transformers-cached'
    else:
        results = generate_vllm(args, common, rows, settings)
        method = torch.load(args.scales, map_location='cpu', weights_only=True)['metadata'].get('method', 'cans') if args.scales else 'base'
        backend = 'vllm-v0-eager'
    write_jsonl(args.output, results)
    metadata = {'model': common['model'],
                'model_source': str(Path(args.model).resolve()) if args.model and Path(args.model).exists() else (args.model or common['model']),
                'revision': common.get('revision'), 'model_config_sha256': common['model_config_sha256'],
                'method': method, 'benchmark': args.benchmark,
                'dtype': 'bfloat16', 'backend': backend, 'count': len(rows),
                'seed': common['seed'], 'settings': settings,
                'data_sha256': sha256(source), 'responses_sha256': sha256(args.output),
                'scales_sha256': sha256(args.scales) if args.scales else None,
                'artifact_sha256': sha256(args.artifact) if args.artifact else None,
                'guide_sha256': sha256(args.guide) if args.guide else None,
                'strength': args.strength, 'layer': args.layer, 'candidate': args.candidate,
                'scope': 'selection' if args.input else 'complete benchmark',
                'max_model_len': common['max_model_len'],
                'hit_token_limit_count': sum(r.get('hit_token_limit',False) for r in results),
                'context_limited_count': sum(r.get('context_limited',False) for r in results)}
    metadata['hit_token_limit_fraction'] = metadata['hit_token_limit_count']/len(results)
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
