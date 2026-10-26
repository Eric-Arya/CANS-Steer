"""Extract CANS Grad/Fisher statistics or baseline neuron rankings and interventions."""

import argparse
from pathlib import Path
import torch

from cans.data_processing import config, encode_span, read_jsonl, save_tensor, sha256
from cans.gradient_fisher_extraction import extract_statistics
from cans.model_interventions import load_model, validate_model_source
from baselines.common import load_config as load_baseline_config, metadata


def extract_grad(args):
    root, common, domain = config(args.config)
    source = root / domain["data"]
    validate_model_source(args.model or common["model"], common["model_config_sha256"])
    model, tokenizer = load_model(args.model or common["model"], device=args.device,
                                  extraction=True, revision=common.get("revision"))
    rows = [encode_span(tokenizer, row) for row in read_jsonl(source)]
    binding = {"model": common["model"], "model_config_sha256": common["model_config_sha256"],
               "data_sha256": sha256(source), "scope": "global/all-token",
               "objective": "mean-target-token-logprob", "model_dtype": "float32",
               "per_example_storage_dtype": "float16", "mean_dtype": "float32",
               "count": len(rows)}
    mean = extract_statistics(model, rows, tokenizer.pad_token_id,
                              checkpoint=args.output.with_suffix(".checkpoint.pt"),
                              binding=binding, resume=args.resume,
                              quantize_per_example=True)
    save_tensor(args.output, {"gradient": mean, "metadata": binding})


def extract_fisher(args):
    root, common, _ = config(args.config)
    source = root / common["reference_data"]
    validate_model_source(args.model or common["model"], common["model_config_sha256"])
    rows = []
    for row in read_jsonl(source):
        prompt, response = row["prompt_token_ids"], row["response_token_ids"]
        rows.append({"input_ids": prompt + response,
                     "target_indices": list(range(len(prompt), len(prompt) + len(response)))})
    model, tokenizer = load_model(args.model or common["model"], device=args.device,
                                  extraction=True, revision=common.get("revision"))
    binding = {"model": common["model"], "model_config_sha256": common["model_config_sha256"],
               "data_sha256": sha256(source), "scope": "global/all-token",
               "estimator": "whole-response-logprob-gradient-square",
               "model_dtype": "float32", "accumulator_dtype": "float64",
               "batch_size": common["fisher_batch_size"], "count": len(rows)}
    fisher = extract_statistics(model, rows, tokenizer.pad_token_id, fisher=True,
                                batch_size=common["fisher_batch_size"],
                                checkpoint=args.output.with_suffix(".checkpoint.pt"),
                                binding=binding, resume=args.resume)
    save_tensor(args.output, {"fisher": fisher.float(), "fisher_fp64": fisher,
                             "metadata": binding})


def extract_baseline(a):
    root,common,spec=load_baseline_config(a.config);cfg=spec[a.method]
    if a.output.exists():raise FileExistsError(a.output)
    source=root/cfg.get('identification_data',cfg['data']);rows=read_jsonl(source)
    if len(rows)!=cfg.get('identification_count',cfg['count']):raise ValueError('Frozen corpus count mismatch')
    validate_model_source(a.model or common['model'],common['model_config_sha256'])
    model,tok=load_model(a.model or common['model'],device=a.device,
                        extraction=a.method in ('aura','aura_jigsaw','caa'),revision=common.get('revision'))
    sources=[source]
    if a.method=='sn_tune':
        # Importance requires explicit causal attention probabilities.
        model.config._attn_implementation='eager'
        from baselines.sn_tune import identify
        result=identify(model,tok,rows,cap=cfg['cap'])
    elif a.method in ('aura','aura_jigsaw'):
        from baselines.aura import extract
        result=extract(model,tok,rows,cfg['format'],cfg.get('response_pool_limit'),cfg.get('text_only',False))
    elif a.method=='caa':
        from baselines.caa import extract
        result={'vectors':extract(model,tok,rows,cfg['format'])}
    elif a.method=='h_neurons':
        from baselines.h_neurons import extract
        result={'features':extract(model,tok,rows,cfg['format'])}
    else:
        from baselines.dap import apply_ia3,identify
        if not a.guide:raise ValueError('--guide is required for DAP')
        obj=torch.load(a.guide,map_location='cpu',weights_only=True)
        if obj['metadata']['model']!=common['model'] or obj['metadata']['domain']!=spec['domain']:
            raise ValueError('DAP guide model/domain mismatch')
        guide,_=load_model(a.model or common['model'],device=a.device,revision=common.get('revision'))
        apply_ia3(guide,obj['ia3_scales']);sources.append(a.guide)
        result=identify(model,guide,tok,rows,cfg['format'],cfg['max_new_tokens'])
    result.update(method=a.method,metadata=metadata(common,spec,a.method,sources))
    if a.guide:
        from cans.data_processing import sha256
        result['metadata']['guide_sha256']=sha256(a.guide)
    save_tensor(a.output,result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True,
                        choices=['grad', 'fisher', 'sn_tune', 'aura', 'aura_jigsaw', 'caa', 'h_neurons', 'dap'])
    parser.add_argument('--config', type=Path, help='Model/domain config for CANS; baseline config for other methods')
    parser.add_argument('--model', help='Matching local model path or Hugging Face model ID')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true', help='Resume Grad/Fisher extraction')
    parser.add_argument('--guide', type=Path, help='Matching IA3-DPO artifact for DAP')
    args = parser.parse_args(argv)
    if args.config is None:
        if args.method != 'fisher':
            parser.error('--config is required except for Fisher with configs/model.yaml')
        args.config = Path('configs/model.yaml')
    if args.resume and args.method not in ('grad', 'fisher'):
        parser.error('--resume is supported only for Grad and Fisher')
    if args.method == 'dap' and not args.guide:
        parser.error('--guide is required for DAP')
    if args.guide and args.method != 'dap':
        parser.error('--guide applies only to DAP')
    if args.method == 'grad':
        extract_grad(args)
    elif args.method == 'fisher':
        extract_fisher(args)
    else:
        extract_baseline(args)


if __name__ == '__main__':
    main()
