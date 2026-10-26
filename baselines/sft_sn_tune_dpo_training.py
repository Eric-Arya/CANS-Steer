"""Portable LoRA SFT, sparse SN-Tune, and MLP-only IA3-DPO training."""
import math
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import get_scheduler
from cans.data_processing import padded_batch, save_tensor
from .common import encode_response, log_probabilities, prefix, save_json
from .sn_tune import STRUCTURES, install_sparse_deltas, sparse_state
from .dap import install_ia3, ia3_state


def sft_rows(tokenizer,rows,cfg):
    return [encode_response(tokenizer,r,style=cfg['format'],target=cfg['target'],
        truncate_at_target=cfg['target']=='span',add_special_tokens=False) for r in rows]


def sn_rows(tokenizer,rows,cfg):
    encoded=[]
    for row in rows:
        if cfg['format']=='raw':text=row['prompt']+'. '+row['response']
        else:
            text=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']},
                {'role':'assistant','content':row['response']}],tokenize=False,enable_thinking=False)
        ids=tokenizer.encode(text,add_special_tokens=True)[:cfg.get('max_length',2048)]
        encoded.append({'id':row['id'],'input_ids':ids,'target_indices':[i for i in range(1,len(ids)) if ids[i]!=tokenizer.eos_token_id]})
    return encoded


def preference_rows(tokenizer,rows,cfg):
    """Freeze preference tokenization; prompt targets excluded, terminal EOS included."""
    result=[]
    for row in rows:
        # Some frozen preference corpora already contain the native chat prefix.
        prompt=row['prompt'] if cfg.get('prompt_is_formatted',False) else prefix(tokenizer,row['prompt'],cfg['format'])
        branches=[]
        for key in ('chosen','rejected'):
            encoded=tokenizer(prompt+row[key],add_special_tokens=False,return_offsets_mapping=True)
            ids=encoded['input_ids'];boundary=next((i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>len(prompt)),len(ids))
            pids,rids=ids[:boundary],ids[boundary:]
            if tokenizer.bos_token_id is not None and (not pids or pids[0]!=tokenizer.bos_token_id):pids=[tokenizer.bos_token_id]+pids
            if not rids or rids[-1]!=tokenizer.eos_token_id:rids=rids+[tokenizer.eos_token_id]
            if len(pids)+len(rids)>cfg['max_length']:pids=pids[-cfg['max_prompt_length']:]
            rids=rids[:cfg['max_length']-len(pids)]
            if not rids:raise ValueError('Preference response lost to truncation')
            branches.append({'id':row['id']+'-'+key,'input_ids':pids+rids,'target_indices':list(range(len(pids),len(pids)+len(rids)))})
        result.append(branches)
    return result


def train(model,tokenizer,rows,cfg,method,output,metadata,selection=None,resume=None,until_epoch=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    seed=cfg.get('seed',112)
    torch.manual_seed(seed);random.seed(seed)
    model.config.use_cache=False
    if method=='sft':
        from peft import LoraConfig, get_peft_model
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,task_type='CAUSAL_LM',
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        encoded=sft_rows(tokenizer,rows,cfg)
    elif method=='sn_tune':
        install_sparse_deltas(model,[selection[name] for name in STRUCTURES])
        encoded=sn_rows(tokenizer,rows,cfg)
    elif method=='dpo':
        encoded=preference_rows(tokenizer,rows,cfg)
        # Frozen base probabilities are evaluated once and reused throughout DPO.
        reference=[]
        with torch.no_grad():
            for pair in encoded:
                values,_=log_probabilities(model,pair,tokenizer.pad_token_id)
                reference.append(values.detach().cpu())
        install_ia3(model)
    else:raise ValueError(method)
    params=[p for p in model.parameters() if p.requires_grad]
    for p in params:p.data=p.data.float()
    if cfg.get('gradient_checkpointing',False):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],betas=(.9,.999),eps=1e-8)
    batch=cfg['effective_batch_size'];micro=cfg['microbatch_size']
    steps=math.ceil(len(rows)/batch)*cfg['epochs']
    scheduler=get_scheduler(cfg['scheduler'],optimizer,num_warmup_steps=math.ceil(steps*cfg.get('warmup_ratio',0)),num_training_steps=steps)
    start_epoch=0;step=0
    if resume:
        checkpoint=torch.load(Path(resume)/'training_state.pt',map_location=model.device,weights_only=True)
        if checkpoint['identity']!=metadata:raise ValueError('Resume config/data identity mismatch')
        current=dict(model.named_parameters())
        with torch.no_grad():
            for name,value in checkpoint['trainable'].items():current[name].copy_(value)
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        torch.set_rng_state(checkpoint['rng'].cpu())
        if torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint['cuda_rng']])
        start_epoch=checkpoint['epoch'];step=checkpoint['step']
    model.train()
    stop_epoch=cfg['epochs'] if until_epoch is None else min(until_epoch,cfg['epochs'])
    if stop_epoch<=start_epoch:raise ValueError('until_epoch must exceed resumed epoch')
    for epoch in range(start_epoch,stop_epoch):
        order=torch.randperm(len(encoded),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        running=0.
        for offset in range(0,len(order),batch):
            group=order[offset:offset+batch];optimizer.zero_grad(set_to_none=True)
            for m in range(0,len(group),micro):
                indices=group[m:m+micro]
                if method=='dpo':
                    pairs=[branch for i in indices for branch in encoded[i]]
                    values,_=log_probabilities(model,pairs,tokenizer.pad_token_id)
                    values=values.view(-1,2)
                    refs=torch.stack([reference[i] for i in indices]).to(values)
                    loss=-F.logsigmoid(cfg['beta']*((values[:,0]-values[:,1])-(refs[:,0]-refs[:,1]))).mean()
                    loss=loss*(len(indices)/len(group))
                else:
                    values,counts=log_probabilities(model,[encoded[i] for i in indices],tokenizer.pad_token_id)
                    if method=='sft':loss=-(values/counts).sum()/len(group)
                    else:
                        # Trainer's per-microbatch causal LM token-mean objective.
                        loss=-values.sum()/counts.sum()*(len(indices)/len(group))
                loss.backward();running+=float(loss.detach())
            if cfg.get('max_grad_norm') is not None:torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'])
            optimizer.step();scheduler.step();step+=1
        if (epoch+1)%cfg.get('checkpoint_every',1)==0 or epoch+1==stop_epoch:
            dest=output/f'epoch_{epoch+1}'
            if dest.exists():raise FileExistsError(dest)
            dest.mkdir()
            if method=='sft':model.save_pretrained(dest/'adapter',safe_serialization=True)
            payload={'metadata':metadata,'method':method,'epoch':epoch+1,'step':step}
            if method=='sn_tune':payload['sparse_deltas']=sparse_state(model)
            elif method=='dpo':payload['ia3_scales']=ia3_state(model)
            elif method=='sft':payload['adapter']='adapter'
            save_tensor(dest/'intervention.pt',payload)
            state={'identity':metadata,'epoch':epoch+1,'step':step,
                   'trainable':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
                   'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                   'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
            save_tensor(dest/'training_state.pt',state)
            save_json(dest/'metadata.json',{**metadata,'epoch':epoch+1,'step':step,
                'optimizer_state':'training_state.pt','mean_step_loss':running/math.ceil(len(rows)/batch),
                'training_precision':'BF16 base, FP32 trainable parameters and loss','evaluation_precision':'BF16'})
        print(f'Epoch {epoch+1}/{cfg["epochs"]}, step {step}',flush=True)
    return model
