"""MLP-only IA3 guide and two-stream dynamic activation patching (DAP)."""
from contextlib import contextmanager
import torch
from torch import nn
from cans.model_interventions import down_projections
from .common import prefix


class IA3Input(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base=base
        self.base.requires_grad_(False)
        self.alpha=nn.Parameter(torch.ones(base.in_features,device=base.weight.device,dtype=torch.float32))
    def forward(self,x):return self.base(x*self.alpha.to(x))


def install_ia3(model):
    model.requires_grad_(False)
    for layer in model.model.layers:layer.mlp.down_proj=IA3Input(layer.mlp.down_proj)


def ia3_state(model):
    return torch.stack([l.mlp.down_proj.alpha.detach().cpu() for l in model.model.layers])


def apply_ia3(model, scales):
    modules=down_projections(model)
    if tuple(scales.shape)!=(len(modules),modules[0].in_features):raise ValueError('IA3 shape mismatch')
    if getattr(model, '_dap_ia3_handles', None):raise ValueError('IA3 guide already installed')
    model._dap_ia3_handles=[]
    for module,scale in zip(modules,scales):
        def hook(module,inputs,alpha=scale):return (inputs[0]*alpha.to(inputs[0]),*inputs[1:])
        model._dap_ia3_handles.append(module.register_forward_pre_hook(hook))


@contextmanager
def capture_inputs(model, target):
    handles=[]
    try:
        for i,module in enumerate(down_projections(model)[:-1]):
            def hook(module,inputs,layer=i):target[layer]=inputs[0].detach()
            handles.append(module.register_forward_pre_hook(hook))
        yield
    finally:
        for handle in handles:handle.remove()


@torch.inference_mode()
def identify(base, guide, tokenizer, rows, style='chat', max_new_tokens=256):
    if len(rows)!=200 or len({r['prompt'] for r in rows})!=200:
        raise ValueError('DAP requires exactly 200 unique frozen prompts')
    depth=len(down_projections(base))-1
    width=down_projections(base)[0].in_features
    square=torch.zeros(depth,width,dtype=torch.float64)
    count=0; completions=[]; left={};right={}
    with capture_inputs(base,left),capture_inputs(guide,right):
        for row in rows:
            text=prefix(tokenizer,row['prompt'],style)
            ids=torch.tensor([tokenizer.encode(text,add_special_tokens=style=='raw')],device=base.device)
            generated=base.generate(ids,max_new_tokens=max_new_tokens,do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id,use_cache=True)
            decoded_prompt=tokenizer.decode(ids[0],skip_special_tokens=True)
            decoded=tokenizer.decode(generated[0],skip_special_tokens=True)
            if not decoded.startswith(decoded_prompt):raise ValueError('Decoded prompt prefix mismatch')
            prompt_ids=tokenizer.encode(decoded_prompt,add_special_tokens=style=='raw')
            response_ids=tokenizer.encode(decoded[len(decoded_prompt):],add_special_tokens=False)
            generated=torch.tensor([prompt_ids+response_ids],device=base.device)
            n=len(prompt_ids)
            base(generated,use_cache=False);guide(generated.to(guide.device),use_cache=False)
            tokens=generated.shape[1]-n
            if not tokens:raise ValueError('Empty DAP continuation')
            for i in range(depth):
                difference=right[i][0,n:].float().cpu()-left[i][0,n:].float().cpu()
                square[i]+=difference.double().square().sum(0)
            count+=tokens
            completions.append({'id':row['id'],'prompt_token_ids':prompt_ids,
                                'response_token_ids':generated[0,n:].tolist()})
    scores=(square/count).sqrt().float()
    ranking=torch.argsort(scores.flatten(),descending=True,stable=True)
    k=scores.numel()*5//100
    return {'scores':scores,'ranking':ranking,'selected':ranking[:k],
            'eligible_neurons':scores.numel(),'k':k,'completion_tokens':count,'completions':completions}


@contextmanager
def intervention(base, guide, selected):
    """Advance an independent guide KV cache on the exact tokens seen by the base."""
    modules=down_projections(base)
    width=modules[0].in_features
    coordinates={i:[] for i in range(len(modules)-1)}
    for index in selected.tolist():
        layer,neuron=divmod(index,width)
        if layer not in coordinates:raise ValueError('DAP cannot patch the final layer')
        coordinates[layer].append(neuron)
    captured={};state={'cache':None};handles=[]
    def guide_forward(module,args,kwargs):
        cache_position=kwargs.get('cache_position')
        if cache_position is not None and int(cache_position[0])==0:state['cache']=None
        ids=kwargs.get('input_ids',args[0] if args else None)
        if ids is None:raise ValueError('DAP requires token input_ids')
        gkwargs={k:v for k,v in kwargs.items() if k in ('attention_mask','position_ids','cache_position')}
        gkwargs={k:v.to(guide.device) if torch.is_tensor(v) else v for k,v in gkwargs.items()}
        result=guide(input_ids=ids.to(guide.device),past_key_values=state['cache'],use_cache=True,**gkwargs)
        state['cache']=result.past_key_values
    with capture_inputs(guide,captured):
        try:
            handles.append(base.register_forward_pre_hook(guide_forward,with_kwargs=True))
            for i,indices in coordinates.items():
                if not indices:continue
                def patch(module,inputs,layer=i,cols=indices):
                    x=inputs[0].clone()
                    reference=captured[layer].to(x)
                    if reference.shape!=x.shape:raise ValueError('DAP streams are not aligned')
                    x[...,cols]=reference[...,cols]
                    return (x,*inputs[1:])
                handles.append(modules[i].register_forward_pre_hook(patch))
            yield
        finally:
            for handle in handles:handle.remove()
