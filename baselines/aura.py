"""Response-only projection max pooling and AUROC-based AurA scaling."""
from contextlib import contextmanager
import numpy as np
import torch
from scipy.stats import rankdata
from .common import prefix


def auroc_columns(labels, values):
    """Exact binary AUROC for each column, using average ranks for ties."""
    labels = np.asarray(labels)
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != len(labels) or set(labels) != {0, 1}:
        raise ValueError('AUROC requires two classes and one value row per label')
    if not np.isfinite(values).all():
        raise ValueError('AUROC activations must be finite')
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = len(labels) - n_positive
    ranks = rankdata(values, axis=0, method='average')
    return (ranks[positive].sum(axis=0) - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def projections(model):
    return {f'model.layers.{i}.mlp.{name}':getattr(layer.mlp,name)
            for i,layer in enumerate(model.model.layers) for name in ('gate_proj','up_proj','down_proj')}


def multipliers(auroc):
    return torch.where(auroc>.5,2-2*auroc,torch.ones_like(auroc))


@torch.inference_mode()
def extract(model, tokenizer, rows, style='raw', response_limit=None, text_only=False):
    modules=projections(model)
    pooled={name:[] for name in modules}
    active=[]
    handles=[]
    for name,module in modules.items():
        def hook(module,inputs,output,key=name):
            pooled[key].append(output[0,active].float().amax(0).cpu().numpy())
        handles.append(module.register_forward_hook(hook))
    try:
        for row in rows:
            prompt='' if text_only else prefix(tokenizer,row['prompt'],style)
            response=row['response']
            # Canonicalize chat response exactly as the original AurA serializer.
            if style=='chat':
                messages=[{'role':'user','content':row['prompt']}]
                text=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':response}],tokenize=False,add_generation_prompt=False,enable_thinking=False)
                empty=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':''}],tokenize=False,add_generation_prompt=False,enable_thinking=False)
                if not text.startswith(prompt) or not empty.startswith(prompt):raise ValueError('Chat serialization prefix mismatch')
                suffix=empty[len(prompt):]
                if suffix and not text.endswith(suffix):raise ValueError('Chat response suffix mismatch')
                serialized=text[len(prompt):len(text)-len(suffix) if suffix else len(text)]
                if serialized not in (response,response.strip()):raise ValueError('Chat template rewrote response')
                response=serialized
            else:text=prompt+response
            enc=tokenizer(text,add_special_tokens=style=='raw',return_offsets_mapping=True)
            if text_only and response_limit:
                enc['input_ids']=enc['input_ids'][:response_limit]
                enc['offset_mapping']=enc['offset_mapping'][:response_limit]
            active[:]=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>a and b>len(prompt) and a<len(prompt)+len(response)]
            if text_only:active[:]=list(range(len(enc['input_ids'])))
            elif response_limit:active[:]=active[:response_limit]
            if not active:raise ValueError(f'No response tokens: {row["id"]}')
            model(input_ids=torch.tensor([enc['input_ids']],device=model.device),use_cache=False)
    finally:
        for handle in handles:handle.remove()
    labels=np.array([r['label'] for r in rows])
    if set(labels)!={0,1}:raise ValueError('AurA requires both response classes')
    scores={}
    for name,values in pooled.items():
        array=np.stack(values)
        scores[name]=torch.tensor(auroc_columns(labels,array),dtype=torch.float32)
    return {'aurocs':scores,'multipliers':{k:multipliers(v) for k,v in scores.items()}}


@contextmanager
def intervention(model, scales):
    modules=projections(model)
    if set(modules)!=set(scales):raise ValueError('AurA artifact must cover all projection outputs')
    handles=[]
    try:
        for name,module in modules.items():
            if len(scales[name])!=module.out_features:raise ValueError('AurA coordinate mismatch')
            def hook(module,inputs,output,scale=scales[name]):return output*scale.to(output)
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:handle.remove()
