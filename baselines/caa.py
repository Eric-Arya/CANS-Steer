"""Balanced A/B contrastive activation addition at decoder residuals."""
from contextlib import contextmanager
import math
import torch
from .common import prefix

TEMPLATE = '{prompt}\n\nResponses:\n\n(A) {a}\n\n(B) {b}\n\nWhich response would you choose?\nAnswer:'


def candidate_layers(depth):
    return list(range(math.ceil(.3*depth), math.floor(.5*depth)+1, 2))


def choices(tokenizer, row, index, style):
    a,b = row['chosen'],row['rejected']
    good,bad = 'A','B'
    if index % 2:
        a,b,good,bad = b,a,bad,good
    prompt = row['prompt']
    if row.get('domain') == 'toxicity' or (style == 'chat' and row.get('domain') != 'hallucination'):
        prompt,a,b = prompt.strip(),a.strip(),b.strip()
    template = TEMPLATE
    if row.get('domain') == 'hallucination':
        template = 'Question:\n' + TEMPLATE.replace('Which response would you choose?\nAnswer:', 'Which response correctly answers the question?\nAnswer: ')
    elif row.get('domain') == 'toxicity':
        template = TEMPLATE.replace('Which response would you choose?\nAnswer:', 'Which continuation is less toxic?\nAnswer: ')
    content = template.format(prompt=prompt,a=a,b=b)
    text = prefix(tokenizer,content,style)
    ids = [tokenizer.encode(text+label,add_special_tokens=style=='raw') for label in (good,bad)]
    if ids[0][:-1] != ids[1][:-1] or len(ids[0]) != len(ids[1]):
        raise ValueError('CAA branches must differ only at the final contrast token')
    return ids


@torch.inference_mode()
def extract(model, tokenizer, rows, style='chat'):
    layers = candidate_layers(len(model.model.layers))
    totals = {i:torch.zeros(model.config.hidden_size,dtype=torch.float64) for i in layers}
    capture = {}
    handles=[]
    for i in layers:
        def hook(module,inputs,output,index=i):
            value = output[0] if isinstance(output,tuple) else output
            capture[index] = value[0,-1].float().cpu()
        handles.append(model.model.layers[i].register_forward_hook(hook))
    try:
        for n,row in enumerate(rows):
            vectors=[]
            for ids in choices(tokenizer,row,n,style):
                model(input_ids=torch.tensor([ids],device=model.device),use_cache=False)
                vectors.append(dict(capture))
            for i in layers:
                totals[i] += (vectors[0][i]-vectors[1][i]).double()
    finally:
        for handle in handles:handle.remove()
    return {i:(v/len(rows)).float() for i,v in totals.items()}


@contextmanager
def intervention(model, vector, layer, strength=1.):
    def hook(module,inputs,output):
        value = output[0] if isinstance(output,tuple) else output
        # Generation calls one prompt prefill, followed by one-token cached steps.
        value = value.clone()
        value[:,-1] += strength*vector.to(value)
        return (value,*output[1:]) if isinstance(output,tuple) else value
    handle=model.model.layers[layer].register_forward_hook(hook)
    try:yield
    finally:handle.remove()
