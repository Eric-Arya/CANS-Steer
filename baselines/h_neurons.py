"""Exact-span CETT features and adaptive sparse H-Neurons classifiers."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from cans.model_interventions import down_projections
from .common import encode_response


@torch.inference_mode()
def extract(model,tokenizer,rows,style='raw'):
    captured={};handles=[];target=[];others=[]
    for i,module in enumerate(down_projections(model)):
        def hook(module,inputs,output,layer=i):
            x=inputs[0][0].float().abs()
            weights=module.weight.float().norm(dim=0)
            norm=output[0].float().norm(dim=-1,keepdim=True)
            cett=x*weights/(norm+1e-8)
            target_sum=cett[target].sum(0)
            other_sum=cett.sum(0)-target_sum if len(target)+len(others)==len(cett) else cett[others].sum(0)
            captured[layer]=((target_sum/len(target)).cpu(),(other_sum/len(others)).cpu())
        handles.append(module.register_forward_hook(hook))
    records=[]
    try:
        for row in rows:
            if 'full_input_token_ids' in row:
                ids=row['full_input_token_ids']
                target=list(range(row['full_target_token_start'],row['full_target_token_end']))
                others=[i for a,b in row['other_full_token_ranges'] for i in range(a,b)]
            else:
                enc=encode_response(tokenizer,row,style=style,target='span',add_special_tokens=style=='raw')
                ids,target=enc['input_ids'],enc['target_indices']
                others=sorted(set(range(len(ids)))-set(target))
            if not target or not others or set(target)&set(others):raise ValueError('Invalid CETT regions')
            if min(target+others)<0 or max(target+others)>=len(ids):raise ValueError('CETT token range out of bounds')
            model.model(input_ids=torch.tensor([ids],device=model.device),use_cache=False)
            records.append({'id':row['id'],'label':row['label'],'group':row['group'],
                'target':torch.stack([captured[i][0] for i in sorted(captured)]),
                'other':torch.stack([captured[i][1] for i in sorted(captured)])})
    finally:
        for handle in handles:handle.remove()
    return records


def validate_split(rows,split):
    by_id={r['id']:r for r in rows}
    train,heldout=set(split['train']),set(split['heldout'])
    if train & heldout or train|heldout!=set(by_id):raise ValueError('Split must cover each row exactly once')
    if {by_id[i]['group'] for i in train}&{by_id[i]['group'] for i in heldout}:
        raise ValueError('Source prompt leakage between train and held-out')
    for part in (train,heldout):
        labels=[by_id[i]['label'] for i in part]
        if not labels or labels.count(0)!=labels.count(1):raise ValueError('Split must be label balanced')


def arrays(records,split):
    validate_split(records,split)
    by_id={r['id']:r for r in records}
    xs=[];ys=[]
    # Preserve the paper classifier loader's feature ordering.
    positive=[by_id[i] for i in split['train'] if by_id[i]['label']==1]
    negative=[by_id[i] for i in split['train'] if by_id[i]['label']==0]
    for group,kind,label in [(positive,'target',1),(negative,'target',0),(negative,'other',0),(positive,'other',0)]:
        for row in group:
            xs.append(row[kind].flatten().numpy());ys.append(label)
    test=[by_id[i] for i in split['heldout']]
    return np.stack(xs),np.array(ys),np.stack([r['target'].flatten().numpy() for r in test]),np.array([r['label'] for r in test])


def fit(records,split,lam):
    if lam<=0:raise ValueError('lambda must be positive')
    x,y,tx,ty=arrays(records,split)
    classifier=LogisticRegression(penalty='l1',C=1/lam,solver='liblinear',max_iter=1000,random_state=42)
    classifier.fit(x,y)
    weights=torch.tensor(classifier.coef_[0],dtype=torch.float32)
    selected=torch.where(weights>0)[0]
    return {'lambda':lam,'weights':weights,'intercept':torch.tensor(classifier.intercept_),
            'selected':selected,'k':len(selected),'heldout_accuracy':float(accuracy_score(ty,classifier.predict(tx))),
            'shape':list(records[0]['target'].shape)}


def adaptive_scan(records,split,max_fits=40):
    """Adapt lambda to observed counts, retaining four approximately log-spaced counts."""
    total=records[0]['target'].numel()
    targets=np.geomspace(max(1,total*.00001),max(4,total*.00035),4)
    cache={}
    def trial(lam):
        key=float(f'{lam:.12g}')
        if key not in cache:
            if len(cache)>=max_fits:raise RuntimeError('Adaptive scan budget reached before endpoints were bracketed')
            cache[key]=fit(records,split,key)
        return cache[key]
    low,high=1.,1.
    while trial(low)['k']<targets[-1]:low/=3
    while trial(high)['k']>targets[0]:high*=3
    chosen=[]
    for target in targets:
        a,b=low,high
        for _ in range(6):
            if len(cache)>=max_fits:break
            mid=(a*b)**.5
            found=trial(mid)
            if found['k']>target:a=mid
            else:b=mid
        available=[v for v in cache.values() if v['k']>0 and v['lambda'] not in chosen]
        if not available:raise ValueError('Could not retain four distinct positive-weight candidates')
        best=min(available,key=lambda v:abs(np.log(max(v['k'],1)/target)))
        chosen.append(best['lambda'])
    return {'candidates':[cache[c] for c in chosen],
            'search_history':[{'lambda':v['lambda'],'k':v['k'],'heldout_accuracy':v['heldout_accuracy']} for v in cache.values()],
            'eligible_neurons':total,'target_counts':targets.tolist(),
            'selection_rule':'Joint review of held-out classifier accuracy and domain intervention scores on the frozen domain benchmark selection subset; no capability metrics.'}
