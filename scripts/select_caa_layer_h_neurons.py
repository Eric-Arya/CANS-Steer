"""Scan H-Neurons lambda and record reviewed H-Neurons or CAA selections."""
import argparse
import json
import math
from pathlib import Path
import torch
from cans.data_processing import read_jsonl,save_tensor,sha256
from baselines.common import load_config,save_json
from baselines.h_neurons import adaptive_scan,validate_split,fit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--method',choices=['h_neurons','caa'],required=True)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--scan',action='store_true')
    mode.add_argument('--fit-selected',action='store_true',help='Refit the recorded selected lambda on the frozen training split')
    p.add_argument('--candidate',type=int,help='Index of the reviewed H candidate (zero-based), or CAA layer')
    p.add_argument('--domain-scores',type=Path,help='JSON with selection_data_sha256 and candidates containing domain_score')
    p.add_argument('--reason',help='Joint selection rationale; required for H-Neurons')
    a=p.parse_args();root,common,spec=load_config(a.config);cfg=spec[a.method]
    if a.output.exists():raise FileExistsError(a.output)
    obj=torch.load(a.artifact,weights_only=True,map_location='cpu')
    if obj['metadata']['model']!=common['model']:raise ValueError('Model mismatch')
    if a.fit_selected:
        if a.method!='h_neurons':p.error('--fit-selected applies only to H-Neurons')
        if 'selected_lambda' not in cfg:p.error('This model needs a recorded selected_lambda in its config')
        split=json.loads((root/cfg['split']).read_text())
        result=fit(obj['features'],split,cfg['selected_lambda'])
        result.update(metadata=obj['metadata'],method='h_neurons',split_sha256=sha256(root/cfg['split']),
                      selection={'mode':'refit recorded selected lambda','lambda':cfg['selected_lambda']})
        save_tensor(a.output,result)
        return
    if a.scan:
        if a.method!='h_neurons':p.error('--scan applies only to H-Neurons')
        split=json.loads((root/cfg['split']).read_text())
        result=adaptive_scan(obj['features'],split)
        result.update(metadata=obj['metadata'],method='h_neurons',split_sha256=sha256(root/cfg['split']))
        save_tensor(a.output,result)
        save_json(a.output.with_suffix('.json'),{k:([{t:r[t] for t in ('lambda','k','heldout_accuracy')} for r in v]
                  if k=='candidates' else v) for k,v in result.items()})
        return
    if a.candidate is None or not a.domain_scores:p.error('--candidate and --domain-scores are required')
    scores=json.loads(a.domain_scores.read_text())
    selection=root/cfg['selection_data']
    if scores['selection_data_sha256']!=sha256(selection):raise ValueError('Wrong domain selection set')
    expected=set(range(len(obj['candidates']))) if a.method=='h_neurons' else set(obj['vectors'])
    observed=[r['candidate'] for r in scores['candidates']]
    if set(observed)!=expected or len(observed)!=len(expected):raise ValueError('Score every candidate exactly once')
    if any(not math.isfinite(r['domain_score']) or not 0<=r['domain_score']<=100 for r in scores['candidates']):
        raise ValueError('Domain scores must be finite percentages')
    if a.method=='h_neurons':
        if not a.reason:p.error('--reason must address both held-out accuracy and domain intervention performance')
        rows=read_jsonl(root/cfg['data']);split=json.loads((root/cfg['split']).read_text());validate_split(rows,split)
        chosen=obj['candidates'][a.candidate]
        result={**chosen,'method':'h_neurons','metadata':obj['metadata']}
    else:
        if a.candidate not in obj['vectors']:raise ValueError('Layer is not in candidate vectors')
        chosen_score=next(r for r in scores['candidates'] if r['candidate']==a.candidate)
        if chosen_score['domain_score']!=min(r['domain_score'] for r in scores['candidates']):
            raise ValueError('CAA selects minimum undesirable-behavior rate at unit strength')
        result={'method':'caa','metadata':obj['metadata'],'vector':obj['vectors'][a.candidate],'layer':a.candidate}
    if not any(r['candidate']==a.candidate for r in scores['candidates']):raise ValueError('Chosen candidate lacks domain evaluation')
    result['selection']={'candidate':a.candidate,'reason':a.reason,'domain_scores':scores,
                          'source_artifact_sha256':sha256(a.artifact),'general_capability_used':False}
    save_tensor(a.output,result)

if __name__=='__main__':main()
