"""Train SFT, SN-Tune, or the IA3-DPO guide using frozen baseline data."""
import argparse
from pathlib import Path
import torch
from cans.data_processing import read_jsonl
from cans.model_interventions import load_model,validate_model_source
from baselines.common import load_config,metadata
from baselines.sft_sn_tune_dpo_training import train


def training_endpoint(cfg, method, until_epoch=None, reported=False):
    """Separate the paper's reported checkpoint from the training ceiling."""
    if reported:
        if method != 'sft':
            raise ValueError('--reported-checkpoint applies only to SFT')
        epoch = cfg.get('reported_epoch')
        if not isinstance(epoch, int) or not 0 < epoch <= cfg['epochs']:
            raise ValueError('reported_epoch must be within the configured training ceiling')
        return epoch
    return until_epoch


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--method',choices=['sft','sn_tune','dpo'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model');p.add_argument('--device',default='cuda:0')
    p.add_argument('--selection',type=Path,help='SN-Tune identification artifact')
    endpoint=p.add_mutually_exclusive_group()
    endpoint.add_argument('--until-epoch',type=int,help='Pause at this epoch to compare the SFT undesirable-behavior rate with the matching CANS point')
    endpoint.add_argument('--reported-checkpoint',action='store_true',help='Train SFT up to reported_epoch from YAML; epochs remains the training ceiling')
    p.add_argument('--resume',type=Path,help='Epoch directory containing training_state.pt')
    a=p.parse_args(argv);root,common,spec=load_config(a.config);cfg=spec[a.method]
    until_epoch=training_endpoint(cfg,a.method,a.until_epoch,a.reported_checkpoint)
    source=root/cfg['data'];rows=read_jsonl(source)
    if len(rows)!=cfg['count']:raise ValueError('Frozen training count mismatch')
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:raise FileExistsError(a.output)
    validate_model_source(a.model or common['model'],common['model_config_sha256'])
    model,tok=load_model(a.model or common['model'],device=a.device,revision=common.get('revision'))
    selection=None;sources=[source]
    if a.method=='sn_tune':
        if not a.selection:p.error('--selection is required for SN-Tune')
        obj=torch.load(a.selection,weights_only=True,map_location='cpu')
        if obj['metadata']['model']!=common['model']:raise ValueError('Selection model mismatch')
        selection=obj['selection'];sources.append(a.selection)
    train(model,tok,rows,cfg,a.method,a.output,metadata(common,spec,a.method,sources),selection,a.resume,until_epoch)

if __name__=='__main__':main()
