"""Check portable baseline intervention semantics and frozen data identities."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,Qwen3Config,Qwen3ForCausalLM
from cans.data_processing import read_jsonl,sha256
from baselines import aura,caa,dap,sn_tune,h_neurons
from baselines.common import load_config

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(params=['llama','qwen3'])
def model(request):
    settings=dict(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=3,
                  num_attention_heads=2,num_key_value_heads=1,attention_dropout=0.)
    cfg=LlamaConfig(**settings) if request.param=='llama' else Qwen3Config(**settings,head_dim=8)
    cfg._attn_implementation='eager'
    torch.manual_seed(112)
    cls=LlamaForCausalLM if request.param=='llama' else Qwen3ForCausalLM
    return cls(cfg).eval()


def test_sparse_delta_matches_dense_update():
    for axis in ['row','column']:
        base=torch.nn.Linear(7,5)
        wrapped=sn_tune.SparseNeuronLinear(deepcopy(base),{1,3},axis)
        with torch.no_grad():wrapped.delta.normal_()
        x=torch.randn(2,4,7)
        weight=base.weight.detach().clone()
        weight.index_add_(0 if axis=='row' else 1,wrapped.indices,wrapped.delta.detach())
        torch.testing.assert_close(wrapped(x),torch.nn.functional.linear(x,weight,base.bias),atol=1e-6,rtol=1e-5)
        wrapped(x).sum().backward()
        assert wrapped.delta.grad is not None and wrapped.base.weight.grad is None


def test_ia3_hooks_capture_post_scaling(model):
    original=deepcopy(model);scales=torch.full((3,24),.25)
    dap.apply_ia3(model,scales)
    before={};after={};ids=torch.tensor([[1,3,8]])
    with dap.capture_inputs(original,before),dap.capture_inputs(model,after):
        original(ids);model(ids)
    torch.testing.assert_close(after[0],before[0]*.25)


def test_dap_empty_selection_matches_base_prefill_and_cache(model):
    guide=deepcopy(model);dap.apply_ia3(guide,torch.full((3,24),.7))
    ids=torch.tensor([[1,3,8]])
    with torch.no_grad():
        expected=model.generate(ids,max_new_tokens=3,do_sample=False,pad_token_id=0)
        with dap.intervention(model,guide,torch.tensor([],dtype=torch.long)):
            actual=model.generate(ids,max_new_tokens=3,do_sample=False,pad_token_id=0)
    assert torch.equal(expected,actual)
    assert not model._forward_pre_hooks


def test_dap_full_eligible_selection_matches_guide_without_final_layer_scaling(model):
    guide=deepcopy(model)
    scales=torch.full((3,24),.7);scales[-1]=1
    dap.apply_ia3(guide,scales)
    ids=torch.tensor([[1,3,8]])
    with torch.no_grad():
        expected=guide(ids).logits
        with dap.intervention(model,guide,torch.arange(48)):
            actual=model(ids).logits
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_aura_and_caa_hooks_remove_cleanly(model):
    ids=torch.tensor([[1,4,7]])
    original=model(ids).logits
    scales={name:torch.ones(m.out_features) for name,m in aura.projections(model).items()}
    with aura.intervention(model,scales):torch.testing.assert_close(model(ids).logits,original)
    with caa.intervention(model,torch.zeros(16),1):torch.testing.assert_close(model(ids).logits,original)
    torch.testing.assert_close(model(ids).logits,original)
    torch.testing.assert_close(aura.multipliers(torch.tensor([.2,.5,.75,1.])),torch.tensor([1.,1.,.5,0.]))


def test_sn_importance_supports_llama_and_qwen(model):
    from types import SimpleNamespace
    args=SimpleNamespace(top_attn=4,top_ffn=5,q_chunk=4)
    scorers=[sn_tune.LayerScorer(i,l.self_attn,l.mlp,args,model.config.model_type) for i,l in enumerate(model.model.layers)]
    try:
        with torch.no_grad():model(torch.tensor([[1,4,8]]),use_cache=False)
        rows=[{'layers':{str(i):s.result() for i,s in enumerate(scorers)}}]
        ranked=sn_tune.aggregate_ranked(rows,3)
        selected=sn_tune.cap_selection(ranked,2,8,2)
        assert all(len(v)<=2 for ls in selected.values() for v in ls.values())
        assert all(max(v)<8 for v in selected['k'].values())
    finally:
        for s in scorers:s.close()


def test_released_baseline_data():
    manifest=json.loads((ROOT/'samples/baselines/manifest.json').read_text())
    for name,record in manifest['files'].items():assert sha256(ROOT/name)==record['sha256'],name
    for domain in ('safety','toxicity','hallucination'):
        root,common,spec=load_config(ROOT/f'configs/baselines/qwen3_{domain}.yaml')
        for method,cfg in spec.items():
            if not isinstance(cfg,dict) or 'data' not in cfg:continue
            assert len(read_jsonl(root/cfg['data']))==cfg['count'],(domain,method)
        h=read_jsonl(ROOT/f'samples/baselines/{domain}/h_neurons.jsonl')
        split=json.loads((ROOT/f'samples/baselines/{domain}/h_split.json').read_text())
        h_neurons.validate_split(h,split)
        for r in h:assert r['response'][r['span_start']:r['span_end']]==r['span'] and r['span']
        target=read_jsonl(ROOT/f'samples/{domain}.jsonl')
        required=1 if domain=='toxicity' else 0
        assert {(r['prompt'],r['response'],r['span']) for r in target}=={(r['prompt'],r['response'],r['span']) for r in h if r['label']==required}
        prompts=read_jsonl(ROOT/f'samples/baselines/{domain}/dap_identification.jsonl')
        assert len(prompts)==len({r['prompt'] for r in prompts})==200


def test_h_expansion_has_three_negative_features_per_positive():
    rows=[{'id':str(i),'group':str(i//2),'label':i%2,'target':torch.ones(2,3)*i,'other':torch.zeros(2,3)} for i in range(8)]
    split={'train':[str(i) for i in range(4)],'heldout':[str(i) for i in range(4,8)]}
    x,y,tx,ty=h_neurons.arrays(rows,split)
    assert len(x)==8 and (y==0).sum()==6 and (y==1).sum()==2
    assert len(tx)==4 and ty.sum()==2


def toy_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    inner=Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'question':3,'answer':4,'bad':5,'.':6},unk_token='[UNK]'))
    inner.pre_tokenizer=Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=inner,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')


def test_sft_resume_restores_adamw_and_matches_uninterrupted(model,tmp_path):
    from baselines.sft_sn_tune_dpo_training import train
    cfg={'format':'raw','target':'response','epochs':2,'checkpoint_every':1,'seed':112,
         'learning_rate':1e-3,'weight_decay':0.,'scheduler':'constant','effective_batch_size':2,'microbatch_size':1,'max_grad_norm':None}
    rows=[{'id':str(i),'prompt':'question ','response':'answer'} for i in range(4)]
    tok=toy_tokenizer();initial=deepcopy(model)
    train(deepcopy(initial),tok,rows,cfg,'sft',tmp_path/'full',{'config':cfg})
    train(deepcopy(initial),tok,rows,cfg,'sft',tmp_path/'resume',{'config':cfg},until_epoch=1)
    train(deepcopy(initial),tok,rows,cfg,'sft',tmp_path/'resume',{'config':cfg},resume=tmp_path/'resume/epoch_1')
    expected=torch.load(tmp_path/'full/epoch_2/training_state.pt',weights_only=True)
    actual=torch.load(tmp_path/'resume/epoch_2/training_state.pt',weights_only=True)
    assert actual['epoch']==2 and actual['step']==4 and actual['optimizer']['state']
    for k,v in expected['trainable'].items():torch.testing.assert_close(v,actual['trainable'][k],rtol=0,atol=0)
    for k,state in expected['optimizer']['state'].items():
        for key,v in state.items():torch.testing.assert_close(v,actual['optimizer']['state'][k][key],rtol=0,atol=0)


def test_dpo_trains_only_ia3(model,tmp_path):
    from baselines.sft_sn_tune_dpo_training import train
    original={n:p.detach().clone() for n,p in model.named_parameters()}
    cfg={'format':'raw','epochs':1,'checkpoint_every':1,'seed':112,'learning_rate':.01,
         'weight_decay':.1,'scheduler':'cosine','effective_batch_size':2,'microbatch_size':1,
         'beta':.1,'max_length':32,'max_prompt_length':16,'max_grad_norm':1.}
    rows=[{'id':str(i),'prompt':'question ','chosen':'answer','rejected':'bad'} for i in range(2)]
    train(model,toy_tokenizer(),rows,cfg,'dpo',tmp_path,{'config':cfg})
    obj=torch.load(tmp_path/'epoch_1/intervention.pt',weights_only=True)
    assert obj['ia3_scales'].shape==(3,24) and torch.isfinite(obj['ia3_scales']).all()
    assert not torch.equal(obj['ia3_scales'],torch.ones(3,24))
    for name,param in model.named_parameters():
        if name.endswith('.alpha'):assert param.requires_grad
        else:
            assert not param.requires_grad
            torch.testing.assert_close(param,original[name.replace('.down_proj.base.','.down_proj.')])


def test_aura_vectorized_auroc_matches_sklearn_with_ties():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    labels = np.array([0, 1, 0, 0, 1, 0, 0])
    values = np.random.default_rng(42).integers(-2, 3, size=(7, 101)).astype(float)
    values[:, 0] = 1  # All ties must give AUROC 0.5.
    expected = [roc_auc_score(labels, values[:, j]) for j in range(values.shape[1])]
    np.testing.assert_allclose(aura.auroc_columns(labels, values), expected, atol=1e-15)


def test_dpo_preserves_formatted_prompt_and_formats_raw_prompt_once():
    from baselines.sft_sn_tune_dpo_training import preference_rows
    tok = toy_tokenizer()
    tok.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    raw = {'id': 'pair', 'prompt': 'question', 'chosen': 'answer', 'rejected': 'bad'}
    rendered = tok.apply_chat_template([{'role': 'user', 'content': raw['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    cfg = {'format': 'chat', 'max_length': 64, 'max_prompt_length': 32}
    expected = preference_rows(tok, [raw], cfg)
    actual = preference_rows(tok, [{**raw, 'prompt': rendered}], {**cfg, 'prompt_is_formatted': True})
    assert actual == expected
    for branch, response in zip(actual[0], ('answer', 'bad')):
        targets = [branch['input_ids'][i] for i in branch['target_indices']]
        assert targets == tok.encode(response, add_special_tokens=False) + [tok.eos_token_id]


def test_safety_caa_uses_the_dpo_character_matched_responses():
    import yaml
    cfg = yaml.safe_load((ROOT / 'configs/baselines/qwen3_safety.yaml').read_text())
    assert cfg['dpo']['prompt_is_formatted'] is True
    dpo = {r['id']: r for r in read_jsonl(ROOT / cfg['dpo']['data'])}
    caa_rows = read_jsonl(ROOT / cfg['caa']['data'])
    assert len(caa_rows) == len(dpo) == 256
    for row in caa_rows:
        assert row['chosen'] == dpo[row['id']]['chosen']
        assert row['rejected'] == dpo[row['id']]['rejected']
        assert len(row['chosen']) == len(row['rejected'])
        assert not row['prompt'].startswith('<|im_start|>')
