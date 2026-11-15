"""Full-Fisher cross terms, conditional KL, and resumable cache identity."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from cans.data_processing import sha256
from cans.gradient_fisher_extraction import response_gradients
from cans.model_interventions import fold_scales

SPEC = importlib.util.spec_from_file_location('approximation_script', Path(__file__).resolve().parents[1]/'cans_evaluate_approximation.py')
approx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(approx)


def test_native_reference_prompt_ids_can_repeat_across_continuations(tmp_path):
    path = tmp_path/'reference.jsonl'
    path.write_text('\n'.join(json.dumps(dict(id='shared_prompt', candidate_index=i,
        prompt_token_ids=[1, 2], fisher_token_ids=[3])) for i in range(4))+'\n')
    rows = approx.reference_rows(path, 4, 4096)
    assert [r['id'] for r in rows] == [0, 1, 2, 3]
    assert all(r['target_indices'] == [2] for r in rows)


def test_full_quadratic_retains_cross_neuron_cancellation():
    scores = torch.tensor([[[1., -1.]], [[2., -2.]]])
    deltas = torch.tensor([[[1., 1.]], [[1., -1.]]])
    projections = approx.project_scores(scores, deltas)
    full = projections.square().mean(0)/2
    sample_fisher = scores.flatten(1).double().T @ scores.flatten(1).double()/len(scores)
    explicit = torch.einsum('pi,ij,pj->p',deltas.flatten(1).double(),sample_fisher,deltas.flatten(1).double())/2
    torch.testing.assert_close(full, explicit)
    assert full.tolist() == [0., 5.]
    diagonal = (scores.double().square().mean(0)*deltas.double().square()).flatten(1).sum(1)/2
    assert diagonal.tolist() == [2.5, 2.5]


def test_sequence_scores_and_conditional_kl_on_tiny_llama():
    torch.manual_seed(42)
    cfg = LlamaConfig(vocab_size=17,hidden_size=8,intermediate_size=12,num_hidden_layers=1,
                      num_attention_heads=2,num_key_value_heads=1,attention_dropout=0.)
    cfg._attn_implementation = 'eager'
    base = LlamaForCausalLM(cfg).eval().requires_grad_(False)
    rows = [dict(input_ids=[1,3,4,7],target_indices=[2,3]),
            dict(input_ids=[2,6,9],target_indices=[2]),
            dict(input_ids=[1,2],target_indices=[])]
    scores = response_gradients(base,rows,0)
    first = response_gradients(base,[dict(input_ids=rows[0]['input_ids'],target_indices=[2])],0)
    second = response_gradients(base,[dict(input_ids=rows[0]['input_ids'],target_indices=[3])],0)
    torch.testing.assert_close(scores[0],first[0]+second[0],atol=1e-7,rtol=1e-4)
    delta = torch.linspace(-.03,.03,12).reshape(1,12)
    edited = deepcopy(base)
    fold_scales(edited,1+delta)
    batched = approx.kl_batch(base,edited,rows,0,1)
    separate = torch.cat([approx.kl_batch(base,edited,[r],0,8) for r in rows])
    torch.testing.assert_close(batched,separate,atol=1e-10,rtol=1e-5)
    assert batched[0] > 0 and batched[1] > 0 and batched[2] == 0
    assert torch.equal(approx.kl_batch(base,base,rows,0,2),torch.zeros(3,dtype=torch.float64))


def test_kl_uses_full_vocabulary_and_sums_response_positions():
    class DistributionModel:
        def __init__(self,logits):
            self.logits = torch.tensor(logits)
            self.embedding = SimpleNamespace(weight=torch.zeros(1))
        def get_input_embeddings(self): return self.embedding
        def __call__(self,ids,**kwargs):
            return SimpleNamespace(logits=self.logits.expand(*ids.shape,-1))
    p,q = DistributionModel([1.,2.,3.]),DistributionModel([3.,2.,1.])
    rows = [dict(input_ids=[0,1,2,0],target_indices=[2,3]),
            dict(input_ids=[0,1],target_indices=[])]
    logp,logq = p.logits.double().log_softmax(-1),q.logits.double().log_softmax(-1)
    token = (logp.exp()*(logp-logq)).sum()
    torch.testing.assert_close(approx.kl_batch(p,q,rows,0,1),torch.stack([2*token,token*0]))


def test_cached_scores_reorder_global_ids_and_bind_inputs(tmp_path):
    chunk = tmp_path/'scores.pt'
    scores = torch.tensor([[[2.,3.]],[[1.,4.]]])
    torch.save(dict(gradients=scores,candidate_indices=[1,0],manifest_sha256='a'*64),chunk)
    binding = dict(score_chunks=[dict(path=str(chunk),sha256=sha256(chunk))],
                   score_index=dict(source_manifest_sha256='a'*64))
    args = SimpleNamespace(output=tmp_path,resume=False)
    rows = [dict(candidate_index=0),dict(candidate_index=1)]
    deltas = torch.tensor([[[1.,-1.]],[[2.,1.]]])
    expected = approx.project_scores(scores[[1,0]],deltas).square().mean(0)/2
    actual = approx.fisher_diagnostic(args,{},rows,deltas,binding)
    assert actual == expected.tolist()
    args.resume = True
    assert approx.fisher_diagnostic(args,{},rows,deltas,binding) == actual
    with pytest.raises(ValueError,match='different model'):
        approx.checkpoint(tmp_path/'projected_scores.pt',dict(binding,changed=True),2,True)
    # Point-index binding prevents relabeling one completed KL point as another.
    point = tmp_path/'kl.pt'
    torch.save(dict(binding=dict(binding,point_index=0),values=torch.ones(2,1)),point)
    with pytest.raises(ValueError,match='different model'):
        approx.checkpoint(point,dict(binding,point_index=1),1,True)


def test_cached_scores_reject_duplicate_or_mismatched_continuations(tmp_path):
    chunk = tmp_path/'scores.pt'
    torch.save(dict(gradients=torch.ones(2,1,2),candidate_indices=[0,0],manifest_sha256='a'*64),chunk)
    binding = dict(score_chunks=[dict(path=str(chunk),sha256=sha256(chunk))],
                   score_index=dict(source_manifest_sha256='a'*64))
    args = SimpleNamespace(output=tmp_path,resume=False)
    with pytest.raises(ValueError,match='continuation IDs'):
        approx.fisher_diagnostic(args,{},[dict(candidate_index=0),dict(candidate_index=1)],torch.ones(1,1,2),binding)


def test_identity_metrics_use_no_fitted_regression():
    result = approx.approximation_metrics([1.,2.,4.],[1.,2.,3.])
    assert result['r_squared_identity'] == .5
    assert result['mape_percent'] == pytest.approx(100/9)
    assert approx.approximation_metrics([1.,0.],[1.,0.])['mape_zero_kl_points_excluded'] == 1


def test_score_index_requires_model_reference_and_extraction_manifest(tmp_path):
    chunk = tmp_path/'scores.pt'
    torch.save(dict(gradients=torch.ones(2,1,2),candidate_indices=[0,1],manifest_sha256='a'*64),chunk)
    path = tmp_path/'index.json'
    value = dict(model_config_sha256='model',reference_sha256='reference',source_manifest_sha256='a'*64,
                 chunks=[dict(path=chunk.name,sha256=sha256(chunk))])
    path.write_text(json.dumps(value))
    records,index = approx.score_sources(path,None,'model','reference')
    assert records[0]['path'] == str(chunk)
    with pytest.raises(ValueError,match='another model'):
        approx.score_sources(path,None,'wrong-model','reference')
    with pytest.raises(ValueError,match='reference corpus'):
        approx.score_sources(path,None,'model','wrong-reference')
    with pytest.raises(ValueError,match='requires a trusted'):
        approx.score_sources(None,[chunk],'model','reference')
    torch.save(dict(gradients=torch.ones(2,1,2),candidate_indices=[0,1],manifest_sha256='b'*64),chunk)
    binding = dict(score_chunks=[dict(path=str(chunk),sha256=sha256(chunk))],score_index=index)
    args = SimpleNamespace(output=tmp_path,resume=False)
    with pytest.raises(ValueError,match='another extraction manifest'):
        approx.fisher_diagnostic(args,{},[dict(candidate_index=0),dict(candidate_index=1)],torch.ones(1,1,2),binding)


def test_frozen_binding_is_atomic_for_concurrent_workers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path/'protocol.json'
    value = dict(model='toy',epsilons=list(range(10)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _:approx.frozen_json(path,value),range(32)))
    assert json.loads(path.read_text()) == value
    with pytest.raises(ValueError,match='Frozen inputs'):
        approx.frozen_json(path,dict(value,model='changed'))
