"""Portable capability contracts: frozen samples, native prompts, gold scoring and aggregation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from cans.capability_benchmarks import (
    DATA, aggregate_capability, mbpp_check, numeric_answer, prepare_prompts,
    resolve_task, response_budget, validate_extension_rows,
)
from cans.data_processing import evaluation_config, read_jsonl


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return '\n'.join(f"{r['role']}: {r['content']}" for r in messages) + '\nassistant: '


def settings(model, task):
    return resolve_task(dict(model=model, max_model_len=4096), task, evaluation_config()['tasks'][task])


def test_full_frozen_datasets_and_alignment():
    mbpp = read_jsonl(DATA/'mbpp.jsonl')
    assert {r['task_id'] for r in mbpp} == set(range(11,511))
    assert all(len(r['test_list']) == 3 for r in mbpp)
    mgsm = read_jsonl(DATA/'mgsm.jsonl')
    assert len(mgsm) == 550
    languages = {r['language'] for r in mgsm}
    assert len(languages) == 11
    reference = [(r['source_index'], numeric_answer(r['answer'])) for r in mgsm if r['language'] == 'en']
    for lang in languages:
        assert [(r['source_index'],numeric_answer(r['answer'])) for r in mgsm if r['language'] == lang] == reference
    with pytest.raises(ValueError,match='complete unchanged'):
        validate_extension_rows(mbpp[:200], 'mbpp')


@pytest.mark.parametrize('model,context,shots,visible', [('Qwen/Qwen3-4B',8192,3,3),('meta-llama/Llama-2-7b-hf',4096,1,1)])
def test_mbpp_model_specific_prompts(model,context,shots,visible):
    common, cfg = settings(model,'mbpp')
    rows = read_jsonl(DATA/'mbpp.jsonl')
    prompts = prepare_prompts(Tokenizer(),rows,'mbpp',cfg)
    assert common['max_model_len'] == context
    assert all(r['demonstration_count'] == shots for r in prompts)
    first = prompts[0]['formatted_prompt']
    for test in rows[0]['test_list'][:visible]:
        assert test in first
    if visible == 1:
        assert rows[0]['test_list'][1] not in first
    assert first.endswith('[BEGIN]\n')
    assert rows[0]['code'] not in first


def test_mgsm_native_labels_and_telugu_context_exception():
    rows = read_jsonl(DATA/'mgsm.jsonl')
    common,cfg = settings('meta-llama/Llama-2-7b-hf','mgsm')
    prompts = prepare_prompts(Tokenizer(),rows,'mgsm',cfg)
    counts = {r['language']:r['demonstration_count'] for r in prompts}
    assert counts == dict(bn=3,de=6,en=6,es=6,fr=6,ja=6,ru=6,sw=6,te=1,th=4,zh=6)
    assert len(cfg['stop']) == 11
    assert response_budget(dict(language='te'),2200,'mgsm',common,cfg) == 1896
    with pytest.raises(ValueError,match='no prompt truncation'):
        response_budget(dict(language='bn'),2200,'mgsm',common,cfg)
    other,chatcfg = settings('Qwen/Qwen3-4B','mgsm')
    assert other['max_model_len'] == 8192 and chatcfg['stop'] == []
    assert all(r['demonstration_count'] == 6 for r in prepare_prompts(Tokenizer(),rows,'mgsm',chatcfg))


def test_numeric_extraction_and_native_mbpp_all_assertions():
    assert numeric_answer('work 7; answer −১,২৩৪') == -1234
    assert numeric_answer('No numeric answer') is None
    row = dict(test_setup_code='offset=3',test_list=['assert f(1)==4','assert f(2)==5','assert f(3)==6'])
    assert mbpp_check((row,'def f(x): return x+offset'),1)['passed']
    result = mbpp_check((row,'def f(x): return 4'),1)
    assert not result['passed'] and len(result['tests']) == 3


def test_aggregate_requires_complete_same_intervention(tmp_path):
    cfg = evaluation_config()['capability_aggregate']; paths=[]
    identity = dict(model='Qwen/Qwen3-4B',model_source='/models/checkpoint',model_config_sha256='config',scales_sha256='one')
    for task in cfg['tasks']:
        path = tmp_path/f'{task}.json'
        path.write_text(json.dumps(dict(benchmark=task,scope='complete benchmark',count=cfg['counts'][task],score=50,generation_identity=identity)))
        paths.append(path)
    assert aggregate_capability(paths,cfg)['score'] == 50
    with pytest.raises(ValueError,match='All five'):
        aggregate_capability(paths[:-1],cfg)
    record=json.loads(paths[-1].read_text());record['generation_identity']['scales_sha256']='other'
    paths[-1].write_text(json.dumps(record))
    with pytest.raises(ValueError,match='different model/intervention'):
        aggregate_capability(paths,cfg)
    record['generation_identity']=identity;record['count']=1;paths[-1].write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Incomplete'):
        aggregate_capability(paths,cfg)


def test_scoring_binds_identity_to_response_hash_and_candidate(tmp_path,monkeypatch):
    import importlib.util
    import sys
    from cans.data_processing import sha256
    source = Path(__file__).resolve().parents[1]/'evaluate_responses.py'
    spec=importlib.util.spec_from_file_location('capability_evaluator',source)
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    responses=tmp_path/'responses.jsonl';responses.write_text('{"id":"one","responses":["A"]}\n')
    frozen=tmp_path/'frozen.jsonl';frozen.write_text('{"id":"one"}\n')
    meta=responses.with_suffix('.metadata.json')
    record=dict(model='model',responses_sha256='wrong',layer=12,candidate=7,artifact_sha256='same')
    meta.write_text(json.dumps(record));output=tmp_path/'summary.json'
    monkeypatch.setattr(sys,'argv',['evaluate_responses.py','--benchmark','mmlu','--responses',str(responses),
        '--benchmark-file',str(frozen),'--output',str(output)])
    monkeypatch.setattr(evaluator,'score_mmlu',lambda rows:dict(score=50))
    with pytest.raises(ValueError,match='different response bytes'):
        evaluator.main()
    record['responses_sha256']=sha256(responses);meta.write_text(json.dumps(record))
    evaluator.main();summary=json.loads(output.read_text())
    assert summary['generation_identity']['layer']==12 and summary['generation_identity']['candidate']==7


def test_vllm_uses_per_prompt_native_mgsm_budgets_and_stops(monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace
    source = Path(__file__).resolve().parents[1]/'generate_responses.py'
    spec=importlib.util.spec_from_file_location('capability_generator',source)
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    calls=[]
    class NativeTokenizer:
        bos_token='<s>'
        def encode(self,text,**kwargs):
            return [1]*2200 if 'ప్రశ్న:' in text else [1]*20
    class Engine:
        def __init__(self,**kwargs):
            assert kwargs['max_model_len']==4096
        def get_tokenizer(self):return NativeTokenizer()
        def generate(self,prompts,parameters,**kwargs):
            assert isinstance(parameters,list)
            calls.extend(parameters)
            return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'],outputs=[SimpleNamespace(text='0',token_ids=[1],finish_reason='stop')]) for p in prompts]
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(LLM=Engine,SamplingParams=lambda **kwargs:SimpleNamespace(**kwargs)))
    common=dict(model='meta-llama/Llama-2-7b-hf',seed=112,max_model_len=4096,evaluation_runtime=evaluation_config()['runtime'])
    cfg=evaluation_config()['tasks']['mgsm'];rows=read_jsonl(DATA/'mgsm.jsonl')
    result=generator.generate_vllm(SimpleNamespace(model=None,scales=None,benchmark='mgsm'),common,rows,cfg)
    assert len(result)==550
    for row,params in zip(result,calls):
        assert len(params.stop)==11
        assert params.max_tokens==(1896 if row['language']=='te' else 2048)
        assert row['context_limited']==(row['language']=='te')


def test_aggregate_rejects_different_derived_model_sources(tmp_path):
    cfg = evaluation_config()['capability_aggregate']; paths=[]
    for task in cfg['tasks']:
        path=tmp_path/f'{task}.json'
        identity=dict(model='canonical/model',model_source='/models/derived-a',model_config_sha256='same-config')
        if task=='mgsm': identity['model_source']='/models/derived-b'
        path.write_text(json.dumps(dict(benchmark=task,scope='complete benchmark',count=cfg['counts'][task],score=50,generation_identity=identity)))
        paths.append(path)
    with pytest.raises(ValueError,match='different model/intervention'):
        aggregate_capability(paths,cfg)
