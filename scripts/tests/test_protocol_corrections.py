"""First-response IFEval scoring and versioned single-BOS Grad extraction."""
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from cans.data_processing import TARGET_TOKENIZATION_POLICY, encode_span, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def script(name):
    spec = importlib.util.spec_from_file_location(f'protocol_correction_{name}', ROOT/f'scripts/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('model,truncate', [('meta-llama/Llama-2-7b-hf',True),
    ('llama2',True), ('Qwen/Qwen3-4B',False), ('meta-llama/Meta-Llama-3-8B-Instruct',False),
    ('meta-llama/Llama-3.1-8B-Instruct',False), (None,False)])
def test_ifeval_preprocesses_both_scorers_and_preserves_originals(tmp_path, monkeypatch, model, truncate):
    module = script('evaluate_responses')
    calls = []
    @dataclass
    class Result:
        follow_all_instructions: bool
        follow_instruction_list: list
    def scorer(mode):
        def run(example, responses):
            text = responses[example.prompt]
            calls.append((mode,text))
            success = text == 'first answer'
            return Result(success,[success])
        return run
    library = SimpleNamespace(InputExample=lambda **kw: SimpleNamespace(**kw),
        test_instruction_following_strict=scorer('strict'),
        test_instruction_following_loose=scorer('loose'))
    monkeypatch.setitem(sys.modules,'instruction_following_eval',SimpleNamespace(evaluation_lib=library))
    monkeypatch.setitem(sys.modules,'langdetect',SimpleNamespace(DetectorFactory=SimpleNamespace(seed=None)))
    rows = [dict(key=i,prompt=f'prompt {i}',instruction_id_list=['one'],kwargs=[{}],responses=[text])
            for i,text in enumerate(['first answer  \n### Instruction: second\n### Instruction: third',
                                     'no delimiter; retain trailing spaces  \n'])]
    before = deepcopy(rows)
    result = module.score_ifeval(rows,tmp_path,model)
    first = 'first answer' if truncate else before[0]['responses'][0]
    assert calls == [('strict',first),('strict',before[1]['responses'][0]),
                     ('loose',first),('loose',before[1]['responses'][0])]
    assert [r['responses'] for r in rows] == [r['responses'] for r in before]
    assert result['strict']['prompt_accuracy'] == (50 if truncate else 0)
    assert result['loose']['prompt_accuracy'] == (50 if truncate else 0)
    assert result['preprocessing']['truncated_count'] == int(truncate)
    assert rows[0]['ifeval_preprocessing']['truncated'] is truncate
    assert rows[1]['ifeval_scoring_text'] == before[1]['responses'][0]


def test_ifeval_cli_detects_native_derived_checkpoint(tmp_path, monkeypatch):
    module = script('evaluate_responses')
    checkpoint = tmp_path/'arbitrary_checkpoint_name';checkpoint.mkdir()
    (checkpoint/'config.json').write_text(json.dumps(dict(model_type='llama',vocab_size=32000,
        max_position_embeddings=4096,rope_scaling=None)))
    assert module.is_llama2_model(checkpoint)
    rows = [dict(id='one')]
    responses = tmp_path/'responses.jsonl';write_jsonl(responses,rows)
    frozen = tmp_path/'ifeval.jsonl';write_jsonl(frozen,rows)
    responses.with_suffix('.metadata.json').write_text(json.dumps(dict(
        model='custom/native-derived',model_source=str(checkpoint))))
    received = []
    def score(rows, external, model=None):
        received.append(model)
        return dict(score=100)
    monkeypatch.setattr(module,'score_ifeval',score)
    monkeypatch.setattr(sys,'argv',['evaluate_responses.py','--benchmark','ifeval','--responses',str(responses),
        '--benchmark-data',str(tmp_path),'--output',str(tmp_path/'summary.json')])
    module.main()
    assert received == [str(checkpoint)]


@pytest.mark.parametrize('style', ['raw','chat'])
def test_encode_span_adds_bos_only_when_absent(style):
    class Tokenizer:
        bos_token = '<bos>'
        def apply_chat_template(self,messages,**kwargs):
            return '<bos>chat: '
        def __call__(self,text,add_special_tokens,return_offsets_mapping):
            assert return_offsets_mapping
            self.seen = (text,add_special_tokens)
            # Character coordinates make target alignment explicit here.
            return dict(input_ids=([1] if add_special_tokens else [])+list(range(10,10+len(text))),
                        offset_mapping=([(0,0)] if add_special_tokens else [])+[(i,i+1) for i in range(len(text))])
        def encode(self,text,add_special_tokens):
            assert not add_special_tokens
            return list(range(100,100+len(text)))
    tokenizer = Tokenizer()
    row = dict(id='one',format=style,prompt='raw: ',response='yes suffix',span='yes',span_start=0,span_end=3)
    encoded = encode_span(tokenizer,row)
    text, add_special = tokenizer.seen
    assert add_special == (style == 'raw')
    assert len(encoded['target_indices']) == 3 and encoded['target_indices'][0] > 0
    assert len(encoded['input_ids']) == len(text)+len(' suffix')+int(add_special)


@pytest.mark.parametrize('environment', ['CANS_TEST_LLAMA3_TOKENIZER','CANS_TEST_LLAMA31_TOKENIZER'])
def test_real_llama_chat_target_encoding_has_one_bos(environment):
    location = os.environ.get(environment)
    if not location:
        pytest.skip(f'Set {environment} to a local tokenizer directory for this integration check')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(location,local_files_only=True)
    encoded = encode_span(tokenizer,dict(id='one',format='chat',prompt='Give a short answer.',
        response='The answer is yes. Extra context.',span='yes',span_start=14,span_end=17))
    assert encoded['input_ids'][0] == tokenizer.bos_token_id
    assert encoded['input_ids'][1] != tokenizer.bos_token_id
    assert len(encoded['target_indices']) > 0
    assert all(0 < i < len(encoded['input_ids']) for i in encoded['target_indices'])


def test_grad_checkpoint_binds_tokenization_and_exact_encoded_inputs(tmp_path, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM
    module = script('extract')
    model = LlamaForCausalLM(LlamaConfig(vocab_size=17,hidden_size=8,intermediate_size=12,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)).eval()
    model.requires_grad_(False)
    rows = [dict(id='one',input_ids=[1,3,5,7],target_indices=[2])]
    write_jsonl(tmp_path/'data.jsonl',[dict(id='one')])
    monkeypatch.setattr(module,'config',lambda _: (tmp_path,
        dict(model='toy',model_config_sha256='toy'),dict(data='data.jsonl')))
    monkeypatch.setattr(module,'validate_model_source',lambda *args: None)
    monkeypatch.setattr(module,'load_model',lambda *args,**kwargs: (model,SimpleNamespace(pad_token_id=0)))
    monkeypatch.setattr(module,'encode_span',lambda *args: deepcopy(rows[0]))
    args = SimpleNamespace(config=Path('toy.yaml'),model=None,device='cpu',output=tmp_path/'gradient.pt',resume=False)
    module.extract_grad(args)
    checkpoint = args.output.with_suffix('.checkpoint.pt')
    state = torch.load(checkpoint,weights_only=True)
    assert state['binding']['tokenization_policy'] == TARGET_TOKENIZATION_POLICY
    assert len(state['binding']['encoded_inputs_sha256']) == 64
    # A historical checkpoint without the new policy cannot be resumed silently.
    historical = deepcopy(state)
    historical['binding'].pop('tokenization_policy')
    historical['binding'].pop('encoded_inputs_sha256')
    torch.save(historical,checkpoint)
    args.resume = True
    with pytest.raises(ValueError,match='different inputs or settings'):
        module.extract_grad(args)
    # The exact input IDs/targets also bind the resume, even with the same corpus.
    torch.save(state,checkpoint)
    rows[0]['target_indices'] = [3]
    with pytest.raises(ValueError,match='different inputs or settings'):
        module.extract_grad(args)
