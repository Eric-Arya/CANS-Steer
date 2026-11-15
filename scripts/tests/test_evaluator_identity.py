"""Pinned binary judges and explicit model-dependent IFEval scoring."""
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from cans.data_processing import evaluation_config, sha256, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def evaluator():
    spec = importlib.util.spec_from_file_location('identity_evaluator', ROOT/'scripts/evaluate_responses.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mock_judge_loader(module, monkeypatch):
    calls = []
    class Inputs(dict):
        def to(self, device):
            return self
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [1 if text == ' yes' else 2]
        def __call__(self, prompts, **kwargs):
            return Inputs(input_ids=torch.ones(len(prompts), 1, dtype=torch.long))
    class Model:
        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[[0., 2., 0.]]]).repeat(len(input_ids), 1, 1))
    def load(name, *, device, revision):
        calls.append((str(name), revision))
        return Model(), Tokenizer()
    monkeypatch.setattr(module, 'load_model', load)
    return calls


def test_official_truthfulqa_judges_load_pinned_revisions(monkeypatch):
    module = evaluator(); calls = mock_judge_loader(module, monkeypatch)
    cfg = evaluation_config()['judges']['truthfulqa']
    result = module.score_truthfulqa([dict(question='Q?', responses=['answer'])], None, 'cpu')
    assert calls == [(cfg['truth_model'], cfg['truth_revision']), (cfg['info_model'], cfg['info_revision'])]
    assert result['score'] == 100
    assert result['judge_sources']['truth']['revision'] == cfg['truth_revision']
    assert result['judge_sources']['info']['revision'] == cfg['info_revision']


def test_custom_and_local_judges_do_not_inherit_official_revision(tmp_path, monkeypatch):
    module = evaluator(); calls = mock_judge_loader(module, monkeypatch)
    local = tmp_path/'local-truth-judge'; local.mkdir()
    (local/'config.json').write_text('{"model_type":"llama"}\n')
    result = module.score_truthfulqa([dict(question='Q?', responses=['answer'])], str(local), 'cpu', 'custom/info')
    assert calls == [(str(local), None), ('custom/info', None)]
    identity = result['judge_sources']['truth']
    assert identity['source'] == 'local' and identity['revision'] is None
    assert identity['files_sha256']['config.json'] == sha256(local/'config.json')
    assert result['judge_sources']['info']['revision'] is None
    with pytest.raises(ValueError, match='local judge'):
        module.truthfulqa_judge_source(local, 'truth', 'remote-sha')


def test_explicit_remote_judge_revision_is_loaded_and_recorded(monkeypatch):
    module = evaluator(); calls = mock_judge_loader(module, monkeypatch)
    result = module.score_truthfulqa([dict(question='Q?', responses=['answer'])], 'custom/truth', 'cpu',
        'custom/info', judge_revision='truth-sha', info_judge_revision='info-sha')
    assert calls == [('custom/truth', 'truth-sha'), ('custom/info', 'info-sha')]
    assert result['judge_sources']['truth']['revision'] == 'truth-sha'
    assert result['judge_sources']['info']['revision'] == 'info-sha'


def ifeval_files(tmp_path, metadata=None):
    rows = [dict(id='one', key=1, prompt='question', instruction_id_list=['one'], kwargs=[{}],
                 responses=['first answer\n### Instruction: repeated question'])]
    response = tmp_path/'responses.jsonl'; write_jsonl(response, rows)
    write_jsonl(tmp_path/'ifeval.jsonl', rows)
    if metadata is not None:
        response.with_suffix('.metadata.json').write_text(json.dumps(metadata))
    return response, rows


def mock_ifeval_library(monkeypatch):
    calls = []
    @dataclass
    class Result:
        follow_all_instructions: bool
        follow_instruction_list: list
    def score(mode):
        def run(example, answers):
            text = answers[example.prompt]; calls.append((mode, text))
            passed = text == 'first answer'
            return Result(passed, [passed])
        return run
    library = SimpleNamespace(InputExample=lambda **kw: SimpleNamespace(**kw),
        test_instruction_following_strict=score('strict'), test_instruction_following_loose=score('loose'))
    monkeypatch.setitem(sys.modules, 'instruction_following_eval', SimpleNamespace(evaluation_lib=library))
    monkeypatch.setitem(sys.modules, 'langdetect', SimpleNamespace(DetectorFactory=SimpleNamespace(seed=None)))
    return calls


def test_external_ifeval_cli_explicit_model_applies_both_first_response_scorers(tmp_path, monkeypatch):
    module = evaluator(); response, rows = ifeval_files(tmp_path)
    calls = mock_ifeval_library(monkeypatch)
    dest = tmp_path/'summary.json'
    monkeypatch.setattr(sys, 'argv', ['evaluate_responses.py', '--benchmark', 'ifeval',
        '--responses', str(response), '--benchmark-data', str(tmp_path), '--model', 'meta-llama/Llama-2-7b-hf',
        '--external', str(tmp_path), '--output', str(dest)])
    module.main()
    result = json.loads(dest.read_text())
    assert calls == [('strict', 'first answer'), ('loose', 'first answer')]
    assert result['score'] == 100 and result['preprocessing']['truncated_count'] == 1
    scored = json.loads(dest.with_suffix('.scored.jsonl').read_text())
    assert scored['responses'] == rows[0]['responses']


def test_unknown_ifeval_cli_requires_model_instead_of_silent_full_response(tmp_path, monkeypatch):
    module = evaluator(); response, _ = ifeval_files(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['evaluate_responses.py', '--benchmark', 'ifeval',
        '--responses', str(response), '--benchmark-data', str(tmp_path), '--output', str(tmp_path/'summary.json')])
    with pytest.raises(ValueError, match='pass --model'):
        module.main()


def test_ifeval_model_resolution_rejects_policy_conflicts_and_keeps_qwen():
    module = evaluator()
    assert module.resolve_ifeval_model(dict(model='Qwen/Qwen3-4B')) == 'Qwen/Qwen3-4B'
    assert not module.is_llama2_model(module.resolve_ifeval_model({}, 'meta-llama/Meta-Llama-3-8B-Instruct'))
    with pytest.raises(ValueError, match='conflicts'):
        module.resolve_ifeval_model(dict(model='meta-llama/Llama-2-7b-hf'), 'Qwen/Qwen3-4B')
    with pytest.raises(ValueError, match='Conflicting'):
        module.resolve_ifeval_model(dict(model='Qwen/Qwen3-4B', model_source='meta-llama/Llama-2-7b-hf'))
    with pytest.raises(ValueError, match='known model ID'):
        module.resolve_ifeval_model({}, 'unknown/model')


def test_ifeval_model_resolution_supports_native_derived_checkpoint(tmp_path):
    module = evaluator(); checkpoint = tmp_path/'arbitrary-name'; checkpoint.mkdir()
    (checkpoint/'config.json').write_text(json.dumps(dict(model_type='llama', vocab_size=32000,
        max_position_embeddings=4096, rope_scaling=None)))
    assert module.resolve_ifeval_model({}, str(checkpoint)) == str(checkpoint)
    assert module.is_llama2_model(module.resolve_ifeval_model(dict(model='custom/model', model_source=str(checkpoint))))


def test_local_checkpoint_config_takes_priority_over_directory_name(tmp_path):
    module = evaluator(); checkpoint = tmp_path/'llama-2-experiments-qwen'; checkpoint.mkdir()
    (checkpoint/'config.json').write_text(json.dumps(dict(model_type='qwen3', vocab_size=151936,
        max_position_embeddings=40960)))
    assert not module.is_llama2_model(checkpoint)
    assert module.resolve_ifeval_model({}, str(checkpoint)) == str(checkpoint)
