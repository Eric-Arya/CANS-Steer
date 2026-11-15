"""Check shared CLI routing, generation protocols, and release coverage."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'scripts/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('filename,is_baseline', [
    ('model.yaml', False), ('safety.yaml', False), ('baselines/qwen3_safety.yaml', True),
])
def test_generation_accepts_existing_config_layouts(filename, is_baseline):
    module = load_script('generate_responses')
    root, common, spec, actual = module.load_generation_config(ROOT / 'configs' / filename)
    assert root == ROOT and actual == is_baseline
    assert common['model'] == 'Qwen/Qwen3-4B'


@pytest.mark.parametrize('extra', [
    ['--scales', 'scales.pt', '--artifact', 'artifact.pt'],
    ['--artifact', 'artifact.pt'], ['--guide', 'guide.pt'],
])
def test_generation_rejects_incompatible_arguments_before_loading(extra, tmp_path):
    module = load_script('generate_responses')
    with pytest.raises(SystemExit) as err:
        module.main(['--config', str(ROOT / 'configs/model.yaml'), '--benchmark', 'mmlu',
                     '--output', str(tmp_path / 'responses.jsonl'), *extra])
    assert err.value.code == 2


@pytest.mark.parametrize('method', ['grad', 'fisher', 'sn_tune', 'aura', 'aura_jigsaw', 'caa', 'h_neurons', 'dap'])
def test_extraction_dispatch_preserves_method_and_options(monkeypatch, tmp_path, method):
    module = load_script('extract')
    called = []
    for name in ['extract_grad', 'extract_fisher', 'extract_baseline']:
        monkeypatch.setattr(module, name, lambda args, name=name: called.append((name, args)))
    command = ['--method', method, '--output', str(tmp_path / 'artifact.pt')]
    if method != 'fisher':
        command += ['--config', 'configs/safety.yaml' if method == 'grad' else 'configs/baselines/qwen3_safety.yaml']
    if method in ('grad', 'fisher'):
        command += ['--resume']
    if method == 'dap':
        command += ['--guide', 'matching_dpo.pt']
    module.main(command)
    name, args = called.pop()
    assert name == ('extract_' + method if method in ('grad', 'fisher') else 'extract_baseline')
    assert args.method == method
    if method == 'fisher':
        assert args.config == Path('configs/model.yaml') and args.resume
    if method == 'dap':
        assert args.guide == Path('matching_dpo.pt')


@pytest.mark.parametrize('benchmark', ['mmlu', 'rtp'])
def test_transformers_generation_keeps_sampling_and_choice_protocol(monkeypatch, tmp_path, benchmark):
    module = load_script('generate_responses')
    calls = []
    class Tokenizer:
        pad_token_id = 0
        def encode(self, text, **kwargs):
            return [dict(A=1, B=2, C=3, D=4)[text]] if text in 'ABCD' else [7, 8]
        def decode(self, ids, **kwargs):
            return 'A'
    class Model:
        device = 'cpu'
        config = SimpleNamespace(max_position_embeddings=4096)
        def generate(self, ids, **kwargs):
            calls.append(kwargs)
            if benchmark == 'mmlu':
                assert kwargs['prefix_allowed_tokens_fn'](0, ids[0]) == [1, 2, 3, 4]
            return torch.cat([ids, torch.tensor([[1]])], dim=1).repeat(kwargs.get('num_return_sequences', 1), 1)
    monkeypatch.setattr(module, 'load_model', lambda *a, **kw: (Model(), Tokenizer()))
    args = SimpleNamespace(model=None, device='cpu', artifact=None, benchmark=benchmark)
    row = {'id': 'one', 'formatted_prompt': 'frozen prompt'}
    settings = {'max_new_tokens': 1 if benchmark == 'mmlu' else 20, 'seed': 42}
    results, method = module.generate_transformers(args, {'model': 'toy', 'seed': 112}, {}, [row], settings)
    assert method == 'base' and results[0]['formatted_prompt'] == row['formatted_prompt']
    if benchmark == 'mmlu':
        assert results[0]['prediction'] == 0 and calls[0]['do_sample'] is False
    else:
        assert len(results[0]['responses']) == 25
        assert (calls[0]['temperature'], calls[0]['top_p'], calls[0]['top_k']) == (.6, .9, 50)


def test_prepare_release_checks_both_manifests_and_grouped_splits(tmp_path):
    report = tmp_path / 'audit.json'
    load_script('prepare_release').main(['--verify-only', '--output', str(report)])
    data = json.loads(report.read_text())
    manifest = json.loads((Path(__file__).resolve().parents[2] / 'samples/manifest.json').read_text())
    baseline_manifest = json.loads((Path(__file__).resolve().parents[2] / 'samples/baselines/manifest.json').read_text())
    assert data['cans_files'] == len(manifest['files'])
    assert data['baseline_files'] == len(baseline_manifest['files'])
    assert data['verified_files'] == len(manifest['files']) + len(baseline_manifest['files'])
    assert set(data['domains']) == {'safety', 'toxicity', 'hallucination'}
    assert data['domains']['toxicity']['heldout_examples'] == 200


@pytest.mark.parametrize('protocol,expected', [
    ('llama', [True, False, False, False, False]),
    ('qwen3', [True, False, True, True, True]),
])
def test_bbh_uses_model_specific_answer_filter_and_case_matching(protocol, expected):
    pytest.importorskip('lm_eval')
    module = load_script('evaluate_responses')
    rows = [{'task': 'boolean_expressions', 'target': 'True', 'responses': [answer]}
            for answer in ['So the answer is True.', 'So the answer is true.',
                           'True', 'The answer is True.']]
    rows.append({'task': 'boolean_expressions', 'target': 'true',
                 'responses': ['So the answer is True.']})
    summary = module.score_bbh(rows, protocol)
    assert [row['correct'] for row in rows] == expected
    assert summary['score'] == 100 * sum(expected) / len(expected)
    assert summary['case_sensitive'] == (protocol == 'llama')


def test_llama_bbh_filter_covers_all_27_tasks_without_extra_normalization():
    pytest.importorskip('lm_eval')
    module = load_script('evaluate_responses')
    frozen = [json.loads(line) for line in (ROOT / 'samples/benchmarks/bbh.jsonl').read_text().splitlines()]
    tasks = {row['task']: row for row in frozen}
    assert len(tasks) == 27
    rows = [{**row, 'responses': [f"So the answer is {row['target']}."]} for row in tasks.values()]
    assert module.score_bbh(rows, 'llama')['score'] == 100
    # The official filter removes the final period; strict matching must retain
    # any other punctuation rather than applying Qwen's normalization.
    rows = [{'task': 'boolean_expressions', 'target': 'True',
             'responses': ['So the answer is True..']}]
    module.score_bbh(rows, 'llama')
    assert rows[0]['prediction'] == 'True.' and not rows[0]['correct']


@pytest.mark.parametrize('model,protocol', [
    ('Qwen/Qwen3-4B', 'qwen3'), ('meta-llama/Llama-2-7b-hf', 'llama'),
    ('meta-llama/Meta-Llama-3-8B-Instruct', 'llama'),
    ('meta-llama/Llama-3.1-8B-Instruct', 'llama'),
])
def test_bbh_cli_selects_protocol_from_generation_metadata(monkeypatch, tmp_path, model, protocol):
    pytest.importorskip('lm_eval')
    module = load_script('evaluate_responses')
    responses = tmp_path / 'responses.jsonl'
    row = {'id': 'test', 'task': 'boolean_expressions', 'target': 'true',
           'responses': ['So the answer is True.']}
    responses.write_text(json.dumps(row) + '\n')
    (tmp_path / 'bbh.jsonl').write_text(json.dumps({'id': 'test'}) + '\n')
    responses.with_suffix('.metadata.json').write_text(json.dumps({'model': model}))
    output = tmp_path / 'summary.json'
    monkeypatch.setattr('sys.argv', ['evaluate_responses.py', '--benchmark', 'bbh',
                                    '--responses', str(responses), '--benchmark-data', str(tmp_path),
                                    '--output', str(output)])
    module.main()
    summary = json.loads(output.read_text())
    assert summary['score'] == (100 if protocol == 'qwen3' else 0)
    assert summary['answer_filter'] == ('flexible-extract' if protocol == 'qwen3' else 'get-answer')
    assert summary['bbh_protocol'] == ('bbh_cot_zeroshot' if protocol == 'qwen3' else 'bbh_cot_fewshot')


def test_bbh_requires_explicit_protocol_without_recognized_metadata(tmp_path):
    module = load_script('evaluate_responses')
    responses = tmp_path / 'responses.jsonl'
    with pytest.raises(ValueError, match='--bbh-protocol'):
        module.resolve_bbh_protocol(responses, 'auto')
    assert module.resolve_bbh_protocol(responses, 'llama') == 'llama'
    responses.with_suffix('.metadata.json').write_text(json.dumps({'model': 'unknown-model'}))
    with pytest.raises(ValueError, match='--bbh-protocol'):
        module.resolve_bbh_protocol(responses, 'auto')


def test_truthfulqa_primary_metric_is_per_response_truth_and_info(monkeypatch):
    module = load_script('evaluate_responses')
    rows = [{'question': 'A question?', 'responses': [answer]} for answer in
            ['A: useful answer\nQ: another question', 'answer two', 'answer three', 'answer four']]
    calls = []
    def judge(rows, name, device, suffix):
        calls.append((name, suffix, [r['answer_text'] for r in rows]))
        return [.9, .5, .8, .1] if suffix == 'True' else [.7, .1, .6, .9]
    monkeypatch.setattr(module, 'truthfulqa_probabilities', judge)
    result = module.score_truthfulqa(rows, 'truth-local', 'cpu', 'info-local')
    assert result['score'] == result['true_info_percent'] == 50
    assert result['true_percent'] == result['info_percent'] == 75
    assert result['true_info_percent'] != result['true_percent'] * result['info_percent'] / 100
    assert result['true_info_count'] == 2 and result['true_info_failure_percent'] == 50
    assert result['metric'] == 'true_info_percent' and result['higher_is_better']
    assert [r['true_info'] for r in rows] == [True, False, True, False]
    assert calls[0][0:2] == ('truth-local', 'True')
    assert calls[1][0:2] == ('info-local', 'Helpful')
    assert calls[0][2][0] == calls[1][2][0] == 'useful answer'
    assert rows[1]['truthful']


def test_truthfulqa_rejects_incomplete_info_judgments(monkeypatch):
    module = load_script('evaluate_responses')
    monkeypatch.setattr(module, 'truthfulqa_probabilities', lambda rows, name, device, suffix: [.9] if suffix == 'True' else [])
    with pytest.raises(ValueError, match='Incomplete'):
        module.score_truthfulqa([{'question': 'Q?', 'responses': ['A']}], 'truth', 'cpu', 'info')


def test_generation_adds_bos_only_to_unwrapped_prompts():
    module = load_script('generate_responses')
    class Tokenizer:
        bos_token = '<bos>'
        def encode(self,text,add_special_tokens):
            tokens = [1,2] if text.startswith(self.bos_token) else [2]
            return ([1] if add_special_tokens else []) + tokens
    assert module.encode_prompt(Tokenizer(),'<bos>chat') == [1,2]
    assert module.encode_prompt(Tokenizer(),'raw') == [1,2]
