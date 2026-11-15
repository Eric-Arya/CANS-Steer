"""Portable frozen MBPP/MGSM protocols and complete five-task aggregation."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unicodedata

from .data_processing import read_jsonl, sha256

DATA = Path(__file__).resolve().parents[1] / 'samples/benchmarks'
EXTENSIONS = ('mbpp', 'mgsm')


def resolve_task(common, task, settings):
    """Resolve only the published task/model overrides, leaving shared defaults intact."""
    common, settings = deepcopy(common), deepcopy(settings)
    if task in EXTENSIONS:
        native = 'llama-2' in common['model'].lower()
        overrides = settings.pop('model_overrides', {})
        settings.update(overrides.get('llama2' if native else 'chat', {}))
        common['max_model_len'] = settings['max_model_len']
    return common, settings


def chat(tokenizer, messages):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError as exc:
        if 'enable_thinking' not in str(exc):
            raise
        return tokenizer.apply_chat_template(messages, **kwargs)


def validate_extension_rows(rows, task):
    expected = read_jsonl(DATA/f'{task}.jsonl')
    fields = ('id','task_id','text','test_list','test_setup_code') if task == 'mbpp' else ('id','language','source_index','question','answer')
    if len(rows) != len(expected) or any(any(row.get(k) != gold.get(k) for k in fields)
            for row,gold in zip(rows,expected)):
        raise ValueError(f'{task} requires the complete unchanged official frozen benchmark')


def prepare_prompts(tokenizer, rows, task, settings):
    """Render model-matched prompts from unchanged frozen examples."""
    if task not in EXTENSIONS:
        return rows
    validate_extension_rows(rows, task)
    raw = settings['prompt_style'] == 'raw'
    examples = (read_jsonl(DATA/'mbpp_demonstrations.jsonl') if task == 'mbpp'
                else json.loads((DATA/'mgsm_exemplars.json').read_text()))
    result = []
    for row in rows:
        if task == 'mbpp':
            def instruction(r):
                return ('You are an expert Python programmer, and here is your task: ' + r['text']
                    + '\nYour code should pass these tests:\n\n'
                    + '\n'.join(r['test_list'][:settings['visible_tests']])
                    + '\nReturn only Python code between [BEGIN] and [DONE].')
            by_id = {r['task_id']: r for r in examples}
            messages = []
            demo_ids = settings['demonstration_task_ids']
            for tid in demo_ids:
                demo = by_id[tid]
                messages.extend([dict(role='user', content=instruction(demo)),
                    dict(role='assistant', content='[BEGIN]\n'+demo['code']+'\n[DONE]')])
            messages.append(dict(role='user', content=instruction(row)))
            text = ('\n\n'.join(m['content'] for m in messages) + '\n[BEGIN]\n' if raw
                    else chat(tokenizer, messages) + '[BEGIN]\n')
            count = len(demo_ids)
        else:
            shots = examples[row['language']]
            count = settings.get('demonstrations_by_language', {}).get(row['language'], settings['num_fewshot'])
            question = re.match(r'.*?[:：]', shots['1']['q']).group(0)
            answer = re.match(r'.*?[:：]', shots['1']['a']).group(0)
            parts = [shots[str(i)]['q']+'\n'+shots[str(i)]['a'] for i in range(1,count+1)]
            parts.append(question+' '+row['question']+'\n'+answer)
            text = '\n\n'.join(parts)
            if not raw:
                text = chat(tokenizer, [dict(role='user',content=text)])
        result.append(dict(row, formatted_prompt=text, demonstration_count=count))
    return result


def response_budget(row, prompt_tokens, task, common, settings):
    remaining = common['max_model_len'] - prompt_tokens
    if remaining < 1:
        raise ValueError('Frozen prompt leaves no response capacity; do not truncate it')
    limit = settings['max_new_tokens']
    if task in EXTENSIONS and remaining < limit:
        if not (task == 'mgsm' and settings['prompt_style'] == 'raw' and row['language'] == 'te'):
            raise ValueError(f'{task} prompt needs {prompt_tokens}+{limit} tokens; no prompt truncation allowed')
    return min(limit,remaining)


def numeric_answer(text):
    normalized = ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in str(text))
    matches = re.findall(r'[-+−]?\d+(?:[,，]\d{3})*(?:\.\d+)?', normalized)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(',', '').replace('，', '').replace('−', '-'))
    except InvalidOperation:
        return None


MBPP_EXECUTOR = '''import contextlib, io, json, resource, sys
row = json.load(sys.stdin)
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
namespace = {}
results = []
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(row['completion'], '<mbpp-solution>', 'exec'), namespace)
        exec(compile(row.get('test_setup_code', ''), '<mbpp-setup>', 'exec'), namespace)
        for test in row['test_list']:
            try:
                exec(compile(test, '<mbpp-test>', 'exec'), namespace)
                results.append({'passed': True, 'result': 'passed'})
            except BaseException as exc:
                results.append({'passed': False, 'result': type(exc).__name__ + ': ' + str(exc)})
    result = {'passed': all(r['passed'] for r in results), 'tests': results}
except BaseException as exc:
    result = {'passed': False, 'tests': results, 'result': type(exc).__name__ + ': ' + str(exc)}
print(json.dumps(result))
'''


def mbpp_check(pair, timeout):
    row, text = pair
    if len(row['test_list']) != 3:
        raise ValueError('MBPP requires all three original test assertions')
    payload = dict(completion=text, test_setup_code=row.get('test_setup_code',''), test_list=row['test_list'])
    env = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PATH='/usr/bin:/bin')
    with tempfile.TemporaryDirectory(prefix='mbpp-native-') as cwd:
        try:
            proc = subprocess.run([sys.executable,'-I','-c',MBPP_EXECUTOR], input=json.dumps(payload),
                capture_output=True,text=True,cwd=cwd,env=env,timeout=timeout)
            return json.loads(proc.stdout) if proc.returncode == 0 else dict(passed=False,tests=[],result=f'exit {proc.returncode}')
        except subprocess.TimeoutExpired:
            return dict(passed=False,tests=[],result='timed out')


def score_extension(rows, frozen, task, settings):
    """Score saved text against frozen gold data, never generated copies of gold fields."""
    validate_extension_rows(frozen, task)
    if len(rows) != len(frozen):
        raise ValueError('Response/gold count mismatch')
    if task == 'mbpp':
        with ThreadPoolExecutor(max_workers=settings['workers']) as pool:
            results = list(pool.map(lambda pair: mbpp_check(pair,settings['timeout_seconds']),
                zip(frozen,(r['responses'][0] for r in rows),strict=True)))
        for row, result in zip(rows,results,strict=True):
            row['evaluation'] = result
        value = 100*sum(r['passed'] for r in results)/len(results)
        return dict(score=value,pass_at_1_percent=value,metric='pass_at_1_percent',higher_is_better=True)
    for row,gold in zip(rows,frozen,strict=True):
        answer = numeric_answer(row['responses'][0]); target = numeric_answer(gold['answer'])
        row.update(extracted_answer=str(answer) if answer is not None else None,correct=answer is not None and answer == target)
    languages = sorted({r['language'] for r in frozen})
    groups = {lang:[r for r,g in zip(rows,frozen,strict=True) if g['language']==lang] for lang in languages}
    per_language = {lang:100*sum(r['correct'] for r in group)/len(group) for lang,group in groups.items()}
    value = sum(per_language.values())/len(per_language)
    return dict(score=value,macro_accuracy_percent=value,accuracy_percent=100*sum(r['correct'] for r in rows)/len(rows),
        per_language=per_language,metric='macro_accuracy_percent',higher_is_better=True)


def aggregate_capability(paths, settings):
    """Require all five complete components for the same model and intervention."""
    components = {}; identity = None
    for path in paths:
        path = Path(path); record = json.loads(path.read_text()); task = record.get('benchmark')
        if task not in settings['tasks'] or task in components:
            raise ValueError('Supply each of the five capability benchmarks exactly once')
        if record.get('scope') != 'complete benchmark' or record.get('count') != settings['counts'][task]:
            raise ValueError(f'Incomplete capability benchmark: {task}')
        current = record.get('generation_identity')
        if not current or not all(current.get(key) for key in ('model','model_source','model_config_sha256')):
            raise ValueError('Summary lacks a verifiable generation identity')
        if identity is not None and current != identity:
            raise ValueError('Capability components belong to different model/intervention settings')
        identity = current
        components[task] = dict(score=record['score'],source=str(path),sha256=sha256(path),count=record['count'])
    if set(components) != set(settings['tasks']):
        raise ValueError('All five complete capability benchmarks are required')
    return dict(score=sum(c['score'] for c in components.values())/len(components),metric='capability_average5_percent',
        higher_is_better=True,components=components,generation_identity=identity,complete=True)
