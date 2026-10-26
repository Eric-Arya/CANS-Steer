"""Shared baseline configuration, tokenization, and provenance."""
import json
from pathlib import Path
import torch
import yaml
from cans.data_processing import read_jsonl, sha256, padded_batch


def load_config(path):
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text())
    root = path.parent.parent.parent
    common = yaml.safe_load((root / spec['model_config']).read_text())
    return root, common, spec


def prefix(tokenizer, prompt, style):
    if style == 'raw':
        return prompt
    if style != 'chat':
        raise ValueError(f'Unknown input style: {style}')
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def encode_response(tokenizer, row, *, style='raw', target='response', add_special_tokens=False,
                    truncate_at_target=False, max_length=None):
    response = row['response']
    start, end = (row['span_start'], row['span_end']) if target == 'span' else (0, len(response))
    if not 0 <= start < end <= len(response):
        raise ValueError(f'Invalid target: {row["id"]}')
    if target == 'span' and response[start:end] != row['span']:
        raise ValueError(f'Span text mismatch: {row["id"]}')
    prompt = prefix(tokenizer, row['prompt'], style)
    text = prompt + (response[:end] if truncate_at_target else response)
    enc = tokenizer(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
    targets = [i for i,(a,b) in enumerate(enc['offset_mapping'])
               if b > a and b > len(prompt)+start and a < len(prompt)+end]
    if not targets or targets[0] == 0:
        raise ValueError(f'Empty/unpredictable target: {row["id"]}')
    if max_length is not None and len(enc['input_ids']) > max_length:
        raise ValueError(f'{row["id"]} exceeds {max_length} tokens; do not silently truncate supervision')
    return {'id':row['id'], 'input_ids':enc['input_ids'], 'target_indices':targets}


def log_probabilities(model, rows, pad_id):
    ids, mask, target = padded_batch(rows, pad_id, model.device)
    position_ids = (mask.cumsum(-1)-1).clamp_min(0)
    logits = model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False).logits[:,:-1].float()
    logp = logits.log_softmax(-1).gather(-1, ids[:,1:,None]).squeeze(-1)
    return (logp * target).sum(-1), target.sum(-1)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def metadata(common, spec, method, sources):
    return {'method':method, 'model':common['model'], 'model_revision':common.get('revision'),
            'model_config_sha256':common['model_config_sha256'], 'domain':spec['domain'],
            'configuration':spec[method], 'sources':{str(p.name):sha256(p) for p in sources}}
