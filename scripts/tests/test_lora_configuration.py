"""Configured LoRA dimensions, scaling and dropout reach actual training."""
from copy import deepcopy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from baselines.sft_sn_tune_dpo_training import lora_settings, train


def training_inputs():
    torch.manual_seed(19)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16,
        intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, attention_dropout=0.))
    model.config._attn_implementation = 'eager'
    backend = Tokenizer(WordLevel({'[UNK]':0, '[PAD]':1, '[EOS]':2,
        'question':3, 'answer':4}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
        unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    rows = [dict(id=str(i), prompt='question ', response='answer') for i in range(4)]
    settings = dict(format='raw', target='response', epochs=2, checkpoint_every=1,
        seed=112, learning_rate=.001, weight_decay=0., scheduler='constant',
        effective_batch_size=2, microbatch_size=1, max_grad_norm=None,
        rank=3, lora_alpha=9, dropout=.125)
    return model, tokenizer, rows, settings


def test_configured_adapter_parameters_reach_peft_modules(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    model, tokenizer, rows, settings = training_inputs()
    wrapped = train(model, tokenizer, rows, settings, 'sft', tmp_path,
                    dict(configuration=settings), until_epoch=1)
    adapter = wrapped.peft_config['default']
    assert (adapter.r, adapter.lora_alpha, adapter.lora_dropout) == (3, 9, .125)
    modules = [module for module in wrapped.modules() if hasattr(module, 'lora_A')]
    assert len(modules) == 7
    for module in modules:
        assert module.lora_A['default'].weight.shape[0] == 3
        assert module.lora_B['default'].weight.shape[1] == 3
        assert module.scaling['default'] == 3.
        assert module.lora_dropout['default'].p == .125


def test_nondefault_adapter_resumes_with_identical_weights_and_adamw(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    model, tokenizer, rows, settings = training_inputs()
    identity = dict(configuration=settings)
    train(deepcopy(model), tokenizer, rows, settings, 'sft', tmp_path/'full', identity)
    train(deepcopy(model), tokenizer, rows, settings, 'sft', tmp_path/'resume', identity,
          until_epoch=1)
    train(deepcopy(model), tokenizer, rows, settings, 'sft', tmp_path/'resume', identity,
          resume=tmp_path/'resume/epoch_1')
    expected = torch.load(tmp_path/'full/epoch_2/training_state.pt', weights_only=True)
    actual = torch.load(tmp_path/'resume/epoch_2/training_state.pt', weights_only=True)
    assert (actual['epoch'], actual['step']) == (2, expected['step'])
    for key, value in expected['trainable'].items():
        torch.testing.assert_close(actual['trainable'][key], value, rtol=0, atol=0)
    for parameter, state in expected['optimizer']['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(actual['optimizer']['state'][parameter][key], value,
                                       rtol=0, atol=0)


@pytest.mark.parametrize('field,value', [('rank',0), ('rank',1.5), ('rank',True),
    ('lora_alpha',0), ('lora_alpha',float('nan')), ('lora_alpha',True),
    ('dropout',-0.1), ('dropout',1.), ('dropout',float('inf')), ('dropout',True)])
def test_invalid_adapter_configuration_is_rejected(field, value):
    with pytest.raises(ValueError, match='LoRA'):
        lora_settings({field:value})


def test_older_python_callers_keep_the_recorded_adapter_defaults():
    assert lora_settings({}) == dict(r=8, lora_alpha=16, lora_dropout=0.)
