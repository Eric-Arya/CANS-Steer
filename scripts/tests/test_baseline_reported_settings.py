"""Reported checkpoints and frozen H-Neurons selection remain executable."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from cans.data_processing import sha256


def script(name):
    path = Path(__file__).resolve().parents[1]/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reported_checkpoint_preserves_training_ceiling():
    module = script('train_sft_sn_tune_dpo')
    cfg = dict(epochs=30, reported_epoch=6)
    assert module.training_endpoint(cfg, 'sft', reported=True) == 6
    assert module.training_endpoint(cfg, 'sft', until_epoch=12) == 12
    assert module.training_endpoint(cfg, 'sft') is None
    assert cfg['epochs'] == 30
    with pytest.raises(ValueError, match='only to SFT'):
        module.training_endpoint(cfg, 'sn_tune', reported=True)
    with pytest.raises(ValueError, match='training ceiling'):
        module.training_endpoint(dict(epochs=3, reported_epoch=6), 'sft', reported=True)


def test_h_multiplier_uses_config_and_explicit_override():
    module = script('generate_responses')
    cfg = dict(h_neurons=dict(reported_multiplier=.5))
    assert module.configured_strength(None, cfg, 'h_neurons') == .5
    assert module.configured_strength(.75, cfg, 'h_neurons') == .75
    assert module.configured_strength(None, cfg, 'sft') == 1.


def test_selected_classifier_binds_actual_corpus_features_and_coordinates(tmp_path, monkeypatch):
    module = script('select_caa_layer_h_neurons')
    rows = [dict(id=str(i), group=str(i//2), label=i%2) for i in range(4)]
    data = tmp_path/'h.jsonl'; data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    split = tmp_path/'split.json'; split.write_text(json.dumps(dict(train=['0','1'], heldout=['2','3'])))
    artifact = tmp_path/'features.pt'
    metadata = dict(model='toy', sources={data.name:sha256(data)})
    torch.save(dict(metadata=metadata, features=rows), artifact)
    cfg = dict(data=data.name, split=split.name, selected_lambda=.5,
               selected_coordinates=[[0, 1],[1, 2]], reported_multiplier=.5)
    monkeypatch.setattr(module, 'load_config', lambda _: (tmp_path,dict(model='toy'),dict(h_neurons=cfg)))
    calls = []
    def fit(features, actual_split, lam):
        calls.append(lam)
        return dict(selected=torch.tensor([1,5]), k=2, shape=[2,3])
    monkeypatch.setattr(module, 'fit', fit)
    def invoke(output):
        monkeypatch.setattr('sys.argv', ['select', '--config','toy.yaml','--method','h_neurons',
            '--artifact',str(artifact),'--fit-selected','--output',str(output)])
        module.main()
    dest = tmp_path/'selected.pt'; invoke(dest)
    saved = torch.load(dest, weights_only=True)
    assert saved['selection']['features_sha256'] == sha256(artifact)
    assert saved['selection']['corpus_sha256'] == sha256(data)
    assert saved['split_sha256'] == sha256(split)
    assert saved['selection']['reported_multiplier'] == .5
    cfg['selected_coordinates'] = [[0,0],[1,2]]
    with pytest.raises(ValueError, match='recorded selected neuron set'):
        invoke(tmp_path/'wrong_coordinates.pt')
    cfg['selected_coordinates'] = [[0,1],[1,2]]
    data.write_text(data.read_text()+'\n')
    with pytest.raises(ValueError, match='another H-Neurons corpus'):
        invoke(tmp_path/'wrong_corpus.pt')
