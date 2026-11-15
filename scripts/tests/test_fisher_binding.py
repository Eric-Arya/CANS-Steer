"""Fisher artifacts must describe the model and exact reference being used."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from cans.data_processing import config, sha256, validate_fisher_binding


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fisher_case(tmp_path):
    reference = tmp_path/'reference.jsonl'
    reference.write_text('{}\n{}\n')
    common = dict(model_config_sha256='model', shape=[2, 3],
                  reference_generation=dict(num_prompts=1, continuations_per_prompt=2))
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    metadata = dict(model_config_sha256='model', estimator='whole-response-logprob-gradient-square',
                    scope='global/all-token', model_dtype='float32', accumulator_dtype='float64',
                    data_sha256=sha256(reference), count=2, shape=[2, 3], coordinate_order='layer-major')
    return common, reference, dict(fisher=tensor, fisher_fp64=tensor.double(), metadata=metadata)


def test_same_model_fisher_from_another_corpus_is_rejected(fisher_case, tmp_path):
    common, reference, artifact = fisher_case
    binding = validate_fisher_binding(artifact, common, reference)
    assert binding['reference_sha256'] == sha256(reference)
    other = tmp_path/'other_corpus.jsonl'; other.write_text('{"corpus":"tulu"}\n{}\n')
    artifact['metadata']['data_sha256'] = sha256(other)
    with pytest.raises(ValueError, match='another reference corpus'):
        validate_fisher_binding(artifact, common, reference)


@pytest.mark.parametrize('field,value,message', [
    ('model_config_sha256', 'another-model', 'another model'),
    ('scope', 'target-only', 'global/all-token'),
    ('model_dtype', 'bfloat16', 'FP32 model'),
    ('accumulator_dtype', 'float32', 'FP64 accumulation'),
    ('count', 1, 'continuation count'),
    ('shape', [3, 2], 'metadata shape'),
    ('estimator', 'token-gradient-square-sum', 'whole-response Fisher'),
])
def test_incompatible_extraction_settings_are_rejected(fisher_case, field, value, message):
    common, reference, artifact = fisher_case
    artifact['metadata'][field] = value
    with pytest.raises(ValueError, match=message):
        validate_fisher_binding(artifact, common, reference)


def test_tensor_shape_and_companion_must_match(fisher_case):
    common, reference, artifact = fisher_case
    wrong = deepcopy(artifact); wrong['fisher'] = artifact['fisher'].T
    with pytest.raises(ValueError, match='tensor shape'):
        validate_fisher_binding(wrong, common, reference)
    wrong = deepcopy(artifact); wrong['fisher_fp64'][0, 0] += 1
    with pytest.raises(ValueError, match='FP64 Fisher companion'):
        validate_fisher_binding(wrong, common, reference)
    wrong = deepcopy(artifact); del wrong['metadata']['data_sha256']
    with pytest.raises(ValueError, match='reference corpus'):
        validate_fisher_binding(wrong, common, reference)


def test_external_flat_artifact_requires_authenticated_sibling_metadata(fisher_case, tmp_path):
    common, reference, release = fisher_case
    artifact = {key:value.flatten() for key,value in release.items() if key != 'metadata'}
    path = tmp_path/'fisher.pt'; torch.save(artifact, path)
    with pytest.raises(ValueError, match='extraction metadata'):
        validate_fisher_binding(artifact, common, reference, path)
    source = dict(release['metadata'])
    source.update(extraction_scope=source.pop('scope'), model_computation_dtype=source.pop('model_dtype'),
                  continuations_sha256=source.pop('data_sha256'), continuation_count=source.pop('count'),
                  artifact_sha256=sha256(path))
    source.pop('shape')
    metadata_path = tmp_path/'metadata.json'
    metadata_path.write_text(json.dumps(source))
    result = validate_fisher_binding(artifact, common, reference, path)
    assert result['metadata_sha256'] == sha256(metadata_path)
    source['artifact_sha256'] = 'wrong-artifact'
    metadata_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='another artifact'):
        validate_fisher_binding(artifact, common, reference, path)


def test_bundled_fisher_binds_all_4096_frozen_reference_continuations():
    _, common, _ = config(ROOT/'configs/model.yaml')
    path = ROOT/common['fisher_artifact']
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    binding = validate_fisher_binding(artifact, common, ROOT/common['reference_data'], path)
    assert binding['count'] == 4096
    assert binding['scope'] == 'global/all-token'
