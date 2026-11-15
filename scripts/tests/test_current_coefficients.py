"""Check current all-coordinate coefficients and configured defaults."""
import importlib.util
from pathlib import Path

import pytest
import torch

from cans.data_processing import sha256
from cans.scaling_solver import cans_scales


def test_damping_median_includes_zero_coordinates():
    gradient = torch.tensor([[1., -2., 3., -4.]])
    fisher = torch.tensor([[0., 0., 2., 4.]])
    # Two zero coordinates make the lower all-coordinate median zero, while
    # filtering to positive coordinates would silently choose a valid damping.
    with pytest.raises(ValueError, match="median is zero"):
        cans_scales(gradient, fisher, .1)
    fisher[0, 1] = 1.
    _, diagnostics = cans_scales(gradient, fisher, .1)
    assert diagnostics['damping'] == 1.
    assert diagnostics['cap'] is None
    assert diagnostics['pre_cap_cost'] == pytest.approx(.1)


@pytest.mark.parametrize('objective_sign', [-1, 1])
def test_grad_only_cli_uses_yaml_coefficient_and_task_sign_once(tmp_path, monkeypatch, objective_sign):
    source = Path(__file__).resolve().parents[1]/'cans_compute_neuron_scales.py'
    spec = importlib.util.spec_from_file_location('current_coefficient_cli', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path/'data.jsonl'; data.write_text('{}\n')
    gradient = torch.tensor([[1., -2., 0.]])
    torch.save(dict(gradient=gradient, metadata=dict(model_config_sha256='toy',
        data_sha256=sha256(data), scope='global/all-token', model_dtype='float32',
        mean_dtype='float32')), tmp_path/'gradient.pt')
    domain = dict(domain='toxicity' if objective_sign < 0 else 'safety', data='data.jsonl',
        artifact='gradient.pt', objective_sign=objective_sign, cans=dict(cap=None),
        **{'grad-only': dict(coefficient=2.)})
    monkeypatch.setattr(module, 'config', lambda _: (tmp_path, dict(model='toy', model_config_sha256='toy'), domain))
    base = ['--config', 'toy.yaml', '--method', 'grad-only']
    for name, extra, coefficient in [('default', [], 2.), ('override', ['--coefficient', '3'], 3.)]:
        output = tmp_path/f'{name}.pt'
        module.main([*base, *extra, '--output', str(output)])
        result = torch.load(output, weights_only=True)
        torch.testing.assert_close(result['scales'], 1 + objective_sign*coefficient*gradient.double())
        assert result['metadata']['formula'] == '1 + c * g'
        assert result['metadata']['cap'] is None
        assert result['metadata']['fisher_used'] is False
