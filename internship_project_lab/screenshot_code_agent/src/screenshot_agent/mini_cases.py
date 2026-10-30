"""Expanded visible development requirements for the known five-task regression.

Authored after prior failure analysis. No hidden test files or reference patches
are read. These checks are development supervision, not held-out performance.
"""
from __future__ import annotations

import json
import textwrap

from .workflow_cases import development_case

MODULES = {
    'marshmallow-code__marshmallow-2123': ['tests/test_validate.py', 'tests/test_fields.py'],
    'pylint-dev__astroid-984': ['tests/unittest_modutils.py'],
    'pyvista__pyvista-4329': ['tests/test_grid.py', 'tests/test_polydata.py'],
    'pvlib__pvlib-python-1832': ['pvlib/tests/test_pvsystem.py', 'pvlib/tests/test_modelchain.py'],
    'pydicom__pydicom-793': ['pydicom/tests/test_filereader.py'],
}
EXTRA = {
    'marshmallow-code__marshmallow-2123': '''
from marshmallow import fields, validate
for factory in (validate.URL, fields.Url):
    try:
        factory(relative=False, absolute=False)
    except ValueError:
        pass
    else:
        raise AssertionError('Both relative and absolute cannot be disabled')
''',
    'pylint-dev__astroid-984': '''
import os, tempfile
from astroid import modutils
with tempfile.TemporaryDirectory() as root:
    missing = os.path.join(root, 'absent_package', 'module.py')
    assert modutils.is_relative('nearby_module', missing) is False
''',
    'pyvista__pyvista-4329': '''
import numpy as np
import pyvista as pv
cells = np.array([4, 0, 1, 2, 3])
grid = pv.UnstructuredGrid(cells, np.array([10], dtype=np.uint8),
    np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]]))
replacement = cells.copy()
replacement[1:3] = replacement[1:3][::-1]
grid.cells = replacement
np.testing.assert_array_equal(grid.cells, replacement)
assert not grid.cells.flags.writeable
''',
    'pvlib__pvlib-python-1832': '''
import numpy as np
from unittest.mock import patch
from pvlib import iam, pvsystem, location, temperature
from pvlib.modelchain import ModelChain
params = {'theta_ref': [0., 35., 70., 90.], 'iam_ref': [1., .97, .7, 0.],
          'method': 'cubic', 'normalize': False, 'pdc0': 220., 'gamma_pdc': -.004}
system = pvsystem.PVSystem(module_parameters=params, inverter_parameters={'pdc0': 220.},
    temperature_model_parameters=temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass'])
mc = ModelChain(system, location.Location(32., -110.), aoi_model='interp',
    dc_model='pvwatts', ac_model='pvwatts', spectral_model='no_loss', temperature_model='sapm')
mc.results.aoi = np.array([0., 25., 65.])
mc.aoi_model()
np.testing.assert_allclose(mc.results.aoi_modifier,
    iam.interp(mc.results.aoi, **{k: params[k] for k in ('theta_ref', 'iam_ref', 'method', 'normalize')}))
for model, options in [('interp', {k:params[k] for k in ('theta_ref','iam_ref','method','normalize')}),
                       ('physical', {'n':1.526, 'K':4., 'L':.002, 'n_ar':1.29})]:
    obj = pvsystem.PVSystem(module_parameters=options)
    with patch.object(iam, model, wraps=getattr(iam, model)) as called:
        obj.get_iam(np.array([15.,45.]), iam_model=model)
    assert called.call_args[1] == options, 'optional IAM parameters lost: ' + repr(called.call_args)
try:
    system.get_iam(25., iam_model='not_a_model')
except ValueError:
    pass
else:
    raise AssertionError('Invalid IAM model must raise ValueError')
''',
}


def verified_case(task_id):
    original = development_case(task_id)
    if task_id not in MODULES:
        return original
    script = original['script'] if original else ''
    script += '\n# Additional public behavior checks after the five-task development run.\n'
    script += textwrap.dedent(EXTRA.get(task_id, '')).strip() + '\n'
    script += 'print("expanded public behavior checks passed")\n'
    return {**(original or {}), 'name': 'mini_verified_dev5_20261029', 'task_id': task_id,
            'note': (original or {}).get('note', 'Fix relative import detection without breaking existing results.')
                    + ' All supplied checks, real constructors and existing public regression behavior must pass.',
            'script': script, 'public_modules': MODULES[task_id],
            'scope': 'known development cases; previous failure feedback used, not held-out evaluation'}


def repair_case(task_id):
    case = verified_case(task_id)
    if not case:
        return None
    case = dict(case)
    notes = {
        'pylint-dev__astroid-984': ' Parent traversal must terminate at filesystem root even when no spec exists; catching ImportError alone can introduce a loop.',
        'pyvista__pyvista-4329': ' Validate the real VTK SetCells overload and preserve cell types when replacing connectivity; a setter that raises TypeError is not a fix.',
        'pvlib__pvlib-python-1832': ' Optional parameters must propagate for physical as well as interp; inspect the underlying function signature, including n_ar.',
        'marshmallow-code__marshmallow-2123': ' Known development API contract: repr always includes relative and absolute; disabling both raises ValueError with message URL validation cannot set both relative and absolute to False.',
    }
    case['note'] += notes.get(task_id, '')
    case['name'] = 'mini_repair_dev5_20261030'
    if task_id == 'marshmallow-code__marshmallow-2123':
        case['script'] += "\nassert 'absolute=True' in repr(validate.URL())\n"
        case['script'] += "try:\n    validate.URL(relative=False, absolute=False)\nexcept ValueError as error:\n    assert str(error) == 'URL validation cannot set both relative and absolute to False'\nelse:\n    raise AssertionError('invalid URL flags accepted')\n"
    return case


def recovery_case(task_id):
    """Run the same visible contracts independently, so one failure hides none."""
    case = repair_case(task_id)
    if not case:
        return None
    base = development_case(task_id)
    groups = {'requested_behavior': base['script']} if base else {}
    if task_id in EXTRA:
        groups['additional_api_contract'] = textwrap.dedent(EXTRA[task_id])
    if task_id == 'marshmallow-code__marshmallow-2123':
        groups['repr_and_error_contract'] = '''
from marshmallow import validate
assert 'absolute=True' in repr(validate.URL()), 'repr must include absolute=True even at its default'
try:
    validate.URL(relative=False, absolute=False)
except ValueError as error:
    assert str(error) == 'URL validation cannot set both relative and absolute to False'
else:
    raise AssertionError('invalid URL flags accepted')
'''
    script = ('import json, traceback\n'
              'groups = json.loads(' + repr(json.dumps(groups)) + ')\n'
              'results = {}\n'
              'for name, source in groups.items():\n'
              '    try:\n'
              '        exec(compile(source, name, "exec"), {"__name__": "__main__"})\n'
              '        results[name] = {"passed": True}\n'
              '    except Exception as error:\n'
              '        results[name] = {"passed": False, "error": traceback.format_exc()}\n'
              'print("__SCA_BEHAVIOR__" + json.dumps(results))\n'
              'raise SystemExit(0 if all(r["passed"] for r in results.values()) else 1)\n')
    return {**case, 'script': script, 'name': 'mini_recovery_dev5_20261030',
            'groups': list(groups), 'note': case['note'] +
            ' All named checks run independently; correct the reported failing contract, '
            'not just a reduced reproduction. New root debug files are omitted from delivery.'}
