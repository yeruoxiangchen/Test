"""Explicit developer regression cases, authored after inspecting v3 failures.

These are visible requirements, NOT held-out evaluation. No reference patch is
loaded or installed by this module. The original final grader remains separate.
"""
from __future__ import annotations

import textwrap

CASES = {
    'marshmallow-code__marshmallow-2123': (
        'Support URL(relative=True, absolute=False) in both fields.Url and validate.URL. '
        'Keep absolute=True backward compatible and retain URL syntax validation. '
        'A relative_only flag alone does not implement the requested regression interface.',
        '''
from marshmallow import fields, validate, ValidationError
v = validate.URL(relative=True, absolute=False)
for text in ['/login', '/login?a=1', '?a=1', '/login#anchor']:
    assert v(text) == text, 'valid relative URL rejected'
    assert fields.Url(relative=True, absolute=False).deserialize(text) == text
for text in ['https://other.example/', 'ftp://other.example/', 'bad path', 'login.html', '']:
    try:
        v(text)
    except ValidationError:
        pass
    else:
        raise AssertionError('invalid/absolute URL accepted: ' + repr(text))
assert validate.URL()('https://example.com') == 'https://example.com'
'''),
    'pvlib__pvlib-python-1616': (
        'The existing development acceptance API is iam.physical(aoi, n_ar=...). '
        'Extend the existing physical model with optional AR coating while preserving defaults, '
        'scalar/array/Series behavior, and equivalence when n_ar=1 or n_ar equals glass index. '
        'The original issue proposed a new fresnel_ar function, but that alone does not cover this API.',
        '''
import numpy as np
from pvlib import iam
aoi = np.array([0., 22.5, 45., 67.5, 90., 100., np.nan])
np.testing.assert_allclose(iam.physical(aoi, n_ar=1.29),
    [1., .99944171, .9917463, .91506158, 0., 0., np.nan], rtol=2e-7, atol=1e-8)
np.testing.assert_allclose(iam.physical(aoi, n_ar=1.), iam.physical(aoi), equal_nan=True)
assert np.isclose(iam.physical(0., n_ar=1.29), 1.)
'''),
    'pydicom__pydicom-816': (
        'Implement generic unknown-encoding and bad-byte fallback, not a literal Not Supplied special case. '
        'Preserve mapping of valid string and list encodings, nested/multiple character sets, warnings, '
        'and strict enforce_valid_values behavior. Handle errors in charset.decode_string and the '
        'fragment decoding/encoding paths; convert_encodings may legitimately retain Python codec names.',
        '''
import warnings
from pydicom import charset, config
config.enforce_valid_values = False
assert charset.convert_encodings('ISO_IR 192') == ['UTF8'], 'valid scalar encoding mapping broke'
assert charset.convert_encodings(['ISO_IR 192']) == ['UTF8']
for name in ['Not Supplied', 'this-encoding-does-not-exist']:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        decoded = charset.decode_string(b'caf\\xe9', charset.convert_encodings([name]), set())
    assert decoded == 'caf\\u00e9', 'unknown charset did not fall back'
    assert caught, 'fallback must warn'
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    decoded = charset.decode_string(b'\\xff', ['UTF8'], set())
assert decoded == '\\ufffd' and caught, 'bad bytes must warn and use replacement characters'
config.enforce_valid_values = True
try:
    charset.decode_string(b'text', ['this-encoding-does-not-exist'], set())
except LookupError:
    pass
else:
    raise AssertionError('strict mode must reject unknown encoding')
finally:
    config.enforce_valid_values = False
'''),
    'pyvista__pyvista-4329': (
        'Expose PolyData.faces and UnstructuredGrid.cells as read-only numpy arrays; '
        'returning a writable copy is not the accepted behavior. Preserve setters for replacement. '
        'Update internal callers that modify faces in place, including demos/logo.py, to copy then assign.',
        '''
import numpy as np
import pyvista as pv
mesh = pv.Sphere()
faces = mesh.faces
assert not faces.flags.writeable, 'PolyData.faces must be read-only'
try:
    faces[1] = faces[1]
except ValueError:
    pass
else:
    raise AssertionError('in-place face writes must raise')
replacement = faces.copy()
mesh.faces = replacement
np.testing.assert_array_equal(mesh.faces, replacement)
grid = pv.UnstructuredGrid(np.array([4, 0, 1, 2, 3]), np.array([10], dtype=np.uint8),
    np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]]))
assert not grid.cells.flags.writeable, 'UnstructuredGrid.cells must be read-only'
'''),
    'sqlfluff__sqlfluff-1733': (
        'Fix the extra whitespace through the interaction of L003/L036/L039. '
        'Inspect L039 handling of whitespace after newline/indent meta segments. '
        'Changing L016 and leaving only debug scripts is not a fix. Verify exact output and idempotence.',
        '''
import sqlfluff
sql = 'WITH example AS (\\n    SELECT my_id,\\n        other_thing,\\n        one_more\\n    FROM\\n        my_table\\n)\\n\\nSELECT *\\nFROM example\\n'
expected = sql.replace('    SELECT my_id,', '    SELECT\\n        my_id,')
actual = sqlfluff.fix(sql, dialect='ansi', rules=['L003', 'L036', 'L039'])
assert actual == expected, 'wrong CTE indentation: ' + repr(actual)
assert sqlfluff.fix(actual, dialect='ansi', rules=['L003', 'L036', 'L039']) == expected, 'fix not idempotent'
'''),
    'marshmallow-code__marshmallow-1702': (
        'Add explicit metadata= while retaining legacy keyword metadata with the repository deprecation warning. '
        'Merge the two dictionaries without mutating the caller input; legacy keyword metadata wins conflicts. '
        'Do not dismiss failures of existing description= calls as intentional API breakage.',
        '''
import warnings
from marshmallow import fields
metadata = {'description': 'explicit', 'custom': 1}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    value = fields.String(description='legacy', extra='ok', metadata=metadata)
assert value.metadata == {'description': 'legacy', 'custom': 1, 'extra': 'ok'}, value.metadata
assert metadata == {'description': 'explicit', 'custom': 1}, 'mutated caller metadata'
assert caught, 'legacy kwargs must warn'
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    value = fields.String(metadata={'hello': 'world'})
assert value.metadata == {'hello': 'world'}
assert not caught, 'explicit metadata should not emit legacy warning'
'''),
    'pvlib__pvlib-python-1832': (
        'Integrate interp throughout iam._IAM_MODEL_PARAMS, Array/PVSystem.get_iam, '
        'ModelChain.infer_aoi_model and interp_aoi_loss. Use theta_ref and iam_ref parameters. '
        'The parameter registry should list required parameters only; get_iam should also forward '
        'provided optional function parameters (including physical n_ar) from module_parameters. '
        'Reuse self.system.get_iam rather than manually implementing only ModelChain arrays.',
        '''
import numpy as np
from pvlib import iam, pvsystem
from pvlib.modelchain import ModelChain
params = {'theta_ref': [0., 40., 80., 90.], 'iam_ref': [1., .98, .6, 0.]}
system = pvsystem.PVSystem(module_parameters=params)
aoi = np.array([0., 20., 60.])
np.testing.assert_allclose(system.get_iam(aoi, iam_model='interp'), iam.interp(aoi, **params))
assert iam._IAM_MODEL_PARAMS['interp'] == {'theta_ref', 'iam_ref'}
mc = object.__new__(ModelChain)
mc.system = system
assert mc.infer_aoi_model().__name__ == 'interp_aoi_loss', 'automatic interp inference missing'
'''),
    'pydicom__pydicom-793': (
        'Format unknown two-character VR at the conversion source in values.py. '
        'ASCII alphabetic VRs retain readable text; if either character is outside A-Z/a-z, '
        'represent BOTH characters as lowercase 0xNN separated by a space. '
        'Changing only dataelem.py or using printable rather than alphabetic classification is insufficient.',
        '''
from pydicom.values import convert_value
for vr, expected in [('@A', '0x40 0x41'), ('A\\x00', '0x41 0x00'), ('\\xffA', '0xff 0x41'), ('ZZ', 'ZZ')]:
    try:
        convert_value(vr, None)
    except NotImplementedError as e:
        assert "Unknown Value Representation '{}'".format(expected) in str(e), str(e)
    else:
        raise AssertionError('unknown VR accepted')
'''),
    'pylint-dev__astroid-1092': (
        'Preserve the bound instance while inferring a generator-backed context manager. '
        'Inspect Generator in bases.py, generator creation in scoped_nodes.py and '
        '_infer_context_manager in protocols.py. Thread the inference/call context through '
        'the real generator object; do not invent InferenceContext.locals or swallow errors. '
        'Assert the inferred class name B, not address-dependent string formatting.',
        '''
import astroid
source = ''' + repr('''
import contextlib
class A:
    @contextlib.contextmanager
    def get(self):
        yield self
class B(A):
    def play(self):
        pass
with B().get() as b:
    b.play()
''') + '''
values = list(astroid.parse(source).ilookup('b'))
assert values and all(getattr(value, '_proxied', None).name == 'B' for value in values), repr(values)
'''),
}

# Existing public regression modules for the explicitly known development cases.
# Restricting IAM checks avoids unrelated solar-position JIT compilation.
PUBLIC_MODULES = {'pvlib__pvlib-python-1616': ['pvlib/tests/test_iam.py']}


def development_case(task_id):
    item = CASES.get(task_id)
    if item is None:
        return None
    note, body = item
    script = ('# Visible development regression, authored from prior failed runs.\n'
              'def check():\n' + textwrap.indent(textwrap.dedent(body).strip(), '    ') + '\n'
              'try:\n    check()\nexcept ImportError:\n    raise\n'
              'except Exception as error:\n'
              '    raise AssertionError("development acceptance: " + str(error))\n'
              'print("development acceptance passed")\n')
    return {'name': 'smoke10_development_20261029', 'task_id': task_id,
            'note': note, 'script': script,
            'public_modules': PUBLIC_MODULES.get(task_id),
            'scope': 'known development regression; prior failure feedback used'}
