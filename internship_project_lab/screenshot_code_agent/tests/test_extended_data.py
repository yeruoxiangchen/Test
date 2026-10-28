import io
import tarfile

import pytest

from screenshot_agent.extended_data import export_source, scored_test_command


def test_sympy_keeps_official_runner_and_function_scoring(tmp_path):
    path='sympy/matrices/tests/test_matrices.py'
    (tmp_path/path).parent.mkdir(parents=True)
    (tmp_path/path).write_text('def test_matrix(): pass\n')
    record={'repo':'sympy/sympy','FAIL_TO_PASS':['test_matrix'],'PASS_TO_PASS':[]}
    original="PYTHONWARNINGS='ignore::UserWarning' bin/test -C --verbose "+path
    command,paths=scored_test_command(record,tmp_path,original)
    assert command==original and paths==[path]


def test_historical_pytest_disables_incompatible_optional_plugins(tmp_path):
    (tmp_path/'test_a.py').write_text('')
    command,paths=scored_test_command({'repo':'pytest-dev/pytest','FAIL_TO_PASS':['test_a.py::test_a'],
                                     'PASS_TO_PASS':[]},tmp_path,'pytest -rA test_a.py')
    assert command=='PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -rA test_a.py'
    assert paths==['test_a.py']


@pytest.mark.parametrize('target',['../../secret','/etc/passwd','missing.py','link.py'])
def test_archive_links_cannot_escape_or_cycle(tmp_path,target):
    record={'repo':'pylint-dev/pylint','base_commit':'pinned'}
    archive=tmp_path/'pylint-dev_pylint_pinned.tar.gz'
    with tarfile.open(archive,'w:gz') as out:
        m=tarfile.TarInfo('root/link.py')
        m.type=tarfile.SYMTYPE
        m.linkname=target
        out.addfile(m)
    with pytest.raises(ValueError):
        export_source(record,tmp_path,tmp_path/'export')
    assert not (tmp_path/'export').exists()


def test_internal_fixture_link_becomes_regular_file(tmp_path):
    record={'repo':'pylint-dev/pylint','base_commit':'pinned'}
    with tarfile.open(tmp_path/'pylint-dev_pylint_pinned.tar.gz','w:gz') as out:
        m=tarfile.TarInfo('root/module.py')
        body=b'answer = 42\n'
        m.size=len(body)
        out.addfile(m,io.BytesIO(body))
        m=tarfile.TarInfo('root/sub/link.py')
        m.type=tarfile.SYMTYPE
        m.linkname='../module.py'
        out.addfile(m)
    _,links=export_source(record,tmp_path,tmp_path/'export')
    result=tmp_path/'export/sub/link.py'
    assert result.read_bytes()==body and not result.is_symlink()
    assert links==[{'path':'sub/link.py','target':'module.py'}]
