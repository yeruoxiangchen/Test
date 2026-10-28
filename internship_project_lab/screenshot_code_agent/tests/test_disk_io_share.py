import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('disk_share', Path(__file__).resolve().parents[1] / 'scripts/disk_io_share.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record(pid, size, count=1, *, comm='python', disk=16, write=False):
    return dict(pid=pid, bytes=size, ios=count, comm=comm, major=8, minor=disk, write=write, us=1000*count)


def test_shares_include_unknown_and_merge_thread_names_but_exclude_other_disk_and_writes():
    report = module.summarize([record(7, 40, comm='worker'), record(7, 20, comm='MainThread'),
                               record(0, 40), record(7, 9000, write=True), record(9, 8000, disk=0)],
                              8, 16, 2, {'read_bytes': 100})
    assert report['read_bytes'] == 100
    assert report['read_ios'] == 3
    assert len(report['rows']) == 2
    assert report['rows'][0]['read_share_pct'] == 60
    assert report['rows'][0]['read_io_share_pct'] == pytest.approx(200/3)
    assert report['rows'][0]['names'] == ['MainThread', 'python', 'worker']
    assert report['rows'][0]['write_bytes'] == 9000
    assert report['rows'][1]['pid'] == 0


def test_read_counts_without_bytes_fail_loudly():
    with pytest.raises(RuntimeError, match='INVALID'):
        module.summarize([record(2, 0, 8)], 8, 16, 5, {'read_bytes': 5000})


def test_empty_disk_has_no_invented_percentage():
    report = module.summarize([], 8, 16, 5, {'read_bytes': 0})
    assert report['rows'] == []
    assert report['trace_to_diskstats_read_pct'] is None


def test_top_limit_does_not_renormalize_and_user_shares_include_hidden_pids(monkeypatch, capsys):
    monkeypatch.setattr(module, 'process_user', lambda pid: 'a' if pid == 1 else 'b')
    report = module.summarize([record(1, 60), record(2, 40)], 8, 16, 1, {'read_bytes': 100})
    module.display(report, 1)
    out = capsys.readouterr().out
    assert '60.0' in out and 'b:' in out and '40.0%' in out
    assert 'additional PIDs are included' in out


def test_diskstats_uses_512_byte_sectors(tmp_path):
    (tmp_path/'stat').write_text('1 0 8 0 2 0 16 0 0 0 0\n')
    assert module.read_disk_stats(tmp_path) == {'read_bytes':4096, 'write_bytes':8192}


def test_partition_and_invalid_names_are_rejected(tmp_path):
    device = tmp_path/'sdb1'
    device.mkdir()
    (device/'partition').write_text('1')
    with pytest.raises(ValueError, match='whole device'):
        module.disk_identity('sdb1', tmp_path)
    with pytest.raises(ValueError, match='disk name'):
        module.disk_identity('../sdb', tmp_path)


def test_installed_source_is_parsed_without_executing_tool_and_zero_byte_bug_is_removed():
    if not Path('/usr/sbin/biotop-bpfcc').is_file():
        pytest.skip('Host BCC tool not installed')
    source = module.probe_source()
    start = source[source.index('int trace_req_start'):source.index('// output')]
    assert 'saved_bytes.update(&req, &length)' in start
    assert 'valp->bytes += req->__data_len' not in source
    assert '__sync_fetch_and_add(&valp->bytes, *length)' in source


def test_unknown_source_refuses_instead_of_silent_wrong_statistics(tmp_path):
    path = tmp_path/'biotop'
    path.write_text("raise RuntimeError('must not execute')\nbpf_text='different code'\n")
    with pytest.raises(ValueError, match='Unsupported'):
        module.probe_source(path)
