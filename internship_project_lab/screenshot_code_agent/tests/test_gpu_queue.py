import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from screenshot_agent.control import atomic_json
from screenshot_agent.gpu_queue import GPULease, QueuePolicy, snapshot, try_lease, wait_for_idle

ROOT = Path(__file__).resolve().parents[1]


def controller(tmp_path):
    history = []
    return SimpleNamespace(output=tmp_path,update=lambda **kw:history.append(kw),history=history)


def readings(idle, *, processes=None, used=None, policy=None):
    processes,used = processes or {},used or {}
    lines = []
    for i in range(8):
        memory = used.get(i,1 if i in idle else 6000)
        lines.append(f'{i}, GPU-{i}, NVIDIA RTX 3090, 24576, {memory}, {24127-memory}, {0 if i in idle else 65}')
    proc_lines = [f'GPU-{i}, {pid}' for i,pids in processes.items() for pid in pids]
    def run(command):
        return '\n'.join(proc_lines if '--query-compute-apps=gpu_uuid,pid' in command else lines)
    return snapshot(policy or QueuePolicy(),run)


def test_low_memory_with_active_compute_process_is_not_idle():
    rows = readings(range(8),processes={2:[12345]},used={2:100})
    assert rows[2]['idle'] is False
    assert rows[2]['blockers']==['compute_process_present']
    assert rows[1]['idle'] is True
    assert rows[1]['required_free_mib']==21197


def test_explicit_sharing_accepts_1786_mib_but_still_requires_headroom():
    rows=readings([3,4,5],processes={2:[12345]},used={2:1786},
                  policy=QueuePolicy(allow_shared_gpus=True))
    assert rows[2]['eligible'] and rows[2]['shared'] and not rows[2]['idle']
    assert rows[2]['required_free_mib']==21197 and rows[2]['utilization']==65
    assert not rows[1]['eligible']  # Six GiB use is above the light-sharing ceiling.
    constrained=readings([3,4,5],processes={2:[12345]},used={2:1786},
                         policy=QueuePolicy(allow_shared_gpus=True,memory_fraction=.95))
    assert not constrained[2]['eligible'] and 'insufficient_free_memory' in constrained[2]['blockers']


def test_shared_memory_spike_resets_stability_before_start(tmp_path):
    policy=QueuePolicy(allow_shared_gpus=True)
    def sample(used):
        return readings([3,4,5],processes={2:[12345]},used={2:used},policy=policy)
    samples=iter([sample(1786),sample(7262),sample(1786),sample(1786),sample(1786)])
    c,sleeps=controller(tmp_path),[]
    lease=wait_for_idle(c,policy,sample=lambda p:next(samples),sleep=sleeps.append,lock_root=tmp_path/'locks')
    try:
        assert set(lease.gpus)=={'2','3','4','5'} and sleeps==[10,10,10]
        assert c.history[0]['stage']=='waiting_for_gpu_capacity'
        assert c.history[-2]['gpu_queue']['stable_eligible_count']==4
    finally:
        lease.close()


def test_shared_policy_prefers_four_truly_idle_devices(tmp_path):
    policy=QueuePolicy(allow_shared_gpus=True)
    rows=readings([0,1,2,3,4],processes={0:[12345]},used={0:1786},policy=policy)
    lease=wait_for_idle(controller(tmp_path),policy,sample=lambda p:rows,sleep=lambda s:None,
                        lock_root=tmp_path/'locks')
    try:
        assert lease.gpus==['1','2','3','4']
    finally:
        lease.close()


@pytest.mark.parametrize('gpu_csv,process_csv',[
    ('0, GPU-0, 3090, 24576, N/A, 24000, 0',''),
    ('0, GPU-0, 3090, 24576, 1, 24000, 0','permission denied'),
    ('0, GPU-0, 3090, 24576, 1, 24000, 0',''),
])
def test_unknown_or_missing_telemetry_cannot_be_treated_as_free(gpu_csv,process_csv):
    def run(command):
        return process_csv if '--query-compute-apps=gpu_uuid,pid' in command else gpu_csv
    with pytest.raises(ValueError):
        snapshot(QueuePolicy(),run)


def test_auto_selects_any_four_after_two_stable_samples(tmp_path):
    sample = iter([readings([]),readings([1,3,5,7]),readings([1,3,5,7]),readings([1,3,5,7])])
    c,sleeps = controller(tmp_path),[]
    lease = wait_for_idle(c,QueuePolicy(),sample=lambda p:next(sample),sleep=sleeps.append,lock_root=tmp_path/'locks')
    try:
        assert lease.gpus==['1','3','5','7']
        assert sleeps==[10,10]
        assert c.history[-1]['stage']=='gpu_memory_ready'
        assert json.loads((tmp_path/'gpu_allocations.jsonl').read_text())['selected_gpus']==lease.gpus
    finally:
        lease.close()


def test_four_transient_idle_cards_do_not_satisfy_two_samples(tmp_path):
    sample = iter([readings([0,1,2,3]),readings([4,5,6,7]),readings([4,5,6,7]),readings([4,5,6,7])])
    c,sleeps = controller(tmp_path),[]
    lease = wait_for_idle(c,QueuePolicy(),sample=lambda p:next(sample),sleep=sleeps.append,lock_root=tmp_path/'locks')
    assert lease.gpus==['4','5','6','7'] and len(sleeps)==2
    lease.close()


def test_lost_gpu_at_final_check_reselects_instead_of_starting(tmp_path):
    sample = iter([readings([0,1,2,3]),readings([0,1,2,3]),readings([0,1,2]),
                   readings([4,5,6,7]),readings([4,5,6,7]),readings([4,5,6,7])])
    lease = wait_for_idle(controller(tmp_path),QueuePolicy(),sample=lambda p:next(sample),sleep=lambda s:None,lock_root=tmp_path/'locks')
    assert lease.gpus==['4','5','6','7']
    old = try_lease(readings([0,1,2,3])[:4],tmp_path/'locks')
    assert old is not None  # Abandoned allocation released all four locks.
    old.close()
    lease.close()


def test_telemetry_failure_resets_stability_and_keeps_waiting(tmp_path):
    samples = iter([readings([0,1,2,3]),subprocess.TimeoutExpired('nvidia-smi',15),
                    readings([0,1,2,3]),readings([0,1,2,3]),readings([0,1,2,3])])
    def sample(policy):
        value = next(samples)
        if isinstance(value,Exception):
            raise value
        return value
    c,sleeps = controller(tmp_path),[]
    lease = wait_for_idle(c,QueuePolicy(),sample=sample,sleep=sleeps.append,lock_root=tmp_path/'locks')
    assert len(sleeps)==3
    assert any(x.get('gpu_queue',{}).get('telemetry_error') for x in c.history)
    lease.close()


def test_parallel_queues_cannot_claim_same_devices_and_can_choose_other_four(tmp_path):
    all_rows = readings(range(8))
    first = try_lease(all_rows[:4],tmp_path/'locks')
    assert first is not None
    assert try_lease(all_rows[:4],tmp_path/'locks') is None
    try:
        lease = wait_for_idle(controller(tmp_path),QueuePolicy(),sample=lambda p:readings(range(8)),
                              sleep=lambda s:None,lock_root=tmp_path/'locks')
        assert lease.gpus==['4','5','6','7']
        lease.close()
    finally:
        first.close()
    released = try_lease(all_rows[:4],tmp_path/'locks')
    assert released is not None
    released.close()


def test_cancel_during_wait_does_not_start_any_program(tmp_path):
    def stop(seconds):
        raise KeyboardInterrupt('cancel queue')
    with pytest.raises(KeyboardInterrupt):
        wait_for_idle(controller(tmp_path),QueuePolicy(),sample=lambda p:readings([]),sleep=stop,lock_root=tmp_path/'locks')
    assert not (tmp_path/'gpu_allocations.jsonl').exists()


def test_service_uses_selected_physical_ids_and_releases_locks_on_exit(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_scale_4gpu as scale

    monkeypatch.setattr(scale,'ROOT',tmp_path)
    monkeypatch.setattr(scale,'MODEL',tmp_path/'model')
    atomic_json(tmp_path/'model/download_complete.json',{'revision':'fixture','files':{}})
    atomic_json(tmp_path/'.coder-env/setup_complete.json',{'vllm_version':'fixture'})
    lease = GPULease(readings([1,3,5,7])[1::2],[])
    closed = []
    monkeypatch.setattr(lease,'close',lambda:closed.append(True))
    policies=[]
    monkeypatch.setattr('screenshot_agent.gpu_queue.wait_for_idle',lambda c,p:policies.append(p) or lease)
    monkeypatch.setattr(scale,'ready_server',lambda *a:None)
    monkeypatch.setattr(scale,'terminate',lambda p:None)
    class Socket:
        def __enter__(self):
            return self
        def __exit__(self,*args):
            pass
        def bind(self,address):
            pass
    monkeypatch.setattr(scale.socket,'socket',Socket)
    c = scale.Controller(tmp_path/'output','test')
    seen = []
    def launch(command,label,env):
        seen.append((command,env))
        return {'proc':SimpleNamespace(poll=lambda:None),'pid':1234}
    monkeypatch.setattr(c,'launch',launch)
    args = SimpleNamespace(gpus='auto',gpu_candidates='0,1,2,3,4,5,6,7',gpu_poll_seconds=10,
        gpu_stable_samples=2,port=18856,split='workflow',gpu_memory_utilization=.8,gpu_headroom_mib=1536,
        allow_shared_gpus=True)
    scale.start_service(c,args,tmp_path,'coder',{'coder':{'revision':'fixture'},'vllm_version':'fixture','tool_call_parser':'qwen3_coder'})
    assert seen[0][1]['CUDA_VISIBLE_DEVICES']=='1,3,5,7'
    assert seen[0][0][seen[0][0].index('--tensor-parallel-size')+1]=='4'
    assert policies[0].allow_shared_gpus and policies[0].max_shared_used_mib==2048
    assert not closed
    c.close()
    assert closed==[True]


def test_queue_cli_accepts_explicit_shared_mode(monkeypatch):
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_repair_workflow as workflow

    def forbidden(*args,**kwargs):
        raise RuntimeError('Validation succeeded; stop before data preparation')
    monkeypatch.setattr(workflow.harness,'storage',forbidden)
    monkeypatch.setattr(sys,'argv',['run_repair_workflow.py','all','--gpus','auto','--allow-shared-gpus'])
    with pytest.raises(RuntimeError,match='Validation succeeded'):
        workflow.main()
