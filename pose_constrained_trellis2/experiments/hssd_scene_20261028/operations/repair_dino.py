"""Archive the failed DINO stage and explicitly carry over unchanged codec stages."""
import ast,fcntl,time,shutil
from pathlib import Path
from pose_constrained_trellis2.experiments.hssd_scene_20261028 import common as c
from pose_constrained_trellis2.experiments.hssd_scene_20261028.runner import identity

def main():
    with (c.OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=c.read(c.OUT/'status.json')
        if state['status']!='failed' or not state['stage'].endswith('_pose'):raise RuntimeError('Expected failed pose stage')
        old=c.read(c.OUT/'runtime.json');new=identity();newkey=c.digest(new)
        changed={k for k,v in old['runtime']['files'].items() if new['files'].get(k)!=v}
        worker=str((c.HERE/'worker.py').relative_to(c.REPO))
        if changed!={worker}:raise RuntimeError(('Unexpected source changes',changed))
        before=c.HERE/'operations/worker_before_dino.py'
        if c.sha(before)!=old['runtime']['files'][worker]:raise RuntimeError('Backup source hash mismatch')
        def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
        a,b=functions(before),functions(c.HERE/'worker.py')
        if {k for k in a if a[k]!=b.get(k)}!={'pose'}:raise RuntimeError('Non-pose functions changed')
        dest=c.OUT/'repairs'/('dino_api_'+str(time.time_ns()));dest.mkdir(parents=True)
        shutil.copy2(before,dest/'worker_before.py')
        for name in ['runtime.json','status.json']:(c.OUT/name).rename(dest/name)
        records=[]
        for p in sorted((c.OUT/'stages').glob('*.json')):
            r=c.read(p)
            if not p.stem.endswith(('_dual512','_dual1024','_encode512','_encode1024','_ss','_decode')):raise RuntimeError('Unexpected completed stage')
            if r['runtime_identity']!=old['identity']:raise RuntimeError('Wrong old identity')
            verified=[]
            for artifact in r['artifacts']:
                f=Path(artifact['path'])
                if f.stat().st_size!=artifact['bytes']:raise RuntimeError('Artifact changed')
                verified.append({'path':str(f),'sha256':c.sha(f)})
            shutil.copy2(p,dest/p.name)
            c.write(p,{**r,'runtime_identity':newkey,'carried_from':old['identity'],'carry_reason':'AST-identical codec functions; only uncompleted pose branch repaired'})
            records.append({'stage':p.stem,'verified_artifacts':verified})
        c.write(c.OUT/'runtime.json',{'identity':newkey,'runtime':new})
        c.write(dest/'repair.json',{'time':c.now(),'old_identity':old['identity'],'new_identity':newkey,
            'reason':'transformers5.9 moved layer list to model.model.layer; named RoPE argument; functional layer_norm preserved',
            'preserved':records,'model_weights_unchanged':True})
        print('Preserved',len(records),'codec stages; new identity',newkey)

if __name__=='__main__':main()
