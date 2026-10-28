"""Reuse prior /data staging, adding only frozen predicted-depth/evidence caches."""
import importlib.util
import json
from pathlib import Path
from . import common as c

_spec=importlib.util.spec_from_file_location('_depth_conf_storage_helpers',c.HERE.parent/'hssd_visible_surface_oracle_20261026/storage.py')
helpers=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(helpers)


def inventory():
    inv=helpers.Inventory(c)
    for group in ('group4','group2'):
        row=inv.json(c.OLD/'depth'/f'{group}.json','frozen RGB depth/confidence receipt')
        if not row['success'] or row['metadata']['gt_geometry_used']:raise ValueError('Invalid predicted depth origin')
        inv.add(row['npz'],'frozen RGB depth and per-view confidence ranks')
    for uid in c.UIDS:
        row=inv.json(c.OLD/'evidence'/uid/'receipt.json','predicted diagonal evidence receipt')
        if not row['success'] or row['gt_geometry_used']:raise ValueError('Invalid predicted diagonal')
        inv.add(row['tensors'],'predicted diagonal evidence for exact replay')
    return inv


def run():
    priority=helpers._low_priority();caches=helpers._cache_directories(c)
    inv=inventory();contract=inv.contract();identity=c.digest(contract)
    base_binding=c.prior.binding(c.ORACLE/'storage.json')
    base=c.prior.read(c.prior.verify(base_binding))
    if base.get('complete') is not True:raise ValueError('Prior /data staging incomplete')
    target=c.OUT/'storage.json'
    if target.exists():
        old=json.loads(target.read_text())
        if (not old['complete'] or old['additional_inventory_identity']!=identity or old['base_manifest']!=base_binding):
            raise ValueError('Storage inventory changed')
        for item in contract:
            row=old['mapping'][item['path']]
            if row['sha256']!=item['sha256'] or row['physical']!=item['physical']:raise ValueError('Changed physical mapping')
            if helpers._stat(row['physical'])!=row['stat']:
                if helpers._hash_file(row['physical'])!=(row['sha256'],row['size']):raise ValueError('Changed staged payload')
        print(json.dumps({'stage':'storage_verified','additional_files':len(contract),'base_reused':len(base['mapping'])}),flush=True)
        return
    mapping=dict(base['mapping']);copied=0
    for i,item in enumerate(contract,1):
        row=helpers._copy_one(inv,item)
        if item['path'] in mapping and mapping[item['path']]['sha256']!=row['sha256']:
            raise ValueError('Additional cache conflicts with prior frozen mapping')
        mapping[item['path']]=row;copied+=row['size']
        c.write(c.OUT/'storage_progress.json',{'time':c.now(),'completed':i,'total':len(contract),
            'additional_bytes':copied,'logical':item['path']})
    c.write(target,{'schema':'depth_confidence_staging.v1','time':c.now(),'complete':True,'success':True,
        'base_manifest':base_binding,'additional_inventory_identity':identity,'mapping':mapping,
        'additional_file_count':len(contract),'additional_bytes':copied,'base_files_reused':len(base['mapping']),
        'priority':priority,'cache_environment':caches,'home_removed':False,
        'scope':'reuse verified /data model/input/reference payloads; stage only cached RGB depth/rank and four diagonal evidence tensors; no VGGT inference/weights or raw assets'})
    print(json.dumps({'stage':'storage_complete','additional_files':len(contract),'additional_bytes':copied,
        'base_reused':len(base['mapping'])}),flush=True)


if __name__=='__main__':run()
