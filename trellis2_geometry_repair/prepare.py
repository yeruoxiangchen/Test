import shutil
from .config import OUT,LEGACY,PARAMS,folder,read,write,setup,now,snapshot


def main():
    setup();rows=[]
    for index in PARAMS['indices']:
        source=LEGACY/'data'/f'{index:03d}';rec=read(source/'observations.json')
        target=read(source/'evaluation_target.json');root=OUT/'data'/f'{index:03d}';root.mkdir(parents=True,exist_ok=True)
        for kind in ('images','masks'):
            entries=[]
            for view,value in enumerate(rec[kind]):
                dest=root/f'{kind}_{view:02d}.png'
                if not dest.exists():shutil.copyfile(value['path'],dest)
                entries.append(dict(path=str(dest)))
            rec[kind]=entries
        write(root/'observations.json',rec);write(root/'evaluation_target.json',target)
        for seed in PARAMS['seeds']:folder(index,seed).mkdir(parents=True,exist_ok=True)
        rows.append(dict(index=index,uid=rec['uid'],split='inspected_development',views=len(rec['images']),
                         source=str(source),rendered_input=True))
    value=dict(time=now(),rows=rows,seeds=PARAMS['seeds'],scope='native T2 backend pilot on four known development objects')
    if (OUT/'registry.json').exists() and read(OUT/'registry.json')['rows']!=rows:raise RuntimeError('pilot cohort changed')
    write(OUT/'registry.json',value);snapshot('prepare');print(value,flush=True)


if __name__=='__main__':main()
