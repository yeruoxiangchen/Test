from .config import OUT,PARAMS,ARMS,case,read


def main():
    cases=[case(i,s) for i in PARAMS['indices'] for s in PARAMS['seeds']]
    print('TRELLIS.2 四组诊断：'+str(sum((r/a/'complete.json').exists() for r in cases for a in ARMS))+'/32；评估 '+str(sum((r/'result.json').exists() for r in cases))+'/8')
    for arm in ARMS:
        print(arm+': '+str(sum((r/arm/'complete.json').exists() for r in cases))+'/8'+('（GT仅作诊断）' if arm.endswith('oracle') else '（输入组）'))
    path=OUT/'pipeline.json'
    if path.exists():
        value=read(path);print(f'GPU {value["gpu"]}：{value["status"]}；{value.get("stage","")}；更新 {value["updated"]}')
        print('日志：'+value.get('active_log',''))
        if value.get('error'):print(value['error'])
    for path in sorted(OUT.glob('parallel_gpu*.json')):
        value=read(path);print(f'GPU {value["gpu"]}：{value["status"]}；{value.get("stage","")}；更新 {value["updated"]}')
        print('日志：'+value.get('active_log',''))
        if value.get('error'):print(value['error'])
    records=[r/'status.json' for r in cases if (r/'status.json').exists()]
    if records:
        item=read(max(records,key=lambda p:p.stat().st_mtime))
        print(f'最近：{item["index"]:03d}/{item["seed"]} {item["arm"]} {item["stage"]} {item.get("step","")}')
    print('全部收尾完成：'+str((OUT/'closeout.json').exists() and read(OUT/'closeout.json')['complete']))


if __name__=='__main__':main()
