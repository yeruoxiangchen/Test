from .config import OUT,PROFILE,PARAMS,ARMS,case,read


def main():
    cases=[case(i,s) for i in PARAMS['indices'] for s in PARAMS['seeds']]
    print(f'实验版本：{PROFILE}；输出：{OUT}')
    count=lambda name:sum((p/name).exists() for p in cases)
    print(f'局部再生成：输入规划 {count("plan.json")}/8；SS路径 {count("ss_complete.json")}/8；Shape采样 {count("sampling_complete.json")}/8')
    print(f'输入决策 {count("input_complete.json")}/8；GT评估 {count("result.json")}/8')
    smoke=OUT/'validation/smoke_complete.json';print('原生GPU检查：'+('通过' if smoke.exists() and read(smoke)['passed'] else '待完成'))
    path=OUT/'pipeline.json'
    if path.exists():
        x=read(path);print(f'GPU {x["gpu"]}：{x["status"]}；{x.get("stage","")}；更新{x["updated"]}');print('日志：'+x.get('active_log',''))
        if x.get('error'):print(x['error'])
    for path in sorted(OUT.glob('parallel_gpu*.json')):
        x=read(path);print(f'辅助GPU {x["gpu"]}：{x["status"]}；{x.get("stage","")}；更新{x["updated"]}');print('日志：'+x.get('active_log',''))
        if x.get('error'):print(x['error'])
    records=[p/'status.json' for p in cases if (p/'status.json').exists()]
    if records:
        x=read(max(records,key=lambda p:p.stat().st_mtime));print(f'最近：{x["index"]:03d}/{x["seed"]} {x["stage"]} {x.get("name","")}')
    print('全部收尾完成：'+str((OUT/'closeout.json').exists() and read(OUT/'closeout.json')['complete']))


if __name__=='__main__':main()
