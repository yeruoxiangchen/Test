from .config import OUT,PARAMS,folder,read


def main():
    roots=[folder(i,s) for i in PARAMS['indices'] for s in PARAMS['seeds']]
    count=lambda filename:sum((r/filename).exists() for r in roots)
    print(f'TRELLIS.2原生几何纠错：SS {count("ss.pt")}/8；Shape {count("shape_complete.json")}/8；原生mesh {count("generation_complete.json")}/8')
    print(f'输入对齐 {count("fit.json")}/8；局部纠错 {count("input_complete.json")}/8；评估 {count("result.json")}/8')
    smoke=OUT/'validation/smoke_complete.json'
    print('原生解码/梯度/恢复smoke：'+('通过' if smoke.exists() and read(smoke)['passed'] else '待完成'))
    if (OUT/'pipeline.json').exists():
        s=read(OUT/'pipeline.json');print(f'GPU {s["gpu"]}：{s["status"]}；{s.get("stage","")}；更新 {s["updated"]}');print('日志：'+s.get('active_log',''))
        if s.get('error'):print(s['error'])
    records=[r/'status.json' for r in roots if (r/'status.json').exists()]
    if records:
        last=read(max(records,key=lambda p:p.stat().st_mtime));print(f'最近记录：{last["index"]:03d}/{last["seed"]} {last["stage"]} {last.get("step","")}；{last["time"]}')
    print('全部收尾完成：'+str((OUT/'closeout.json').exists() and read(OUT/'closeout.json')['complete']))


if __name__=='__main__':main()
