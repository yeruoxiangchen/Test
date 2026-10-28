from .config import OUT, PARAMS, folder, registry, read


def main():
    rows=registry()['rows'] if (OUT/'registry.json').exists() else []
    cases=[folder(r['index'],s) for r in rows for s in PARAMS['seeds']]
    count=lambda name:sum((p/name).exists() for p in cases)
    print(f'第十轮自然错误纠正：输入 {len(rows)}/16；基础生成 {count("baseline_complete.json")}/32；区域 {count("region_complete.json")}/32')
    print(f'深度证据 {count("dense_provider.json")}/32；纠错决策 {count("input_complete.json")}/32；评估 {count("complete.json")}/32')
    smoke=read(OUT/'smoke_complete.json') if (OUT/'smoke_complete.json').exists() else None
    print('端到端smoke：'+('已通过（不以指标涨跌为通过条件）' if smoke else '待完成'))
    if (OUT/'pipeline.json').exists():
        state=read(OUT/'pipeline.json');print(f'GPU {state["gpu"]}：{state["status"]}；{state.get("stage","")}；更新 {state["updated"]}')
        print('日志：'+state.get('active_log',''))
        if state.get('error'):print(state['error'])
    progress=[p/'status.json' for p in cases if (p/'status.json').exists()]
    if progress:
        latest=max(progress,key=lambda p:p.stat().st_mtime);r=read(latest)
        print(f'最近记录：{r["index"]:03d}/{r["seed"]} {r["stage"]} {r.get("step","")}；{r["time"]}')
    done=read(OUT/'closeout.json').get('complete',False) if (OUT/'closeout.json').exists() else False
    print(f'全部收尾完成：{done}')


if __name__=='__main__':main()
