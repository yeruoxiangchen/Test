"""Read-only progress, safe to run while the background experiment writes."""
from .config import OUT,PARAMS,ARMS,folder,registry,read


def main():
    rows=registry()['rows'];expected=len(rows)*len(PARAMS['seeds'])
    inputs=sum((folder(r['index'],s)/'input_complete.json').exists() for r in rows for s in PARAMS['seeds'])
    evaluated=sum((folder(r['index'],s)/'complete.json').exists() for r in rows for s in PARAMS['seeds'])
    arms=sum((folder(r['index'],s)/arm/'result.json').exists() for r in rows for s in PARAMS['seeds'] for arm in ARMS)
    print(f'输入决策：{inputs}/{expected}；指标案例：{evaluated}/{expected}；四组指标：{arms}/{expected*4}')
    for path in sorted(OUT.glob('pipeline_gpu*.json')):
        try:state=read(path)
        except (ValueError,FileNotFoundError):continue
        print(f'GPU {state["gpu"]}：{state["status"]}；{state.get("stage")}; 更新 {state["updated"]}')
        print('日志：'+state.get('active_log',''))
        if state.get('error'):print('错误：'+state['error'])
    print('全部收尾完成：'+str((OUT/'closeout.json').exists()))


if __name__=='__main__':main()
