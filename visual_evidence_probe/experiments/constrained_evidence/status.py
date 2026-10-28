from .config import OUT,PARAMS,ARMS,folder,read


def main():
    inputs=sum((folder(i,s)/'input_complete.json').exists() for i in PARAMS['dev_indices'] for s in PARAMS['seeds'])
    results=sum((folder(i,s)/arm/'result.json').exists() for i in PARAMS['dev_indices'] for s in PARAMS['seeds'] for arm in ARMS)
    print(f'第八轮输入决策：{inputs}/16；三组指标：{results}/48')
    for p in sorted(OUT.glob('pipeline_gpu*.json')):
        try:d=read(p)
        except (ValueError,FileNotFoundError):continue
        print(f'GPU {d["gpu"]}：{d["status"]}；{d.get("stage")}; 更新 {d["updated"]}');print('日志：'+d.get('active_log',''))
        if d.get('error'):print(d['error'])
    updates=sorted((OUT/'objects').glob('*/*/status.json'),key=lambda p:p.stat().st_mtime)
    if updates:
        latest=read(updates[-1]);detail=latest.get('stage','')
        if latest.get('arm'):detail+=f' / {latest["arm"]} / {latest.get("step",0)}/{PARAMS["steps"]}步 / 回退{latest.get("rollbacks",0)}组'
        print(f'最近记录：{latest["index"]:03d}/{latest["seed"]} {detail}；{latest["time"]}')
    print('开发实验收尾完成：'+str((OUT/'closeout.json').exists()))


if __name__=='__main__':main()
