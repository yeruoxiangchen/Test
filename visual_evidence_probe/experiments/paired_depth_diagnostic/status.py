from .config import OUT,PARAMS,ARMS,folder,read


def main():
    cases=[folder(i,s) for i in PARAMS['indices'] for s in PARAMS['seeds']]
    sealed=sum((p/'optimization_complete.json').exists() for p in cases)
    trajectories=sum((p/arm/'complete.json').exists() for p in cases for arm in ARMS)
    metrics=sum((p/arm/'result.json').exists() for p in cases for arm in ARMS)
    print(f'第九轮机制诊断：优化案例 {sealed}/8；四组轨迹 {trajectories}/32；四组指标 {metrics}/32')
    print('全部组均使用GT交点共同支持；GT深度仅用于oracle目标；不是实际方法成绩。')
    for path in sorted(OUT.glob('pipeline_gpu*.json')):
        try:d=read(path)
        except (ValueError,FileNotFoundError):continue
        print(f'GPU {d["gpu"]}：{d["status"]}；{d.get("stage")}; 更新 {d["updated"]}');print('日志：'+d.get('active_log',''))
        if d.get('stage')=='closeout':print('收尾包含2次同进程数值重复对照，不计入32组主实验。')
        if d.get('error'):print(d['error'])
    updates=sorted((OUT/'objects').glob('*/*/status.json'),key=lambda p:p.stat().st_mtime)
    if updates:
        d=read(updates[-1]);print(f'最近记录：{d["index"]:03d}/{d["seed"]} {d["stage"]} {d.get("arm","")} {d.get("step",0)}步；{d["time"]}')
    print('全部诊断收尾完成：'+str((OUT/'closeout.json').exists()))


if __name__=='__main__':main()
