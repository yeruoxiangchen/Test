from .config import OUT,PARAMS,case,read

def main():
    paths=[case(i,s) for i in PARAMS['indices'] for s in PARAMS['seeds']]
    count=lambda name:sum((p/name).exists() for p in paths)
    print(f'V3原生几何引导；输出：{OUT}')
    print(f'实际输入生成 {count("generation_method.json")}/8；输入决策封存 {count("method_sealed.json")}/8；配对诊断 {count("diagnostic_complete.json")}/8；评估 {count("result.json")}/8')
    path=OUT/'validation/smoke_complete.json';print('原生GPU检查：'+('通过' if path.exists() and read(path)['passed'] else '待完成'))
    for path in sorted((OUT/'workers').glob('gpu*.json')):
        value=read(path);print(f'GPU {value["gpu"]}：{value["status"]}；{value.get("stage","")}；更新{value["updated"]}');print('日志：'+value.get('active_log',''))
        if value.get('error'):print(value['error'])
    for p in paths:
        progress=p/'generation_method_progress.json'
        if progress.exists() and not (p/'generation_method.json').exists():
            specs=read(progress)['candidates'];print(f'{p.parent.name}/{p.name} 方法候选 {len(specs)}/10；最近 '+specs[-1]['name'])
    print('全部收尾完成：'+str((OUT/'closeout.json').exists() and read(OUT/'closeout.json')['complete']))

if __name__=='__main__':main()
