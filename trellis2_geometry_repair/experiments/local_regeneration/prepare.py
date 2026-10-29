from pathlib import Path

from .config import OUT,BASE,PARAMS,ARMS,PROFILE,case,source,read,write,digest,setup,snapshot


def main():
    setup();protocol=Path(__file__).with_name('PROTOCOL_V1.md' if PROFILE=='v1' else 'PROTOCOL.md')
    identity=dict(params=PARAMS,arms=list(ARMS),base=str(BASE),protocol_sha256=digest(protocol),development_only=True)
    path=OUT/'protocol.json'
    if path.exists() and read(path)!=identity:raise RuntimeError('local regeneration protocol changed')
    write(path,identity)
    for index in PARAMS['indices']:
        for seed in PARAMS['seeds']:
            root=case(index,seed);root.mkdir(parents=True,exist_ok=True)
            paths=[source(index,seed)/name for name in ('original_latent.pt','ss.pt','fit.json')]
            paths+=[BASE/'data'/f'{index:03d}'/'observations.json',BASE/'data'/f'{index:03d}'/'reference_context.pt']
            value=dict(files=[dict(path=str(p),sha256=digest(p)) for p in paths],gt_used=False)
            if (root/'binding.json').exists() and read(root/'binding.json')!=value:raise RuntimeError('base identity changed')
            write(root/'binding.json',value)
    snapshot();print('Native local-regeneration protocol and inputs prepared.',flush=True)


if __name__=='__main__':main()
