from pathlib import Path
from .config import OUT,V2,PARAMS,ARMS,case,previous,original,digest,read,write,setup,snapshot

def main():
    setup()
    if not read(V2/'closeout.json')['complete']:raise RuntimeError('V2 must be frozen and complete')
    identity=dict(params=PARAMS,arms=list(ARMS),v2=str(V2),protocol_sha256=digest(Path(__file__).with_name('PROTOCOL.md')),development_only=True)
    path=OUT/'protocol.json'
    if path.exists() and read(path)!=identity:raise RuntimeError('V3 protocol changed; create a separate experiment')
    write(path,identity)
    for i in PARAMS['indices']:
        for s in PARAMS['seeds']:
            files=[previous(i,s)/n for n in ('plan.pt','input_complete.json',PARAMS['support']+'.pt')]
            files += [original(i,s)/'original_latent.pt']
            value=dict(inputs=[dict(path=str(p),sha256=digest(p)) for p in files],GT_used=False)
            path=case(i,s)/'binding.json'
            if path.exists() and read(path)!=value:raise RuntimeError('frozen input/support binding changed')
            write(path,value)
    snapshot();print('V3 input-only protocol and fixed supports prepared.',flush=True)

if __name__=='__main__':main()
