"""Read-only package overlay for the older vLLM environment; never pip-upgrades it."""
from pathlib import Path
import json
from kb.config import Settings

source = Path('/home/zjr/anaconda3/envs/chartvlm/lib/python3.11/site-packages')
target = Settings().root/'judge_packages'
target.mkdir(exist_ok=True)
linked = {}
for name in ('transformers','tokenizers'):
    paths = [source/name, *source.glob(name+'-*.dist-info')]
    if len(paths) != 2 or any(not p.is_dir() for p in paths):
        raise RuntimeError('Expected one installed package and one metadata directory: '+name)
    for p in paths:
        dest = target/p.name
        if dest.exists() and dest.resolve()!=p.resolve():
            raise RuntimeError('Refusing to overwrite unrelated overlay: '+str(dest))
        if not dest.exists():dest.symlink_to(p, target_is_directory=True)
        linked[dest.name]=str(p)
(target/'receipt.json').write_text(json.dumps({'read_only_links':linked,'shared_environment_modified':False},indent=2)+'\n')
print(json.dumps(linked,indent=2))
