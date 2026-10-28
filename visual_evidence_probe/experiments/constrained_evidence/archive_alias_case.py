"""Keep the affected debugging case outside the scientific cohort."""
from pathlib import Path
from .config import OUT,folder,write,now


def main():
    source=folder(10,42);archive=OUT/'debug_optimizer_alias_v1/010_42'
    if archive.exists():raise ValueError('debug archive already exists')
    archive.parent.mkdir(parents=True,exist_ok=True);source.rename(archive)
    fixed=[]
    for p in archive.rglob('*'):
        if p.is_symlink():
            target=p.readlink()
            if str(target).startswith(str(source)+'/'):
                new=archive/target.relative_to(source);p.unlink();p.symlink_to(new)
                fixed.append(dict(path=str(p),old_target=str(target),new_target=str(new)))
    write(OUT/'optimizer_alias_correction.json',dict(time=now(),affected_case=[10,42],
        archive=str(archive),retained_case_reason='other completed cases have zero rollbacks',
        bug='Adam load_state_dict reused checkpoint tensors; second consecutive rollback had mutated moments',
        correction='deepcopy saved optimizer state on EVERY load; regression-tested repeated rollback',
        internal_symlinks_rebased=fixed,scientific_results_must_be_rerun=True))
    print(dict(archived=str(archive),kept_all_prior_files=True),flush=True)


if __name__=='__main__':main()
