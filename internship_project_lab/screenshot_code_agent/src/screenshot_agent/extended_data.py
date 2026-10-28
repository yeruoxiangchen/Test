"""Predeclared, offline SWE-bench derivatives with per-task dependency prefixes."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

from .benchmark import official_grade
from .config import RuntimeConfig
from .control import Events, atomic_json
from .data import render_screenshot
from .official_data import DATASET_REVISION, install_test_patch, source_archive, test_command
from .repository import git
from .runtime import VenvSandbox, create_sandbox

QUOTAS = {'pytest-dev/pytest':8, 'sympy/sympy':20, 'sphinx-doc/sphinx':11,
          'pylint-dev/pylint':6, 'pallets/flask':2, 'mwaskom/seaborn':3}


def select_extended(parquet, output, excluded):
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet).to_pylist()
    selected, reserves = [], []
    for repo,quota in QUOTAS.items():
        candidates = []
        for row in sorted((r for r in rows if r['repo']==repo),key=lambda r:r['instance_id']):
            if row['instance_id'] in excluded:
                continue
            blocks = [m for m in re.finditer(r'```[^\n]*\n(.*?)```',row['problem_statement'],re.S)
                      if 2<=len(m[1].splitlines())<=35 and len(m[1])<=2200]
            if not blocks:
                continue
            row.update(sca_split='dev' if repo in ['pytest-dev/pytest','sympy/sympy'] else 'test',
                       screenshot_block=blocks[0][1],screenshot_span=list(blocks[0].span()))
            candidates.append(row)
        if len(candidates)<quota:
            raise ValueError(f'Insufficient predeclared source candidates: {repo}')
        selected.extend(candidates[:quota])
        if repo=='sympy/sympy':
            reserves.extend(candidates[quota:quota+10])
    records = [dict(r,selection_role='primary') for r in selected]+[dict(r,selection_role='environment_reserve') for r in reserves]
    output.mkdir(parents=True,exist_ok=True)
    old = output/'selected_records.json'
    if old.exists() and json.loads(old.read_text())!=records:
        raise ValueError('Selection already frozen; use a new version')
    atomic_json(old,records)
    atomic_json(output/'selection.json',{'revision':DATASET_REVISION,'dataset_sha256':hashlib.sha256(parquet.read_bytes()).hexdigest(),
        'primary_ids':[r['instance_id'] for r in selected],'reserve_ids':[r['instance_id'] for r in reserves],
        'excluded_previously_seen_ids':sorted(excluded),'quotas':QUOTAS,'target_validated_tasks':50,
        'split_rule':'Previously inspected pytest/sympy repositories are development; the other four repositories are test. Grouping precedes all variants.',
        'reserve_rule':'Only environment/gold preflight failures may use the predeclared ordered reserves, before any model results. All primary and reserve preflight outcomes retained.',
        'scope':'Offline Python repositories. Requests tasks requiring live HTTP excluded before selection; no quality-based filtering.'})
    return records


def profile(record):
    repo,version = record['repo'],record['version']
    if repo=='pylint-dev/pylint':
        return 'pylint_'+version.replace('.','_')
    if repo=='sphinx-doc/sphinx':
        return 'sphinx_old' if int(version.split('.')[0])<=4 else 'sphinx_'+version.split('.')[0]
    return repo.split('/')[-1]


COMMON = ['setuptools==68.0.0','setuptools-scm==6.4.2','wheel==0.42.0','pytest==7.4.4',
          'pytest-mock==3.12.0','pytest-xdist==3.6.1','hypothesis==6.88.1','attrs==23.2.0',
          'packaging==23.2','iniconfig==1.1.1','py==1.11.0','toml==0.10.2','tomli==2.0.1',
          'typing-extensions==4.12.2','six==1.16.0','mpmath==1.2.1','flit_core==3.9.0']


def requirements(name):
    extra = []
    if name.startswith('pylint'):
        astroid = {'pylint_2_13':'2.9.3','pylint_2_14':'2.11.6','pylint_2_15':'2.11.7'}[name]
        extra = ['astroid=='+astroid,'mccabe=='+('0.7.0' if name.endswith('15') else '0.6.1'),
                 'isort==5.13.2','dill==0.3.7','platformdirs==3.11.0','tomlkit==0.12.3','GitPython==3.1.43']
    if name.startswith('sphinx'):
        extra = ['Jinja2=='+('2.11.3' if name=='sphinx_old' else '3.1.4'),
                 'MarkupSafe=='+('2.0.1' if name=='sphinx_old' else '3.0.2'),
                 'docutils=='+('0.21.2' if name=='sphinx_old' else '0.18.1' if name=='sphinx_5' else '0.20.1'),
                 'sphinxcontrib-applehelp==1.0.2','sphinxcontrib-devhelp==1.0.2','sphinxcontrib-htmlhelp==2.0.0',
                 'sphinxcontrib-jsmath==1.0.1','sphinxcontrib-qthelp==1.0.3','sphinxcontrib-serializinghtml==1.1.5',
                 'Babel==2.16.0','imagesize==1.4.1','Pygments==2.18.0','snowballstemmer==2.2.0',
                 'requests==2.32.3','alabaster==0.7.13','html5lib==1.1','Cython==0.29.37']
    if name=='flask':
        extra = ['Werkzeug==2.3.8','Jinja2==3.1.2','MarkupSafe==2.1.1','itsdangerous==2.1.2',
                 'click==8.1.7','blinker==1.6.3','asgiref==3.8.1','python-dotenv==1.0.0']
    if name=='seaborn':
        extra = ['numpy==1.25.2','pandas==2.0.0','scipy==1.11.4','matplotlib==3.7.2','Pillow==10.4.0','python-dateutil==2.8.2']
    return COMMON+extra


def setup_profiles(records, base, destination):
    destination.mkdir(parents=True,exist_ok=True)
    for name in sorted({profile(r) for r in records}):
        prefix = destination/name
        lock = destination/(name+'.requirements.txt')
        lock.write_text('\n'.join(requirements(name))+'\n')
        state = prefix/'sca_profile.json'
        if state.exists() and json.loads(state.read_text())['requirements']==requirements(name):
            continue
        if not prefix.exists():
            shutil.copytree(base,prefix,symlinks=True)
        subprocess.run([str(prefix/'bin/python'),'-m','pip','install','--only-binary=:all:',
            '--cache-dir',str(destination/'pip_cache'),'-r',str(lock)],check=True)
        frozen = subprocess.check_output([str(prefix/'bin/python'),'-m','pip','freeze'],text=True)
        (destination/(name+'.freeze.txt')).write_text(frozen)
        atomic_json(prefix/'sca_profile.json',{'profile':name,'requirements':requirements(name)})


class SetupSandbox(VenvSandbox):
    """Only construction may write its private, task-specific dependency prefix."""

    def command(self, argv, mounts=None):
        command = super().command(argv,mounts)
        i = next(i for i,v in enumerate(command[:-2]) if v=='--ro-bind' and command[i+1]==str(self.prefix))
        command[i] = '--bind'
        return command


def export_source(record, cache, destination):
    """Materialize only audited internal file links in the Pylint fixtures.

    Agent snapshots still contain no symlinks. The old strict archive importer
    and every previous dataset remain unchanged.
    """
    if record['repo']!='pylint-dev/pylint':
        return source_archive(record,cache,destination),[]
    archive=cache/(record['repo'].replace('/','_')+'_'+record['base_commit']+'.tar.gz')
    if not archive.exists():
        raise ValueError('Download the pinned Pylint source archive before export')
    links=[]
    with tarfile.open(archive) as source:
        members=source.getmembers()
        roots={Path(m.name).parts[0] for m in members}
        if len(roots)!=1:
            raise ValueError('Source archive must have one root')
        indexed={}
        for member in members:
            full=Path(member.name)
            rel=Path(*full.parts[1:])
            if full.is_absolute() or '..' in full.parts or not (member.isfile() or member.isdir() or member.issym()):
                raise ValueError('Unsafe source archive member')
            if rel!=Path('.'):
                if rel.as_posix() in indexed:
                    raise ValueError('Duplicate source archive member')
                indexed[rel.as_posix()]=member
        resolved={}
        for name,member in indexed.items():
            target,seen=name,set()
            while indexed[target].issym():
                if target in seen:
                    raise ValueError('Cyclic source link')
                seen.add(target)
                link=indexed[target].linkname
                target=posixpath.normpath(posixpath.join(posixpath.dirname(target),link))
                if target.startswith('/') or target=='..' or target.startswith('../') or target not in indexed:
                    raise ValueError('Source link escapes archive or has no target')
            if member.issym():
                if not indexed[target].isfile():
                    raise ValueError('Only internal file links may be materialized')
                links.append({'path':name,'target':target})
            resolved[name]=indexed[target]
        destination.mkdir(parents=True,exist_ok=True)
        for name,member in resolved.items():
            path=destination/name
            if member.isdir():
                path.mkdir(parents=True,exist_ok=True)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                with source.extractfile(member) as src,path.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
                path.chmod(member.mode & 0o777)
    return hashlib.sha256(archive.read_bytes()).hexdigest(),links


def scored_test_command(record,repo,original):
    if record['repo']=='sympy/sympy':
        # The official SymPy parser scores function names, not pytest node IDs.
        paths=[p for p in shlex.split(original) if p.endswith('.py') and (repo/p).is_file()]
        command=original
    else:
        wanted=[]
        for key in ['FAIL_TO_PASS','PASS_TO_PASS']:
            wanted.extend(json.loads(record[key]) if isinstance(record[key],str) else record[key])
        paths=sorted({name.split('::')[0] for name in wanted})
        command='python -m pytest -rA '+' '.join(shlex.quote(p) for p in paths)
        if record['repo']=='pytest-dev/pytest':
            # These historical pytest checkouts predate the installed xdist API.
            command='PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 '+command
    if not paths or any(Path(p).is_absolute() or '..' in Path(p).parts or not (repo/p).is_file() for p in paths):
        raise ValueError('Scored test paths are absent or unsupported')
    return command,paths


def build_extended(output, tasks, harness, profiles, cache, timeout=180):
    output,tasks,harness,profiles,cache = [p.resolve() for p in (output,tasks,harness,profiles,cache)]
    records = json.loads((output/'selected_records.json').read_text())
    for record in records:
        upstream=tasks/'tasks'/record['instance_id']
        for name,key in [('test.patch','test_patch'),('gold.patch','patch')]:
            if (upstream/name).read_text().strip()!=record[key].strip():
                raise ValueError(f"Task/HF patch mismatch before construction: {record['instance_id']}")
    audits = []
    validated = []
    for number,record in enumerate(records,1):
        instance = record['instance_id']
        folder = output/instance
        if (folder/'audit.json').exists():
            item = json.loads((folder/'audit.json').read_text())
            audits.append(item)
            if item['status']=='validated':
                validated.append(item['manifest_row'])
            continue
        folder.mkdir(exist_ok=True)
        item = {'instance_id':instance,'repo':record['repo'],'selection_role':record['selection_role'],'status':'error'}
        print(f'Extended preflight [{number}/{len(records)}] {instance}',flush=True)
        try:
            repo = folder/'repo'
            if repo.exists():
                raise ValueError('Incomplete task folder retained; archive it before retrying this preflight')
            item['source_archive_sha256'],item['materialized_internal_links'] = export_source(record,cache,repo)
            upstream = tasks/'tasks'/instance
            for name,key in [('test.patch','test_patch'),('gold.patch','patch')]:
                if (upstream/name).read_text().strip()!=record[key].strip():
                    raise ValueError('Task/HF patch mismatch')
            install_test_patch(repo,record['test_patch'])
            prefix = profiles/'tasks'/instance
            if prefix.exists():
                raise ValueError('Task prefix already exists without a completed audit')
            prefix.parent.mkdir(exist_ok=True)
            shutil.copytree(profiles/profile(record),prefix,symlinks=True)
            runtime = RuntimeConfig(backend='venv',prefix=str(prefix))
            setup = SetupSandbox(repo,Events(folder/'setup_events.jsonl'),[],timeout,repo,runtime=runtime)
            version = record['version']+'.0' if record['version'].count('.')==1 else record['version']
            command = 'SETUPTOOLS_SCM_PRETEND_VERSION='+shlex.quote(version)+' python -m pip install --no-deps --no-build-isolation --disable-pip-version-check -e .'
            installed = setup.run(command,log_path=folder/'installation.log')
            setup.cancel()
            if installed['returncode'] or installed['timed_out']:
                raise RuntimeError('Offline sandboxed package installation failed; inspect installation.log')
            original = test_command((upstream/'eval.sh').read_text())
            # Preserve all official scored test files. Drop tox packaging and
            # non-test fixture arguments; immutable task installation is above.
            command,paths = scored_test_command(record,repo,original)
            body = "#!/bin/bash\nset -uo pipefail\nexport MPLBACKEND=Agg\nprintf '\\n>>>>> Start Test Output\\n'\n"+command+"\nSCA_TEST_EXIT=$?\nprintf '\\n>>>>> End Test Output\\n>>>>> Test Exit Code: %s\\n' \"$SCA_TEST_EXIT\"\nexit \"$SCA_TEST_EXIT\"\n"
            (repo/'.sca_eval.sh').write_text(body)
            protected = sorted({'.sca_eval.sh',*paths,*re.findall(r'^\+\+\+ b/(.+)$',record['test_patch'],re.M),
                                *[str(p.relative_to(repo)) for p in repo.rglob('conftest.py')]})
            for name in ['src/_pytest/_version.py']:
                if (repo/name).exists():
                    protected.append(name)
            metadata = {'record':record,'harness_checkout':str(harness),'task_checkout':str(tasks),
                'original_test_command':original,'test_command':command,'runtime_adaptation':'Offline, per-task Python prefix and all official scored test files; tox packaging/fixture arguments removed. Not original Docker protocol.'}
            atomic_json(folder/'benchmark.json',metadata)
            (folder/'reference.patch').write_text(record['patch'])
            begin,end = record['screenshot_span']
            problem = record['problem_statement'][:begin]+'[Read the attached screenshot for this original issue code/output block.]'+record['problem_statement'][end:]
            visible = record['screenshot_block'].replace('\r\n','\n').rstrip()+'\n'
            for theme in ['light','dark']:
                render_screenshot(visible,folder/f'screenshot_{theme}.png',theme)
            task = {'task_id':instance,'repo':'repo','problem':problem,'test_command':'bash .sca_eval.sh',
                'protected_paths':protected,'split':record['sca_split'],'group_id':record['repo'],
                'provenance':'swebench_lite_offline_screenshot_derivative','benchmark':'benchmark.json',
                'reference_patch':'reference.patch','runtime':runtime.model_dump(),
                'images':[{'image_id':'shot','path':'screenshot_light.png','reference':'screenshot_light.txt','layout':'screenshot_light.layout.json'}]}
            atomic_json(folder/'task_light.json',task)
            row = {'task_id':instance,'task_file':instance+'/task_light.json','split':record['sca_split'],
                   'group_id':record['repo'],'provenance':task['provenance']}
            item['manifest_row'] = row
            events = Events(folder/'preflight_events.jsonl')
            baseline_check=folder/'baseline_check'
            shutil.copytree(repo,baseline_check)
            sb = create_sandbox(baseline_check,events,protected,timeout,repo,runtime)
            baseline = sb.run('bash .sca_eval.sh',log_path=folder/'baseline_test.log')
            sb.cancel()
            before = official_grade(folder/'benchmark.json',folder/'baseline_test.log')
            gold_repo = folder/'gold_check'
            shutil.copytree(repo,gold_repo)
            git(gold_repo,'apply','--binary','-',stdin=record['patch'].encode())
            gold = create_sandbox(gold_repo,events,protected,timeout,repo,runtime)
            execution = gold.run('bash .sca_eval.sh',log_path=folder/'reference_test.log')
            gold.cancel()
            after = official_grade(folder/'benchmark.json',folder/'reference_test.log')
            item.update(baseline=before,reference=after,baseline_timeout=baseline['timed_out'],reference_timeout=execution['timed_out'])
            failure = before.get('tests_status',{}).get('FAIL_TO_PASS',{}).get('failure')
            item['status'] = 'validated' if before['parsed'] and failure and after['resolved'] and not baseline['timed_out'] and not execution['timed_out'] else 'environment_failed'
            if item['status']=='validated':
                validated.append(row)
            shutil.rmtree(gold_repo)
            shutil.rmtree(baseline_check)
        except Exception as error:
            item.update(status='environment_failed',error=str(error),error_type=type(error).__name__)
        audits.append(item)
        atomic_json(folder/'audit.json',item)
        atomic_json(output/'build_status.json',{'completed':len(audits),'total':len(records),'validated':len(validated),'current':instance})
        print(f"  {item['status']}: {item.get('error','')}",flush=True)
    selected = validated[:50]
    for name,rows in [('tasks.jsonl',selected),('validated_pool.jsonl',validated)]:
        (output/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report = {'selected_pool':len(records),'validated_pool':len(validated),'evaluation_tasks':len(selected),
              'evaluation_repositories':len({r['group_id'] for r in selected}),'audits':audits,
              'official_docker_protocol':False,'test_visibility':'Visible protected test files; external reference source patch and grading metadata.',
              'scope':'Infrastructure/gold preflight only; no model outcomes used for task admission.'}
    atomic_json(output/'build_report.json',report)
    return report
