"""Single-agent repair workflow with repository retrieval and executable feedback.

Planner/limited retry design references wingwen/CoreCoder-Agent. Unlike its
text-only completion checks, public tests are executed on independent copies.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import signal
import threading
import time
import traceback
from pathlib import Path

from corecoder import Agent

from .config import ImageInput
from .context import RepairContext
from .control import BudgetExceeded, ContextWindowExceeded, Events, NoProgress, RunCancelled, atomic_json
from .harness import PROTOCOL, HarnessAPIModel, QwenTokenCounter
from .hidden_eval import grade_hidden, validate_hidden_task
from .images import ImageStore, OCRReader, OracleReader, VisionReader
from .models import APIModel, ModelOutputError
from .repo_search import RepoSearch
from .repository import export_patch, prepare_repository, replay_patch, validate_tree
from .runtime import create_sandbox
from .scale_data import tree_hash
from .tools import INTEGER, STRING, FunctionTool, build_tools, filesystem_result, schema
from .workflow_config import PhaseEnded, WorkflowBudget
from .workflow_feedback import VisibleFeedback
from .workflow_view import export_delivery, render_run

METHOD = 'corecoder_retrieval_workflow_v2'
PHASES = [('locate','找到与问题有关的文件、函数和调用关系'),
          ('reproduce','用已有测试或复现脚本检查问题，记录观察结果'),
          ('repair','修改生产代码并检查相关行为')]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def implementation():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


class SerialReader:
    """Vision requests share one durable pending-token reservation."""
    def __init__(self, reader):
        self.reader, self.lock = reader, threading.RLock()
        self.name = reader.name

    def read(self, *args, **kwargs):
        with self.lock:
            return self.reader.read(*args, **kwargs)


def json_object(text):
    start, end = text.find('{'), text.rfind('}')
    value = json.loads(text[start:end+1])
    if not isinstance(value,dict):
        raise ValueError('Expected JSON object')
    return value


class WorkflowEvents(Events):
    def __init__(self, directory):
        super().__init__(directory/'events.jsonl')
        self.directory, self.updated = directory, 0.0

    def emit(self, event, **fields):
        super().emit(event,**fields)
        with self.lock:
            if (event not in {'model_response','tool_start','tool_result','model_error'}
                    or time.monotonic()-self.updated < 1):
                return
            self.updated = time.monotonic()
            description = fields.get('name') or {'model_response':'模型返回，继续执行',
                                               'model_error':'模型请求异常'}.get(event,event)
            atomic_json(self.directory/'progress.json',{'event':event,'description':description,'time':time.time()})
            render_run(self.directory)


class RepairWorkflow:
    def __init__(self, task, config, directory, *, reader_name='vlm', resume=False,
                 counter=None, api_factory=HarnessAPIModel, grader=grade_hidden, model_inference=True):
        self.task, self.config, self.directory = task, config, Path(directory)
        self.reader_name, self.resume = reader_name, resume
        self.counter, self.api_factory, self.grader = counter, api_factory, grader
        self.model_inference = model_inference
        self.sandbox = self.budget = self.agent = None
        self.state = {}
        self.tree = {'max_files':task.runtime.max_repo_files,'max_bytes':task.runtime.max_repo_bytes,
                     'allowed_symlinks':task.runtime.allowed_symlinks}
        self.events = WorkflowEvents(self.directory)
        self.feedback = VisibleFeedback(self)

    def save(self, **fields):
        self.state.update(fields,updated_at=time.time())
        atomic_json(self.directory/'workflow.json',self.state)
        render_run(self.directory)

    def phase(self, name):
        self.save(stage=name)
        self.events.emit('workflow_stage',stage=name)

    def setup(self):
        if self.task.benchmark:
            validate_hidden_task(self.task)
        identity = digest({'task':self.task.model_dump(),'config':self.config.model_dump(),'reader':self.reader_name})
        source = implementation()
        if self.resume:
            self.state = json.loads((self.directory/'workflow.json').read_text())
            if self.state['identity'] != identity or self.state['source'] != source:
                raise ValueError('Task, workflow configuration or implementation changed; use a new run directory')
            baseline, workspace = self.directory/'baseline', self.directory/'workspace'
            if tree_hash(baseline) != self.state['baseline_hash']:
                raise ValueError('Private baseline changed; cannot resume')
            validate_tree(workspace,**self.tree)
            if (self.directory/'result.json').exists():
                result = json.loads((self.directory/'result.json').read_text())
                if result['status'] != 'paused':
                    raise ValueError('Completed attempts are retained; resume only interrupted/paused work')
                archive = self.directory/'attempts'
                archive.mkdir(exist_ok=True)
                (self.directory/'result.json').rename(archive/f'paused_{time.time_ns()}.json')
            inputs = [ImageInput.model_validate(x) for x in self.state.get('frozen_images',[])]
            self.model_inference = self.state['model_inference']
        else:
            atomic_json(self.directory/'task.json',self.task.model_dump())
            atomic_json(self.directory/'config.json',self.config.model_dump())
            self.save(task_id=self.task.task_id, problem=self.task.problem, reader=self.reader_name,status='running',stage='snapshot',
                      method=METHOD,identity=identity,source=source,model_inference=self.model_inference,
                      started_at=time.time(),plan=[],readings=[],checks=[],reviews=[],completed_phases=[],
                      attempted_phases=[],retry_count=0)
            baseline, workspace = prepare_repository(Path(self.task.repo),self.directory,**self.tree)
            # Freeze user inputs too, so resume does not depend on their originals.
            assets = self.directory/'inputs'
            assets.mkdir()
            inputs = []
            for index, original in enumerate(self.task.images):
                values = original.model_dump()
                for key in ('path','reference','layout'):
                    if values.get(key):
                        src = Path(values[key])
                        target = assets/f'{index}_{key}{src.suffix}'
                        shutil.copy2(src,target)
                        values[key] = str(target)
                inputs.append(ImageInput.model_validate(values))
            self.save(baseline_hash=tree_hash(baseline),frozen_images=[x.model_dump() for x in inputs])
        self.baseline, self.workspace = baseline, workspace
        self.budget = WorkflowBudget(self.config,self.events,self.counter or QwenTokenCounter(self.config.tokenizer_path),
                                     self.directory/'budget.json',resume=self.resume)
        self.sandbox = create_sandbox(workspace,self.events,self.task.protected_paths,
            self.config.limits.command_timeout_seconds,baseline,self.task.runtime)
        self.sandbox.probe()
        self.images = ImageStore(inputs,self.directory/'screenshots',self.events)
        self.reader = None
        if inputs and not self.state.get('grading_started'):
            self.reader = (OracleReader(self.images) if self.reader_name=='oracle' else
                           OCRReader(self.images) if self.reader_name=='ocr' else
                           VisionReader(self.images,APIModel(self.config.vision,self.budget,self.events,kind='vision')))
            self.reader = SerialReader(self.reader)
        self.save(status='running')

    def make_tools(self):
        tools = build_tools(self.sandbox,self.images,self.reader,self.budget,self.events,self.task.test_command,
                            agent_policy=self.config.agent,screenshot_policy=self.config.screenshots)
        if not self.task.images:
            tools = [t for t in tools if t.name not in {'read_screenshot','crop_screenshot'}]
        lock = next(t.lock for t in tools if t.name == 'edit_file')
        for tool in tools:
            if tool.name == 'run_tests':
                tool.function = lambda: self.check_summary(self.feedback.check('Agent请求公共验证'))
                tool.description = 'Compare the public test baseline and rerun any frozen issue reproducer.'
        def search(query, top_k=6):
            # Index refresh runs at phase boundaries, after commands have joined.
            # During model/tool execution we read only this trusted in-memory index.
            return {'matches':self.index.search(query,top_k,refresh=False),
                    'note':'Indexed at the last phase boundary. Read current source before editing.'}
        def repo_map(query=''):
            return self.index.repo_map(query,self.config.workflow.map_chars,refresh=False)
        def apply_patch(file_path, patch):
            return filesystem_result(self.sandbox.filesystem('apply_patch',{'file_path':file_path,'patch':patch},
                                      timeout=self.budget.remaining_seconds(),feedback_version=2))
        tools += [FunctionTool('repo_search','Find relevant code using identifiers, error text and file hints.',
                    schema({'query':STRING,'top_k':INTEGER},['query']),search,self.budget,self.events,readable=True),
                  FunctionTool('repo_map','Show repository files and important source symbols.',
                    schema({'query':STRING}),repo_map,self.budget,self.events,readable=True),
                  FunctionTool('apply_patch','Apply multiple exact SEARCH/REPLACE blocks to one file; all anchors must match.',
                    schema({'file_path':STRING,'patch':STRING},['file_path','patch']),apply_patch,
                    self.budget,self.events,lock,readable=True),
                  FunctionTool('register_reproduction',
                    'Freeze a standalone Python assertion script and run it against the ORIGINAL source. '
                    'The expected assertion must describe the issue; import/runtime errors do not count. '
                    'The controller reruns identical script bytes after repair.',
                    schema({'script_path':STRING,'expected_failure':STRING},['script_path','expected_failure']),
                    self.feedback.register_reproduction,self.budget,self.events,lock,readable=True)]
        cwd = self.sandbox.description()['repository_mount']
        for tool in tools:
            tool.description = tool.description.replace('/workspace',cwd)
        return tools

    def refresh_index(self):
        validate_tree(self.workspace,**self.tree)
        self.index = RepoSearch(self.workspace,self.directory/'index/repository.json')
        evidence = '\n'.join(str(r.get('raw_text','')) for r in self.state['readings'])
        self.query = self.task.problem+'\n'+evidence
        mapping = self.index.repo_map(self.query,self.config.workflow.map_chars)
        atomic_json(self.directory/'repository_map.json',mapping)
        atomic_json(self.directory/'retrieval.json',self.index.search(self.query,self.config.workflow.retrieval_top_k,refresh=False))
        return mapping

    def initialize_agent(self):
        self.phase('reading')
        if self.task.images:
            already_read = {r['image_id'] for r in self.state['readings']}
            for item in self.task.images[:self.config.limits.max_vision_calls]:
                if item.image_id in already_read:
                    continue
                self.budget.tool('read_screenshot',{'image_id':item.image_id})
                self.budget.screenshot_calls += 1
                self.budget.save()
                reading = self.reader.read(item.image_id).model_dump()
                self.state['readings'].append({'image_id':item.image_id,**reading})
                self.save()
        self.phase('indexing')
        mapping = self.refresh_index()
        self.api = self.api_factory(self.config.code,self.budget,self.events,policy=self.config.agent)
        if not self.state['plan']:
            self.phase('planning')
            default = [{'phase':phase,'description':desc,'status':'pending'} for phase,desc in PHASES]
            try:
                self.budget.stage(2)  # One request plus the bounded transport retry.
                response = self.api.complete([
                    {'role':'system','content':'Plan a repository repair. Return JSON {"steps":[{"phase":"locate","description":"..."},'
                     '{"phase":"reproduce","description":"..."},{"phase":"repair","description":"..."}]}. '
                     'Use exactly these phases in order. Describe concrete evidence to seek. Do not claim any action is completed.'},
                    {'role':'user','content':self.query+'\nRepository map:\n'+mapping['output']}])
                rows = json_object(response.choices[0].message.content or '')['steps']
                if [r['phase'] for r in rows] != [p for p,_ in PHASES]:
                    raise ValueError('Invalid phase order')
                plan = [{'phase':r['phase'],'description':str(r['description'])[:1500],'status':'pending'} for r in rows]
            except (ValueError,KeyError,TypeError,ModelOutputError,PhaseEnded) as error:
                plan = default
                self.events.emit('plan_fallback',error=str(error))
            self.save(plan=plan)
        self.tools = self.make_tools()
        cwd = self.sandbox.description()['repository_mount']
        self.anchor = (f'Problem:\n{self.task.problem}\nRepository: {cwd}\nPublic test command: {self.task.test_command}\n'
                       f'Screenshot evidence (may contain recognition errors):\n{json.dumps(self.state["readings"],ensure_ascii=False)}\n'
                       f'Repository map:\n{mapping["output"]}')
        self.agent = Agent(llm=self.api,tools=self.tools,max_rounds=self.config.limits.max_steps,
                           max_context_tokens=self.config.max_context_tokens)
        self.agent.context = RepairContext(self.config.max_context_tokens,self.anchor,self.events)
        # A new conversation on resume observes the actual current patch; it never
        # blindly replays an interrupted mutation from serialized tool calls.
        self.agent.messages = [{'role':'user','content':self.anchor}]
        self.system = (f'You repair source code in {cwd}. Commands run in an isolated workspace; network is disabled. '
            'Use repo_map/repo_search to narrow the files, then read exact current source before editing. '
            'Tests, configuration paths marked read-only, and .git cannot be modified. '
            'Use existing public tests or create a separate reproduction script. '
            'During reproduction, write a standalone Python script asserting the required behavior and call '
            'register_reproduction with its relative path and expected assertion text. '
            'A script must fail with that assertion on original source and pass after the fix. '
            'An import error, unavailable dependency or timeout is not a reproduced bug. '
            'Public regressions are compared to original source; do not repair unrelated pre-existing failures. '
            'Reference evaluation tests and their results are unavailable during repair. '
            'The controller checks actual patches and tests. Your textual claims do not mark a task successful. '
            'After explaining a concrete hypothesis, implement the justified change; avoid repeated broad reading. '
            'You may use apply_patch with <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks. '
            'Conclude each requested phase with a concise account of observed facts and remaining uncertainty.')
        self.agent.context.bind_request(self.budget.counter,self.config.max_prompt_tokens,
                                        lambda:self.agent._full_messages()[:1],self.agent._tool_schemas)

    def run_phase(self,name,description,calls):
        self.phase(name)
        self.budget.stage(calls)
        selected = self.tools
        if name == 'locate':
            selected = [t for t in self.tools if t.name in {'repo_map','repo_search','read_file','grep','glob',
                                                          'read_screenshot','crop_screenshot'}]
        self.agent.tools = selected
        self.agent._tool_by_name = {t.name:t for t in selected}
        gap = ''
        if name in {'repair','retry'} and not self.state.get('reproduction',{}).get('baseline_reproduced'):
            gap = '\nIssue reproduction is unverified. Establish and register an assertion reproducer before claiming a fix.'
        self.agent._system = self.system+f'\nCurrent phase: {name}.\n{description}'+gap
        current = export_patch(self.baseline,self.workspace,self.directory/'current.patch',self.task.protected_paths,**self.tree)
        prompt = (f'Execute phase {name}: {description}\nCurrent changed files: {current["changed_files"]}\n'
                  f'Current diff excerpt:\n{(self.directory/"current.patch").read_text(errors="replace")[:10000]}')
        answer, reason = '', 'returned'
        try:
            answer = self.agent.chat(prompt)
        except PhaseEnded:
            reason = 'phase_budget'
        except (NoProgress,ModelOutputError) as error:
            reason = str(error)
        finally:
            atomic_json(self.directory/'conversation.json',self.agent.messages)
        self.events.emit('phase_finished',stage=name,reason=reason,answer=answer[:6000])
        self.save(last_answer=answer[:6000])
        self.refresh_index()
        return reason

    def public_check(self,label):
        self.phase('public_check')
        return self.feedback.check(label)

    @staticmethod
    def check_summary(check):
        reproduction = check['reproduction']
        candidate = reproduction.get('candidate',{})
        regression = dict(check['regression'])
        for key in ('lost_passing_tests','additional_failures','preexisting_failures'):
            if len(regression.get(key,[])) > 20:
                regression[key+'_total'] = len(regression[key])
                regression[key] = regression[key][:20]
                regression['case_list_truncated'] = True
        return {'public_command':check['command'],'regression':regression,
                'reproduction':{k:v for k,v in reproduction.items() if k!='candidate'},
                'reproduction_output':candidate.get('output','')[-5000:],
                'patch_replay':check['patch_replay'],'changed_files':check['changed_files'],
                'public_output':check['output'][-5000:],'passed':check['passed']}

    def failure_feedback(self,check):
        self.phase('review')
        self.budget.stage(2)
        observed = self.check_summary(check)
        try:
            response = self.api.complete([
                {'role':'system','content':'Diagnose a failed or empty code repair using the actual diff and test output. '
                 'Return JSON {"issues":["..."],"next_action":"..."}. '
                 'Do not invent executed actions or test results. Fix additional regression failures and the original issue. '
                 'Do not treat preexisting_failures or inconclusive environment checks as new regressions. '
                 'If reproduction is missing/unverified, create an assertion script and call register_reproduction. '
                 'No-new-regression alone does not establish that the issue is fixed.'},
                {'role':'user','content':self.task.problem+'\nObserved:\n'+json.dumps(observed)+
                 '\nDiff:\n'+(self.directory/'current.patch').read_text(errors='replace')[:12000]}])
            review = json_object(response.choices[0].message.content or '')
            if not isinstance(review.get('issues'),list) or not isinstance(review.get('next_action'),str):
                raise ValueError('Invalid review')
        except (ValueError,KeyError,TypeError,ModelOutputError,PhaseEnded):
            review = {'issues':['Public tests failed or no patch was produced.'],
                      'next_action':'Inspect the recorded failure and current source, make a justified fix, then test.'}
        review['observed'] = observed
        self.state['reviews'].append(review)
        self.save()
        return json.dumps(review,ensure_ascii=False)

    def repair(self):
        self.feedback.baseline()
        self.initialize_agent()
        for row in self.state['plan']:
            name = row['phase']
            if name in self.state.get('attempted_phases',[]):
                continue
            reason = self.run_phase(name,row['description'],getattr(self.config.workflow,name+'_calls'))
            status = 'returned' if reason == 'returned' else 'incomplete'
            if name == 'reproduce':
                status = 'verified' if self.state.get('reproduction',{}).get('baseline_reproduced') else 'incomplete'
            row.update(status=status,exit_reason=reason)
            self.state.setdefault('attempted_phases',[]).append(name)
            if status in {'returned','verified'}:
                self.state['completed_phases'].append(name)
            self.save()
        check = self.public_check('初次修复后的公共测试')
        while check['retry_needed'] and self.state['retry_count'] < self.config.workflow.max_retries:
            feedback = self.failure_feedback(check)
            self.save(retry_count=self.state['retry_count']+1)
            self.run_phase('retry','Use this observed feedback to repair remaining issues:\n'+feedback,
                           self.config.workflow.retry_calls)
            check = self.public_check(f'第{self.state["retry_count"]}次纠错后的公共测试')
        self.save(public_test_passed=check['passed'],public_patch_present=bool(check['patch_bytes']))

    def finalize(self):
        # This flag is saved BEFORE reference tests are installed. An interrupted
        # grader can resume verification, but can never return feedback to the Agent.
        self.save(grading_started=True)
        self.phase('grading')
        patch = self.directory/'patch.diff'
        exported = export_patch(self.baseline,self.workspace,patch,self.task.protected_paths,**self.tree)
        target = self.directory/f'verification_{time.time_ns()}'
        replay = replay_patch(self.baseline,target,patch,**self.tree)
        resolved, final = False, {'patch_replay':replay}
        if replay['applied']:
            if self.task.benchmark:
                final['hidden'] = self.grader(self.task,target,self.events,self.directory/'final_test.log',
                                             timeout=self.config.limits.command_timeout_seconds)
                resolved = bool(final['hidden']['resolved'])
            else:
                sb = create_sandbox(target,self.events,self.task.protected_paths,self.config.limits.command_timeout_seconds,
                                    self.baseline,self.task.runtime)
                try:
                    final['public'] = sb.run(self.task.test_command,log_path=self.directory/'final_test.log')
                finally:
                    sb.cancel()
                resolved = final['public']['returncode']==0 and not final['public']['timed_out']
        result = {**exported,'validation':final,'resolved':resolved,'status':'verified' if resolved else 'unresolved'}
        if final.get('hidden',{}).get('runtime_failure'):
            result.update(status='error',runtime_failure=True,failure_stage='hidden_grading')
        return result


def run_workflow_task(task, config, directory, **kwargs):
    directory = Path(directory)
    resume = kwargs.get('resume',False)
    directory.mkdir(parents=True,exist_ok=resume)
    with (directory/'workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        flow = RepairWorkflow(task,config,directory,**kwargs)
        # Validate resume identity before overwriting any previous result.
        if resume:
            prior = json.loads((directory/'workflow.json').read_text())
            if prior['identity'] != digest({'task':task.model_dump(),'config':config.model_dump(),
                                           'reader':kwargs.get('reader_name','vlm')}):
                raise ValueError('Resume arguments differ from the saved task')
            if prior.get('source') != implementation():
                raise ValueError('Implementation changed; resume using the frozen source or use a new directory')
            if not prior.get('baseline_hash') or tree_hash(directory/'baseline') != prior['baseline_hash']:
                raise ValueError('Baseline is incomplete or changed; cannot resume this attempt')
            if (directory/'result.json').exists() and json.loads((directory/'result.json').read_text())['status'] != 'paused':
                raise ValueError('This attempt already ended; use a new directory for a new attempt')
        handlers = {}
        def cancel(signum,frame):
            if flow.sandbox:
                flow.sandbox.cancel()
            raise RunCancelled(f'Interrupted by signal {signum}')
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig] = signal.signal(sig,cancel)
        result = {'task_id':task.task_id,'method':METHOD,'framework':'corecoder_workflow',
                  'protocol':PROTOCOL if task.benchmark else 'user_public_tests_workflow_v1',
                  'resolved':False,'model_inference':kwargs.get('model_inference',True)}
        try:
            flow.setup()
            if not flow.state.get('grading_started'):
                try:
                    flow.repair()
                except BudgetExceeded as error:
                    result['exit_reason'] = 'context_limit' if isinstance(error,ContextWindowExceeded) else 'budget_exceeded'
                    flow.save(stop_reason=str(error))
                except Exception as error:
                    # Preserve and grade any useful patch even after a model error.
                    result.update(exit_reason='model_error',agent_error=str(error),runtime_failure=True)
            result.update(flow.finalize())
        except (RunCancelled,KeyboardInterrupt) as error:
            result.update(status='paused',error=str(error))
        except Exception as error:
            result.update(status='error',error=str(error),runtime_failure=True)
            (directory/'error.log').write_text(traceback.format_exc())
        finally:
            if flow.sandbox:
                flow.sandbox.cancel()
            for sig,handler in handlers.items():
                signal.signal(sig,handler)
            if flow.budget:
                flow.budget.save()
            result.update(started_at=flow.state.get('started_at'),finished_at=time.time(),
                usage=flow.budget.snapshot() if flow.budget else {},retry_count=flow.state.get('retry_count',0),
                model_inference=flow.model_inference,
                verification_scope='隐藏参考测试，仅在Agent结束后评分' if task.benchmark else
                                   '用户指定的公共测试在独立副本中执行；不代表隐藏测试或需求全部覆盖')
            atomic_json(directory/'result.json',result)
            flow.save(status=result['status'],stage='paused' if result['status']=='paused' else 'completed')
            render_run(directory)
            export_delivery(directory)
            render_run(directory)
        return result
