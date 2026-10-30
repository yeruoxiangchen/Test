"""Single-agent framework comparison with a shared hidden-test verifier."""
from __future__ import annotations

import hashlib
import json
import signal
import time
import traceback
from pathlib import Path

from corecoder import Agent
from openai import APIError

from .context import EvidenceRepairContext, RepairContext
from .control import BudgetExceeded, ContextWindowExceeded, Events, NoProgress, RunCancelled, atomic_json
from .harness import PROTOCOL, HarnessAPIModel, HarnessBudget, QwenTokenCounter, make_mini
from .hidden_eval import grade_hidden, validate_hidden_task
from .images import ImageStore
from .mini_delivery import MiniDelivery
from .mini_verified import VerifiedDelivery
from .models import ModelOutputError
from .progress import RepairState
from .public_display import PublicDisplay
from .repository import export_patch, export_session_patch, prepare_repository, replay_patch
from .runtime import create_sandbox
from .tools import build_tools
from .workflow_cases import development_case


def install_development_case(task, baseline, workspace, verified=False, repair=False, recovery=False):
    """Expose exactly the existing v4 developer hints, never reference patches."""
    if recovery:
        from .mini_cases import recovery_case
        case = recovery_case(task.task_id)
    elif repair:
        from .mini_cases import repair_case
        case = repair_case(task.task_id)
    elif verified:
        from .mini_cases import verified_case
        case = verified_case(task.task_id)
    else:
        case = development_case(task.task_id)
    if not case:
        return None
    relative = Path('.sca_scratch/development_acceptance.py')
    for root in (baseline, workspace):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(case['script'])
    return {**case, 'path': relative.as_posix(),
            'script_sha256': hashlib.sha256(case['script'].encode()).hexdigest()}


def run_harness_task(task, config, run_dir, *, counter=None, api_factory=HarnessAPIModel, grader=grade_hidden):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    events = Events(run_dir / "events.jsonl")
    result = {"task_id": task.task_id, "framework": config.framework, "protocol": PROTOCOL,
              "status": "running", "resolved": False, "started_at": time.time(),
              "model_inference": True, "validation": {}}
    sandbox = budget = agent = display = delivery = None
    stage = "configuration"
    old_handlers = {}

    def update():
        atomic_json(run_dir / "status.json", {**result, "stage": stage,
                                              "usage": budget.snapshot() if budget else {}})

    def cancel(signum, frame):
        if budget:
            budget.cancelled.set()
        if sandbox:
            sandbox.cancel()
        raise RunCancelled(f"Received signal {signum}")

    try:
        validate_hidden_task(task)
        if task.images:
            raise ValueError("This framework comparison uses original issue text; screenshot mode is a separate experiment")
        atomic_json(run_dir / "task.json", task.model_dump())
        atomic_json(run_dir / "config.json", config.model_dump())
        budget = HarnessBudget(config, events, counter or QwenTokenCounter(config.tokenizer_path))
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, cancel)
        stage = "snapshot"
        update()
        tree = {"max_files": task.runtime.max_repo_files, "max_bytes": task.runtime.max_repo_bytes,
                "allowed_symlinks": task.runtime.allowed_symlinks}
        baseline, workspace = prepare_repository(Path(task.repo), run_dir, **tree)
        case = install_development_case(task, baseline, workspace, config.mini_verified,
                                        config.mini_repair_control, config.mini_recovery) if config.development_casebook else None
        protected = [*task.protected_paths, *([case['path']] if case else [])]
        result.update(development_regression=config.development_casebook, prior_failure_feedback_used=bool(case))
        if case:
            atomic_json(run_dir / 'development_case.json', case)
            result['development_case_sha256'] = case['script_sha256']
        sandbox = create_sandbox(workspace, events, protected,
                                 config.limits.command_timeout_seconds, baseline, task.runtime)
        if config.mini_session:
            if config.framework != 'mini':
                raise ValueError('mini_session must be used with the mini framework')
            sandbox.enable_agent_session()
        if config.xvfb_executable and task.runtime.backend == 'oci' and (baseline / 'pyvista').is_dir():
            display = PublicDisplay(config.xvfb_executable, run_dir)
            sandbox.public_display = display
            result['public_display'] = {'display': display.name, 'cpu_only': True}
        stage = "sandbox_probe"
        update()
        result["validation"]["sandbox_probe"] = sandbox.probe()
        result["sandbox"] = sandbox.description()
        if case:
            original = sandbox.run('python3 ' + case['path'], timeout=min(60, budget.remaining_seconds()))
            atomic_json(run_dir / 'development_baseline.json', original)
        api = api_factory(config.code, budget, events, policy=config.agent)
        stage = "agent"
        update()
        cwd = sandbox.description()["repository_mount"]
        prompt = f"Problem:\n{task.problem}\n\nRepository: {cwd}\nPublic test command: {task.test_command}"
        if case:
            prompt += ('\nVisible development acceptance (same prior failed run feedback as v4):\n'
                       + case['note'] + '\nFrozen read-only script:\n' + case['script']
                       + '\nRun it with: python3 ' + case['path']
                       + '\nOriginal failure:\n' + original['output'][-2500:]
                       + '\nThis developer check is partial; also test the full issue and existing behavior.')
        if config.framework == "mini":
            if config.mini_verified:
                from .mini_recovery import RecoveryDelivery
                stage = 'public_regression_baseline'
                update()
                delivery_class = RecoveryDelivery if config.mini_recovery else VerifiedDelivery
                delivery = delivery_class(baseline, workspace, run_dir, protected, tree, sandbox, budget, case,
                                            runtime=task.runtime, problem=task.problem)
                delivery.initialize()
                prompt += ('\nPublic regression modules (compared to the original baseline): '
                           + ', '.join(delivery.modules))
                stage = 'agent'
                update()
            elif config.mini_session:
                delivery = MiniDelivery(baseline, workspace, run_dir, protected, tree, sandbox, budget, case)
            agent = make_mini(config, budget, events, sandbox, run_dir, api=api, submission_check=delivery)
        else:
            state = RepairState(events) if config.agent.mode == "v7" else None
            tools = build_tools(sandbox, ImageStore([], run_dir / "screenshots", events), None,
                                budget, events, task.test_command, agent_policy=config.agent, repair_state=state)
            tools = [t for t in tools if t.name not in {"read_screenshot", "crop_screenshot"}]
            for tool in tools:
                tool.description = tool.description.replace("/workspace", cwd)
            agent = Agent(llm=api, tools=tools, max_rounds=config.limits.max_steps,
                          max_context_tokens=config.max_context_tokens)
            agent._system = (
                f"You repair a repository in {cwd}. Each shell call starts in {cwd}. "
                "Read the relevant source, identify the root cause, make a minimal justified patch, "
                "and run existing repository tests or your own reproduction scripts. "
                "Reference evaluation tests and their scores are unavailable. Network access is disabled. "
                "Tests and .git are read-only. Preserve existing behavior. "
                "Read file output has actual newlines; line numbers are not source code. "
                "After a failed edit inspect the current source and adjust the edit. "
                "Finish with an accurate summary of changes and tests; the evaluator exports the workspace diff.")
            agent.context = (EvidenceRepairContext(config.max_context_tokens, prompt, events, state)
                             if state else RepairContext(config.max_context_tokens, prompt, events))
            agent.context.bind_request(budget.counter, config.max_prompt_tokens,
                                       lambda: agent._full_messages()[:1], agent._tool_schemas)
        try:
            if config.framework == "mini":
                answer = agent.run(prompt)
                result["framework_exit"] = answer
                status = answer.get("exit_status")
                result["exit_reason"] = {"Submitted": "submitted", "LimitsExceeded": "budget_exceeded",
                                         "TimeExceeded": "budget_exceeded"}.get(status, status or "unknown")
            else:
                answer = agent.chat(prompt)
                result["exit_reason"] = ("step_limit" if answer == "(reached maximum tool-call rounds)" else "submitted")
                result["agent_answer"] = answer
        except (BudgetExceeded, NoProgress) as error:
            reason = ('context_limit' if isinstance(error,ContextWindowExceeded) else
                      'budget_exceeded' if isinstance(error,BudgetExceeded) else 'no_progress')
            result.update(exit_reason=reason,
                          agent_error=str(error))
        except ModelOutputError as error:
            result.update(exit_reason=error.reason, agent_error=str(error))
        except APIError as error:
            result.update(exit_reason="model_error", agent_error_type=type(error).__name__, agent_error=str(error),
                          runtime_failure=True, failure_stage='model_transport')
        except Exception as error:
            result.update(exit_reason="agent_error", agent_error_type=type(error).__name__, agent_error=str(error),
                          runtime_failure=True, failure_stage='agent_execution')
        finally:
            atomic_json(run_dir / "conversation.json", agent.messages)
        stage = "patch_export"
        update()
        exporter = export_session_patch if config.mini_session else export_patch
        if config.mini_verified:
            result.update(delivery.select_final(run_dir/'patch.diff'))
        else:
            result.update(exporter(baseline, workspace, run_dir / "patch.diff", protected, **tree))
        if config.framework == 'mini':
            result['recoverable_tool_errors'] = agent.env.recoverable_errors
        if (run_dir / 'delivery_check.json').exists():
            result['validation']['delivery'] = json.loads((run_dir / 'delivery_check.json').read_text())
        stage = "independent_hidden_verification"
        update()
        validation = replay_patch(baseline, run_dir / "verification", run_dir / "patch.diff", **tree)
        result["validation"]["patch_replay"] = validation
        if validation["applied"]:
            scored = grader(task, run_dir / "verification", events, run_dir / "verification_test.log",
                            timeout=config.final_test_timeout_seconds or config.limits.command_timeout_seconds)
            result["validation"]["hidden"] = scored
            result["resolved"] = scored["resolved"]
        result["status"] = "verified" if result["resolved"] else "unresolved"
        if result["validation"].get("hidden", {}).get("runtime_failure"):
            result.update(status="error", runtime_failure=True, failure_stage="hidden_grading")
        if result.get("agent_error_type") in {"APITimeoutError", "APIConnectionError"}:
            result["runtime_failure"] = True
    except RunCancelled as error:
        result.update(status="cancelled", error=str(error), runtime_failure=True)
    except BudgetExceeded as error:
        result.update(status="unresolved", exit_reason="budget_exceeded", error=str(error))
    except Exception as error:
        result.update(status="error", failure_stage=stage, error=str(error), error_type=type(error).__name__,
                      runtime_failure=True)
        (run_dir / "error.log").write_text(traceback.format_exc())
    finally:
        if sandbox:
            sandbox.cancel()
        if display:
            display.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        result.update(finished_at=time.time(), usage=budget.snapshot() if budget else {})
        atomic_json(run_dir / "result.json", result)
        stage = "completed"
        update()
    return result
