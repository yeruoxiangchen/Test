"""Task orchestration around an unmodified CoreCoder Agent loop."""

from __future__ import annotations

import json
import os
import shutil
import signal
import time
import traceback
from pathlib import Path

from corecoder import Agent

from .benchmark import official_grade
from .config import RunConfig, Task
from .context import EvidenceRepairContext, RepairContext
from .control import Budget, BudgetExceeded, Events, NoProgress, RunCancelled, atomic_json
from .images import ImageStore, OCRReader, OracleReader, VisionReader
from .models import APIModel, ModelOutputError, ReplayModel
from .progress import RepairState
from .repository import export_patch, prepare_repository, replay_patch
from .runtime import create_sandbox
from .tools import build_tools

SYSTEM = """You repair a repository using screenshot evidence and executable tests.
All repository tools run in an isolated /workspace. The screenshot reader is an external tool.
Treat screenshot text as fallible evidence: search repository filenames and source to confirm it.
Preserve existing behavior and fix the root cause. Tests and .git are protected read-only.
Use the available screenshot tools when evidence is ambiguous. Cropping does not itself read text.
Run the fixed run_tests tool to check changes and use failure output to continue repairing.
Shell cwd resets to /workspace each call; prefer read_file, grep, glob and unique-match edit_file.
Finish with a concise explanation of the modification and the tests actually run.
The independent verifier, not your final statement, determines whether the patch is successful."""


def run_task(task: Task, config: RunConfig, run_dir: Path, *, reader_name: str = "vlm",
             model_name: str = "api") -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    events = Events(run_dir / "events.jsonl")
    enhanced = config.agent.mode in {"v6", "v7"}
    repair_state = RepairState(events) if config.agent.mode == "v7" else None
    budget = Budget(config.limits, events, separate_summaries=enhanced)
    sandbox = None
    started = time.time()
    result = {"task_id": task.task_id, "split": task.split, "group_id": task.group_id or task.task_id,
              "provenance": task.provenance, "reader": reader_name, "code_backend": model_name,
              "model_inference": model_name == "api", "status": "running", "resolved": False,
              "started_at": started, "pid": os.getpid(), "validation": {}, "failure_stage": None}
    stage = "configuration"
    agent = None
    old_handlers = {}

    def status():
        observed = {"repair_state": repair_state.snapshot()} if repair_state is not None else {}
        atomic_json(run_dir / "status.json", {**result, "stage": stage, "usage": budget.snapshot(), **observed})

    def cancel(signum, frame):
        budget.cancelled.set()
        if sandbox is not None:
            sandbox.cancel()
        raise RunCancelled(f"Received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, cancel)
    try:
        atomic_json(run_dir / "task.json", task.model_dump())
        atomic_json(run_dir / "config.json", config.model_dump())
        status()
        if model_name == "api":
            config.code.require_live(config.limits.max_cost_usd)
        elif model_name != "replay" or not task.replay:
            raise ValueError("Use api, or replay with a task that supplies a recorded script")
        if reader_name == "vlm":
            config.vision.require_live(config.limits.max_cost_usd)
        stage = "snapshot"
        status()
        tree_limits = {"max_files": task.runtime.max_repo_files, "max_bytes": task.runtime.max_repo_bytes,
                       "allowed_symlinks": task.runtime.allowed_symlinks}
        baseline, workspace = prepare_repository(Path(task.repo), run_dir, **tree_limits)
        for name in task.protected_paths:
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
                raise ValueError("protected_paths must be repository-relative literal paths")
        sandbox = create_sandbox(workspace, events, task.protected_paths,
                                 config.limits.command_timeout_seconds, baseline, task.runtime)
        stage = "sandbox_probe"
        status()
        result["validation"]["sandbox_probe"] = sandbox.probe()
        result["sandbox"] = sandbox.description()
        stage = "baseline_test"
        status()
        # Even a failing test command may mutate repository files. Never execute it
        # against the private base used later for patch creation and grading.
        baseline_check = run_dir / "baseline_check"
        shutil.copytree(baseline, baseline_check, symlinks=True)
        base_sandbox = create_sandbox(baseline_check, events, task.protected_paths,
                                     config.limits.command_timeout_seconds, baseline, task.runtime)
        base_log = run_dir / "baseline_test.log"
        base_test = base_sandbox.run(task.test_command, timeout=budget.remaining_seconds(), log_path=base_log)
        result["validation"]["baseline"] = base_test
        baseline_passed = base_test["returncode"] == 0
        if task.benchmark:
            scored = official_grade(Path(task.benchmark), base_log)
            result["validation"]["baseline_official"] = scored
            if not scored["parsed"]:
                raise RuntimeError("Official baseline tests did not yield parseable results")
            baseline_passed = scored["resolved"]
        if base_test["timed_out"]:
            raise RuntimeError("Baseline test timed out; this task is not a validated repair instance")
        if baseline_passed:
            result["status"] = "baseline_passed"
            result["failure_stage"] = stage
            result["error"] = "Baseline tests already pass; provide a command reproducing the failure"
            return result
        stage = "reading"
        status()
        store = ImageStore(task.images, run_dir / "screenshots", events)
        if reader_name == "oracle":
            reader = OracleReader(store)
        elif reader_name == "ocr":
            reader = OCRReader(store)
        elif reader_name == "vlm":
            reader = VisionReader(store, APIModel(config.vision, budget, events, kind="vision"))
        else:
            raise ValueError(f"Unknown reader: {reader_name}")
        test_count = 0

        def run_official_tests():
            nonlocal test_count
            test_count += 1
            log = run_dir / f"agent_test_{test_count}.log"
            test = sandbox.run(task.test_command, timeout=budget.remaining_seconds(), log_path=log)
            score = official_grade(Path(task.benchmark), log)
            feedback = {name: {"passed": len(v.get("success", [])), "failed": v.get("failure", [])}
                        for name, v in score.get("tests_status", {}).items()}
            return {**test, "official_tests": feedback, "resolved": score["resolved"]}

        tools = build_tools(sandbox, store, reader, budget, events, task.test_command,
                            test_runner=run_official_tests if task.benchmark else None,
                            screenshot_policy=config.screenshots, agent_policy=config.agent,
                            repair_state=repair_state)
        first_readings = []
        if task.images:
            first_readings.append(json.loads(next(t for t in tools if t.name == "read_screenshot").execute(
                image_id=task.images[0].image_id)))
        if not config.screenshots.allow_reread:
            tools = [t for t in tools if t.name not in {"read_screenshot","crop_screenshot"}]
        model = APIModel(config.code, budget, events, policy=config.agent) if model_name == "api" else ReplayModel(
            Path(task.replay), budget, events)
        agent = Agent(llm=model, tools=tools, max_rounds=config.limits.max_steps,
                      max_context_tokens=config.max_context_tokens)
        agent._system = SYSTEM
        prompt = (f"Problem:\n{task.problem}\n\nRepository: /workspace\nFixed tests: {task.test_command}\n"
                  f"Screenshots:\n{json.dumps(store.public(), ensure_ascii=False)}\n"
                  f"Initial screenshot evidence:\n{json.dumps(first_readings, ensure_ascii=False)}")
        if enhanced:
            agent.context = (EvidenceRepairContext(config.max_context_tokens, prompt, events, repair_state)
                             if repair_state is not None else RepairContext(config.max_context_tokens, prompt, events))
            agent._system += ("\nRead the relevant source before editing. Read small ranges and use grep/glob to locate symbols. "
                              "File output contains actual newlines; line-number prefixes are not part of the source. "
                              "After an edit_match_failed response, inspect the actual context and change the edit. "
                              "Never repeat an identical failed edit on unchanged contents. "
                              "Check python_syntax feedback and run_tests after edits; do not replace repair actions with explanations.")
        if repair_state is not None:
            actual_cwd = sandbox.description().get("repository_mount", "/workspace")
            agent._system = agent._system.replace("Shell cwd resets to /workspace", f"Shell cwd resets to {actual_cwd}")
            agent._system += (
                "\nObserved repair state is produced by tool feedback. Keep plans and hypotheses distinct from executed edits. "
                "After locating the relevant code, either implement a justified minimal patch or identify the exact missing evidence. "
                "Avoid repeatedly reading unchanged ranges without a new question. A failing import in protected tests can request "
                "a missing production API; repair production code rather than modifying protected tests. "
                "If the specified tests pass on the current changes, finish with an accurate summary rather than unrelated edits.")
        stage = "agent"
        status()
        try:
            answer = agent.chat(prompt)
            result["agent_answer"] = answer
            result["exit_reason"] = "step_limit" if answer == "(reached maximum tool-call rounds)" else "submitted"
        except BudgetExceeded as error:
            result["exit_reason"] = "budget_exceeded"
            result["agent_error"] = str(error)
        except NoProgress as error:
            result.update(exit_reason="no_progress", agent_error=str(error))
        except ModelOutputError as error:
            result.update(exit_reason=error.reason, agent_error=str(error))
        except Exception as error:
            # API/context errors may occur AFTER a useful edit. Always export and
            # independently verify that edit, retaining the failure as evidence.
            result["exit_reason"] = "model_error"
            result["agent_error"] = str(error)
            result["agent_error_type"] = type(error).__name__
            (run_dir / "agent_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            atomic_json(run_dir / "conversation.json", agent.messages)
        stage = "patch_export"
        status()
        result.update(export_patch(baseline, workspace, run_dir / "patch.diff", task.protected_paths, **tree_limits))
        stage = "independent_verification"
        status()
        validation = replay_patch(baseline, run_dir / "verification", run_dir / "patch.diff", **tree_limits)
        result["validation"]["patch_replay"] = validation
        if validation["applied"]:
            verifier = create_sandbox(run_dir / "verification", events, task.protected_paths,
                                      config.limits.command_timeout_seconds, baseline, task.runtime)
            final_log = run_dir / "verification_test.log"
            test = verifier.run(task.test_command, log_path=final_log)
            result["validation"]["final"] = test
            result["resolved"] = test["returncode"] == 0 and not test["timed_out"]
            if task.benchmark:
                scored = official_grade(Path(task.benchmark), final_log)
                result["validation"]["final_official"] = scored
                result["resolved"] = scored["resolved"] and not test["timed_out"]
        result["status"] = "verified" if result["resolved"] else "unresolved"
        if enhanced and result.get("exit_reason") == "submitted" and not result["resolved"]:
            result["exit_reason"] = "submitted_unresolved" if result.get("changed_files") else "no_patch_submitted"
        if not result["resolved"]:
            result["failure_stage"] = "repair_or_verification"
        if model_name == "replay":
            result["interpretation"] = "Scripted plumbing check using the reference patch; not model repair accuracy"
    except RunCancelled as error:
        result.update(status="cancelled", failure_stage=stage, error=str(error))
    except BudgetExceeded as error:
        result.update(status="budget_exceeded", failure_stage=stage, error=str(error))
    except Exception as error:
        result.update(status="error", failure_stage=stage, error=str(error), error_type=type(error).__name__)
        (run_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if sandbox is not None:
            sandbox.cancel()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        result["usage"] = budget.snapshot()
        if repair_state is not None:
            result["repair_state"] = repair_state.snapshot()
        result["finished_at"] = time.time()
        result["elapsed_seconds"] = round(result["finished_at"]-started, 3)
        atomic_json(run_dir / "result.json", result)
        status()
        events.emit("run_finished", result=result)
        if agent is not None and not (run_dir / "conversation.json").exists():
            atomic_json(run_dir / "conversation.json", agent.messages)
    return result
