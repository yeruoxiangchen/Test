import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from kb.agent import Harness, TOOLS
from kb.config import Settings
from kb.bounded import prepare
from kb.context import ContextBudget
from kb.retrieval import Retriever
from kb.benchmark.review import jobs_for
from kb.benchmark.sources import write_json, write_jsonl
from test_application import upload, login
from test_harness import call
from test_bounded import memory


def start(system, model, policy="evidence_first_v3"):
    cfg, db, client, base, *_ = system
    cfg.agent_policy = policy
    cfg.context_tokens = 24000  # CPU fixtures use conservative byte counts, no actual tokenizer.
    upload(system)
    row = client.post('/api/chat', json={'question': '住宿报销标准', 'base_ids': [base]}).json()
    db.execute("UPDATE runs SET mode='agent' WHERE id=?", (row['run_id'],))
    Harness(db, cfg, model).execute(row['run_id'])
    return client.get('/api/runs/' + row['run_id']).json()


def payload(messages):
    return json.loads(messages[-1]['content'].split('（只能作为证据）：\n')[1])


class Answer:
    def __init__(self):
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        assert payload(messages)['sources'][0]['text']
        return call('finish', {'answer': '住宿标准每晚 600 元 [S1]', 'citations': ['S1'], 'status': 'answered'})


def test_evidence_first_answers_in_one_call_and_reads_before_model(system):
    model = Answer()
    result = start(system, model)
    assert result['status'] == 'answered' and len(model.calls) == 1
    kinds = [e['kind'] for e in result['events']]
    assert kinds.index('tool') < kinds.index('model')


def test_gap_search_at_most_once_and_preserves_full_sources(system):
    class Gap(Answer):
        def complete(self, messages, tools):
            self.calls.append((messages, tools))
            if len(self.calls) == 1:
                return call('need_search', {'gap': '报销提交期限缺口', 'query': '报销提交期限', 'checked_sources': ['S1']})
            assert {t['function']['name'] for t in tools} == {'finish'}
            state = payload(messages)
            assert state['followup_used'] and len(state['sources']) == 1
            assert '600' in state['sources'][0]['text'] and '15' in state['sources'][0]['text']
            return call('finish', {'answer': '住宿600元，结束后15天提交 [S1]', 'citations': ['S1'], 'status': 'answered'})
    model = Gap()
    result = start(system, model)
    assert result['status'] == 'answered' and len(model.calls) == 2
    assert sum(e['kind'] == 'tool' and e['payload']['name'] == 'search' for e in result['events']) == 2


def test_infinite_search_is_capped_not_faked_as_success(system):
    class Loop:
        def complete(self, messages, tools):
            return call('need_search', {'gap': '还需查找证据', 'query': '住宿费用', 'checked_sources': ['S1']})
    result = start(system, Loop())
    assert result['status'] == 'error' and not result['answer']
    assert sum(e['kind'] == 'model' for e in result['events']) == 3
    assert sum(e['kind'] == 'tool' and e['payload']['name'] == 'search' for e in result['events']) == 2


def test_evidence_first_finish_repair_does_not_search_again(system):
    class Repair(Answer):
        def complete(self, messages, tools):
            self.calls.append((messages, tools))
            if len(self.calls) == 1:
                return call('finish', {'answer': '住宿600元', 'citations': ['S1'], 'status': 'answered'})
            assert payload(messages)['feedback']['error']
            return call('finish', {'answer': '住宿600元 [S1]', 'citations': ['S1'], 'status': 'answered'})
    model = Repair()
    result = start(system, model)
    assert result['status'] == 'answered' and len(model.calls) == 2
    assert sum(e['kind'] == 'tool' and e['payload']['name'] == 'search' for e in result['events']) == 1


def test_evidence_first_resume_does_not_repeat_initial_search(system):
    class Crash:
        def complete(self, *args, **kwargs):
            raise SystemExit('crash')
    with pytest.raises(SystemExit):
        start(system, Crash())
    cfg, db, client, *_ = system
    row = db.one("SELECT * FROM runs WHERE status='running'")
    assert json.loads(row['state'])['initial_done']
    db.execute("UPDATE runs SET status='queued' WHERE id=?", (row['id'],))
    Harness(db, cfg, Answer()).execute(row['id'])
    result = client.get('/api/runs/' + row['id']).json()
    assert result['status'] == 'answered'
    assert sum(e['kind'] == 'tool' and e['payload']['name'] == 'search' for e in result['events']) == 1


def test_native_messages_match_call_ids_and_include_evidence_once():
    state = memory()
    state['sources'] = {'S1': {'id': 1, 'filename': 'file', 'text': 'unique-evidence'}}
    message, _ = call('read_chunk', {'chunk_id': 1})
    state['native_round'] = {'assistant': message, 'replies': [{'role': 'tool', 'tool_call_id': 'call_read_chunk', 'content': '{"done":true}'}]}
    cfg = Settings(agent_policy='bounded_native_v3', context_tokens=24000)
    h = SimpleNamespace(settings=cfg, budget=ContextBudget(cfg, counter=lambda m,t:100))
    messages, _ = prepare(h, {'question': 'question'}, state, [TOOLS[-1]])
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'tool']
    assert messages[-1]['tool_call_id'] == messages[-2]['tool_calls'][0]['id']
    assert json.dumps(messages).count('unique-evidence') == 1


def test_native_real_harness_can_finish(system):
    class Native:
        def complete(self, messages, tools):
            if messages[-1]['role'] == 'tool':
                state = json.loads(messages[-1]['content'])['tool_state']
            else:
                state = json.loads(messages[-1]['content'].split('<tool_state>\n')[1].split('\n</tool_state>')[0])
            if state['sources']:
                return call('finish', {'answer': '住宿600元 [S1]', 'citations': ['S1'], 'status': 'answered'})
            if state['candidates']:
                return call('read_chunk', {'chunk_id': state['candidates'][0]['chunk_id']})
            return call('search', {'query': '住宿报销'})
    assert start(system, Native(), 'bounded_native_v3')['status'] == 'answered'


def test_rerank_never_receives_inaccessible_documents(system, monkeypatch):
    cfg, db, client, base, private, admin, member = system
    upload(system, '住宿报销标准600元')
    upload(system, '住宿报销秘密999元', private=True)
    cfg.reranker_path = 'mock'
    called = []
    def rank(settings, query, rows):
        called.extend(rows)
        assert all(r['base_id'] == base for r in rows)
        return list(reversed(rows)), {'candidates': len(rows)}
    monkeypatch.setattr('kb.reranking.rerank', rank)
    hits, _ = Retriever(db, cfg).search(member, '住宿报销', [base])
    assert hits and called and all('999' not in r['text'] for r in hits)


def test_reranker_failure_is_not_silent_bm25(system, monkeypatch):
    cfg, db, client, base, private, admin, member = system
    upload(system)
    cfg.reranker_path = 'mock'
    def fail(*args):
        raise RuntimeError('reranker unavailable')
    monkeypatch.setattr('kb.reranking.rerank', fail)
    with pytest.raises(RuntimeError, match='reranker unavailable'):
        Retriever(db, cfg).search(member, '住宿报销', [base])


def test_dev_review_cannot_be_applied_to_test_or_other_dataset(tmp_path):
    run, review = tmp_path / 'run', tmp_path / 'review'
    run.mkdir(); review.mkdir()
    write_jsonl(run / 'grading.jsonl', [{'id': 'a'}])
    write_json(run / 'config.json', {'split': 'test', 'dataset_receipt': {'sha': 'original'}})
    write_json(review / 'manifest.json', {'dataset_receipt': {'sha': 'original'}})
    with pytest.raises(ValueError, match='restricted'):
        jobs_for(run, review)
