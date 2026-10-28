import json
from types import SimpleNamespace
import pytest
from kb.agent import Harness
from kb.guarded import assess, clarification, render_claims, spans
from kb.benchmark import semantic
from kb.benchmark.gap_stress import prepare, FirstSearchFilter
from kb.benchmark.sources import read_jsonl
from test_harness import call
from test_v3 import start


def memory():
    return {'sources':{'S1':{'filename':'规则','text':'住宿标准600元\n\n报销期限15天','id':1}}}


def assessment(state='covered',query='',evidence=None):
    return {'intent':'确认住宿标准','requirements':[{'need':'住宿标准','state':state,'evidence':evidence or ['S1:1']}],
            'clarifying_questions':[], 'query':query}


def test_rendering_uses_host_evidence_and_rejects_fabrication():
    state=memory()
    assert spans(state)['S1:2']['text']=='报销期限15天'
    answer,refs=render_claims({'claims':[{'text':'标准600元','evidence':['S1:1']}]},state)
    assert answer=='标准600元 [S1]' and refs==['S1']
    for bad in [[],{'claims':[{'text':'标准600元','evidence':['S99:1']}]},
                {'claims':[{'text':'标准600元 [S9]','evidence':['S1:1']}]}]:
        with pytest.raises(ValueError):render_claims(bad,state)


def test_clarify_cannot_smuggle_unconfirmed_actions():
    for questions in [['请先尝试删除本地标签，然后告诉我是否成功？'], ['运行 `git tag -d a` 后还有吗？'],['没有问题']]:
        with pytest.raises(ValueError):clarification(questions)
    assert '目标环境' in clarification(['目标环境是测试还是生产？'])
    args=assessment('needs_user');args['clarifying_questions']=['目标名称是什么？']
    assert assess(args,memory())[0]=='clarify'
    assert assess(assessment('missing',query=''),memory())[0]=='insufficient'


class Grounded:
    def __init__(self):self.calls=0
    def complete(self,messages,tools):
        self.calls+=1
        if tools[0]['function']['name']=='assess_evidence':return call('assess_evidence',assessment())
        return call('submit_grounded',{'claims':[{'text':'住宿标准600元','evidence':['S1:1']}]})


def test_guarded_real_api_harness_cites_read_sources(system):
    model=Grounded();result=start(system,model,'guarded_v4')
    assert result['status']=='answered' and model.calls==2
    assert result['citations'][0]['label']=='S1'
    assert any(e['kind']=='grounding' for e in result['events'])


def test_invalid_span_feedback_uses_real_tool_call_id(system):
    class Repair(Grounded):
        def complete(self,messages,tools):
            if self.calls==0:
                self.calls+=1;return call('assess_evidence',assessment(evidence=['S999:1']))
            if self.calls==1:
                assert messages[-1]['role']=='tool'
                assert messages[-1]['tool_call_id']==messages[-2]['tool_calls'][0]['id']
            return super().complete(messages,tools)
    model=Repair();result=start(system,model,'guarded_v4')
    assert result['status']=='answered' and model.calls==3


@pytest.mark.parametrize('enabled',[True,False])
def test_followup_control_and_hard_bound(system,enabled):
    system[0].guarded_followup=enabled
    class Gap(Grounded):
        def complete(self,messages,tools):
            self.calls+=1
            return call('assess_evidence',assessment('missing',query='期限'+str(self.calls)))
    result=start(system,Gap(),'guarded_v4')
    assert result['status']=='insufficient'
    searches=[e for e in result['events'] if e['kind']=='tool' and e['payload']['name']=='search']
    assert len(searches)==(2 if enabled else 1)


def test_guarded_cancel_and_checkpoint_resume(system):
    class Crash(Grounded):
        def complete(self,*args,**kwargs):raise SystemExit('crash')
    with pytest.raises(SystemExit):start(system,Crash(),'guarded_v4')
    cfg,db,client,*_=system
    row=db.one("SELECT * FROM runs WHERE status='running'")
    assert json.loads(row['state'])['initial_done']
    db.execute("UPDATE runs SET status='queued' WHERE id=?",(row['id'],))
    Harness(db,cfg,Grounded()).execute(row['id'])
    result=client.get('/api/runs/'+row['id']).json()
    assert result['status']=='answered'
    assert sum(e['kind']=='tool' and e['payload']['name']=='search' for e in result['events'])==1


def test_synthetic_gap_set_and_first_search_only(tmp_path):
    prepare(tmp_path)
    cases=read_jsonl(tmp_path/'dev.jsonl')
    assert len(cases)==48 and len({c['family_id'] for c in cases})==12
    assert read_jsonl(tmp_path/'test.jsonl')==[]
    retriever=SimpleNamespace(search=lambda *a,**kw:([{'id':1},{'id':2}],[]))
    filtered=FirstSearchFilter(retriever,[2])
    assert filtered.search()[0]==[{'id':1}]
    assert filtered.search()[0]==[{'id':1},{'id':2}]


def test_judge_quote_is_from_candidate_not_gold():
    job={'citations':[{'label':'S1','text':'资料支持600元'}], 'gold_evidence':[{'text':'900元'}]}
    grade={'correctness':1,'faithfulness':1,'citation_support':1,'rationale':'理由','claims':[
        {'claim':'600元','supported':1,'citation_labels':['S1'],'evidence_quote':'600元','reason':'依据'}]}
    semantic.validate(grade,job)
    grade['claims'][0]['evidence_quote']='900元'
    with pytest.raises(ValueError):semantic.validate(grade,job)
    with pytest.raises(ValueError):semantic.validate([],job)
    resolved=semantic.resolve_grade({'claims':[{'claim':'600元','supported':1,'evidence_ids':['S1:1']}]},job)
    assert resolved['claims'][0]['evidence_quote']=='资料支持600元'
    with pytest.raises(ValueError):semantic.resolve_grade({'claims':[{'supported':1,'evidence_ids':['GOLD:1']}]},job)


def test_safe_refusal_is_not_correct_on_answerable_question():
    grade={'correctness':1,'faithfulness':1,'citation_support':1,'claims':[],'rationale':'没有编造'}
    job={'expected_action':'answered','prediction_status':'insufficient'}
    with pytest.raises(ValueError,match='未完成题目要求'):semantic.validate(grade,job)
    grade['correctness']=0
    semantic.validate(grade,job)


def test_judge_failure_stays_missing_and_payload_is_blinded(tmp_path,monkeypatch):
    import httpx
    requests=[]
    def handler(request):
        body=json.loads(request.content);requests.append(body)
        if request.url.path=='/tokenize':return httpx.Response(200,json={'count':100,'max_model_len':12288})
        return httpx.Response(200,json={'choices':[{'message':{'content':'not JSON'}}]})
    real=httpx.Client
    monkeypatch.setattr(semantic.httpx,'Client',lambda **kw:real(transport=httpx.MockTransport(handler)))
    judge=object.__new__(semantic.Judge)
    judge.url='http://mock/v1';judge.tokenizer='http://mock/tokenize';judge.headers={}
    judge.model='judge';judge.output_tokens=2200;judge.context=12288
    grade=judge.grade({'id':'secret_case','arm':'secret_arm','expected_correctness':0,'answer':'答'},tmp_path)
    assert grade['judgment_status']=='error' and grade['correctness'] is None
    assert all('secret_arm' not in json.dumps(r) and 'expected_correctness' not in json.dumps(r) and 'secret_case' not in json.dumps(r) for r in requests)


def test_model_grading_keeps_original_error_and_all_ids(tmp_path,monkeypatch):
    class FakeJudge:
        identity={'fake':True};model='fake';url='fake';output_tokens=1
        def __init__(self,root):pass
        def grade(self,job,output):return {'judgment_status':'error','correctness':None,'faithfulness':None,'citation_support':None}
    monkeypatch.setattr(semantic,'Judge',FakeJudge)
    rows=semantic.evaluate_jobs([{'id':'a','prediction_status':'error'},{'id':'b','prediction_status':'answered'}],tmp_path/'g',tmp_path,{})
    assert len(rows)==2 and rows[0]['correctness']==0 and rows[0]['origin']=='execution_failure_rule'
    assert rows[1]['correctness'] is None
    assert json.loads((tmp_path/'g/status.json').read_text())['judge_errors']==1
