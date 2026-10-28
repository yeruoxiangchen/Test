import copy
import json
from types import SimpleNamespace

import pytest

from kb.agent import Harness, TOOLS
from kb.bounded import available_tools, prepare, tool
from kb.config import Settings
from kb.context import ContextBudget
from test_application import upload
from test_harness import call


def memory():
    return {"history":[], "sources":{}, "candidates":[], "searches":[], "feedback":{},
            "turns":0, "tool_calls":0, "seen_chunks":[], "user":{}, "base_ids":[]}


def names(state):
    return {t['function']['name'] for t in available_tools(state,Settings())}


def test_policy_moves_from_search_to_read_and_reserves_finalization():
    state=memory()
    assert 'search' in names(state)
    state.update(last_action='search',searches=['query'],candidates=[{'chunk_id':1}])
    assert names(state)=={'read_chunk','finish'}
    state['last_action']='read_chunk'
    assert 'search' in names(state)
    state['turns']=4
    assert names(state)=={'finish'}
    state.update(turns=1,tool_calls=10)
    assert names(state)=={'finish'}


def test_duplicate_search_does_not_hit_retriever_or_duplicate_sources():
    state=memory();state['searches']=['a query']
    def forbidden(*args):
        raise AssertionError('duplicate query must not retrieve again')
    h=SimpleNamespace(tool=forbidden)
    before=copy.deepcopy(state)
    result,changed=tool(h,{'question':'q'},'search',{'query':' A  QUERY '},state)
    assert result['duplicate'] and not changed and before==state


def test_context_admission_rolls_back_large_evidence_and_keeps_question():
    cfg=Settings(context_tokens=2500,output_tokens=100,tokenize_url='')
    def original_tool(name,args,state):
        source={'id':1,'filename':'doc','text':'evidence '*1000}
        state['sources']['S1']=source
        return {'citation':'S1',**source}
    h=SimpleNamespace(settings=cfg,tool=original_tool,
        budget=ContextBudget(cfg,counter=lambda m,t:sum(len(x.get('content','')) for x in m)))
    state=memory();before=copy.deepcopy(state)
    with pytest.raises(ValueError,match='未加入此项资料'):
        tool(h,{'question':'current question'},'read_chunk',{'chunk_id':1},state)
    assert state==before
    messages,_=prepare(h,{'question':'current question'},state,[TOOLS[-1]])
    assert 'current question' in messages[-1]['content']


def tool_state(messages):
    return json.loads(messages[-1]['content'].split('<tool_state>\n')[1].split('\n</tool_state>')[0])


class BoundedModel:
    def __init__(self, repair=False, crash=False):
        self.repair=repair
        self.crash=crash
        self.requests=[]

    def complete(self,messages,tools):
        self.requests.append((messages,tools))
        state=tool_state(messages)
        allowed={t['function']['name'] for t in tools}
        if state['sources']:
            if self.crash:
                raise SystemExit('process death after checkpoint')
            if self.repair and 'finish' in allowed and len(allowed)>1:
                return call('finish',{'answer':'住宿标准 600 元','citations':['S1'],'status':'answered'})
            assert state['sources'][0]['text']
            return call('finish',{'answer':'住宿标准 600 元 [S1]','citations':['S1'],'status':'answered'})
        if state['candidates']:
            assert 'search' not in allowed
            return call('read_chunk',{'chunk_id':state['candidates'][0]['chunk_id']})
        return call('search',{'query':'住宿报销'})


def start(system,model):
    cfg,db,client,base,*_=system
    cfg.agent_policy='bounded_v2'
    upload(system)
    run_id=client.post('/api/chat',json={'question':'住宿报销标准','base_ids':[base]}).json()['run_id']
    db.execute("UPDATE runs SET mode='agent' WHERE id=?",(run_id,))
    Harness(db,cfg,model).execute(run_id)
    return run_id,client.get('/api/runs/'+run_id).json()


def test_new_harness_finishes_with_strict_citation_repair(system):
    model=BoundedModel(repair=True)
    _,result=start(system,model)
    assert result['status']=='answered' and len(model.requests)==4
    assert result['citations'][0]['label']=='S1'
    assert {t['function']['name'] for t in model.requests[-1][1]}=={'finish'}
    assert any(e['kind']=='tool' and 'error' in e['payload']['result'] for e in result['events'])
    # Canonical full evidence occurs once, not once per previous model turn.
    assert len(tool_state(model.requests[-1][0])['sources'])==1


def test_new_harness_checkpoint_resume_reuses_evidence_and_budget(system):
    with pytest.raises(SystemExit):
        start(system,BoundedModel(crash=True))
    cfg,db,client,*_=system
    saved=db.one("SELECT * FROM runs WHERE status='running'")
    state=json.loads(saved['state'])
    assert state['turns']==2 and len(state['sources'])==1
    db.execute("UPDATE runs SET status='queued' WHERE id=?",(saved['id'],))
    model=BoundedModel()
    Harness(db,cfg,model).execute(saved['id'])
    assert len(model.requests)==1
    row=db.one('SELECT * FROM runs WHERE id=?',(saved['id'],))
    assert row['status']=='answered' and json.loads(row['state'])['turns']==3


def test_new_harness_never_accepts_unread_citation_or_exceeds_budget(system):
    class Invalid:
        def complete(self,messages,tools):
            return call('finish',{'answer':'invented [S999]','citations':['S999'],'status':'answered'})
    _,result=start(system,Invalid())
    assert result['status']=='error' and result['answer'] is None
    assert sum(e['kind']=='model' for e in result['events'])==system[0].max_turns
    assert sum(e['kind']=='tool' for e in result['events'])<=system[0].max_tool_calls
