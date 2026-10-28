import importlib.util
from pathlib import Path
import pytest
from kb.benchmark.sources import write_json

spec=importlib.util.spec_from_file_location('eval_v4',Path(__file__).resolve().parents[1]/'scripts/eval_v4.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


def fixtures(monkeypatch,tmp_path,missing=False,critical_bad=False):
    (tmp_path/'calibration').mkdir();(tmp_path/'browser').mkdir()
    write_json(tmp_path/'calibration/calibration_report.json',{'passed':True})
    write_json(tmp_path/'browser/report.json',{'status':'passed'})
    def row(i,correct,family=None):
        return {'id':str(i),'family_id':family or str(i),'status':'answered','trace':[],
                'metrics':{'semantic_correctness':int(correct),'faithfulness':1,'semantic_citation_support':1,'judge_failed':False}}
    def get(root,name):
        if name.startswith('gap_'):
            enabled=name.endswith('_followup') and not name.endswith('_no_followup')
            correct=enabled or 'withheld' not in name
            rows=[row(i,correct if i<8 else True,'gap-'+str(i) if i<8 else 'control-'+str(i)) for i in range(12)]
            if 'withheld' in name:
                for r in rows[:8]:r['trace']=[{'kind':'stress_observation','intervention_effective':True}]
            return rows
        baseline=name=='v4_fixed-bm25'
        rows=[row(i,i<(80 if baseline else 85),'unknown-'+str(i) if i<20 else None) for i in range(100)]
        if missing and not baseline:rows[99]['metrics']['judge_failed']=True;rows[99]['metrics']['semantic_correctness']=None
        if critical_bad and not baseline:rows[1]['metrics']['semantic_correctness']=0
        return rows
    monkeypatch.setattr(queue,'scored',get)


def test_release_gate_selects_candidate_without_touching_test(tmp_path,monkeypatch):
    fixtures(monkeypatch,tmp_path)
    result=queue.gate_decision(tmp_path)
    assert result['candidate']=='agent-guarded-bm25' and result['test_released']
    assert not (tmp_path/'test').exists()


@pytest.mark.parametrize('reason',['missing','critical'])
def test_gate_rejects_judge_missingness_and_critical_errors(tmp_path,monkeypatch,reason):
    fixtures(monkeypatch,tmp_path,missing=reason=='missing',critical_bad=reason=='critical')
    result=queue.gate_decision(tmp_path)
    assert result['candidate'] is None and not result['test_released']


def test_stages_are_dev_only_and_old_predictions_are_read_only(tmp_path):
    plan=queue.stages(tmp_path/'out',tmp_path/'runtime')
    for name,command,target in plan:
        assert '--split' not in command or command[command.index('--split')+1]=='dev'
        assert 'test' not in command
        if name.startswith('grade_v3_'):
            assert 'grade' in command and '--review-dir' in command
            assert target.is_relative_to(tmp_path/'out')
    assert len([n for n,_,_ in plan if n.startswith('grade_v3_')])==6
