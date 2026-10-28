"""Synthetic cross-document dev probes; missing-first-evidence is a diagnostic intervention."""
from pathlib import Path
from .sources import write_json,write_jsonl,digest
from .dataset import build,freeze,verify_freeze


def prepare(root):
    root=Path(root)
    if (root/'freeze.json').exists():return verify_freeze(root)
    (root/'corpus').mkdir(parents=True,exist_ok=True)
    docs,catalog,hidden=[],[],{}
    def doc(name,text):
        (root/'corpus'/f'{name}.txt').write_text(text)
        docs.append({'id':name,'title':name,'path':f'corpus/{name}.txt','sha256':digest(text.encode()),'visibility':'all','question_split':'dev','kind':'synthetic_fixture'})
        return {'document_id':name,'quote':text}
    def case(name,q,ref,facts,evidence,action='answered'):
        catalog.append({'id':name,'split':'dev','questions':[q,'请按演练资料说明：'+q,'支持工单：'+q,'请按演练资料说明：'+q],
          'context':'我们只讨论文档中的合成演练租户，不是真实企业部署。','reference_answer':ref,'required_facts':facts,'evidence':evidence,
          'expected_action':action,'category':'multi_evidence' if len(evidence)>1 else 'single_evidence' if evidence else action,
          'source_kind':'synthetic_gap_v1'})
    for i,(tenant,role) in enumerate(zip(['青禾','星河','松岚','海川','云杉','晨曦','远山','清泉'],['部门主管','采购负责人','财务专员','项目经理','运营主管','合规负责人','技术主管','值班经理'])):
        code='PX-'+str(31+i);amount=1300+i*700
        a=doc(f'tenant_{i}',f'合成演练租户{tenant}的采购申请由{role}批准；二次复核金额规则见制度{code}，本页不列金额。')
        b=doc(f'policy_{i}',f'合成演练租户{tenant}的采购金额复核制度{code}规定：采购金额超过{amount}元时需要二次复核，未超过时不需要。')
        name=f'gap-{i:02d}'
        case(name,f'演练租户{tenant}的采购申请由谁批准，金额超过多少元需要二次复核？',f'{role}批准，超过{amount}元需要二次复核。',[[role],[str(amount)],['元']],[a,b])
        for v in range(1,5):hidden[f'{name}-{v}']=b['document_id']
    e=doc('helpdesk','合成演练服务台规定：普通工单应在7天内归档。')
    case('control-archive','演练服务台普通工单几天内归档？','7天内归档。',[['7'],['天']],[e])
    e=doc('support','合成演练文档指定的测试联系邮箱为 support42@example.invalid，仅用于测试。')
    case('control-contact','演练文档里的测试联系邮箱是什么？','support42@example.invalid',[['support42@example.invalid']],[e])
    case('control-clarify','我要删除那个配置，目标名称和环境都没提供，请先问我缺哪些信息。','询问配置名称及目标环境，不先提供操作命令。',[],[],'clarify')
    case('control-unknown','这些演练租户真正的生产服务器密码是多少？','合成资料不能证明实际生产密码，应拒答。',[],[],'insufficient')
    write_jsonl(root/'documents.jsonl',docs);write_jsonl(root/'catalog.jsonl',catalog)
    write_json(root/'source_manifest.json',{'kind':'synthetic_gap_v1','design_revision':2,'human_verified':False,'not_business_data':True,
       'revision_reason':'Include tenant on both pages so natural retrieval can contain both; measure actual withholding instead of assuming intervention removed evidence.'})
    write_json(root/'authoring_spec.json',{'kind':'synthetic_gap_v1','withhold_on_first_search':hidden,
       'plan':'12 direct family representatives selected with --limit 12; natural and withheld-first-search conditions; no hidden ids or gold shown to agent'})
    build(root)
    return freeze(root)


class FirstSearchFilter:
    """Wrap only retrieval; subsequent searches and ACL-aware reads are unchanged."""
    def __init__(self,retriever,hidden_ids):
        self.retriever=retriever;self.hidden=set(hidden_ids);self.used=False
        self.removed_count=0
    def __getattr__(self,name):return getattr(self.retriever,name)
    def search(self,*args,**kwargs):
        rows,warnings=self.retriever.search(*args,**kwargs)
        if not self.used:
            self.removed_count=sum(r['id'] in self.hidden for r in rows)
            rows=[r for r in rows if r['id'] not in self.hidden]
            self.used=True
        return rows,warnings
