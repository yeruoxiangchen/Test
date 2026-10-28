import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from Levenshtein import distance
from PIL import Image

from screenshot_agent import codeocr_benchmark as bench
from screenshot_agent.config import RunConfig
from screenshot_agent.images import Reading
from screenshot_agent.report import edit_distance


def upstream_fake(checkout):
    def render(source,**kwargs):
        return [Image.new('RGB',(800,200),'white')]

    def corrupt(path,ratio):
        p=Path(path).with_name(f'original_ratio{ratio}.png')
        p.write_bytes(Path(path).read_bytes())
        return str(p)

    def metrics(reference,hypothesis):
        return {'cer':distance(reference,hypothesis)/len(reference),'wer':0,'line_sim':0}

    return render,corrupt,metrics


def test_full_builder_keeps_long_sources_and_all_variants(tmp_path,monkeypatch):
    checkout=tmp_path/'upstream'
    (checkout/'reconstruction/dataset').mkdir(parents=True)
    source=[{'id':'long','repo':'a/repo','url':'https://example.org/a','code':'value = 1234567890\n'*130},
            {'id':'small','repo':'b/repo','url':'https://example.org/b','code':'value = 1\n'}]
    (checkout/'reconstruction/dataset/dataset.json').write_text(json.dumps(source))
    monkeypatch.setattr(bench,'upstream',upstream_fake)
    font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
    report=bench.build_full_codeocr(checkout,tmp_path/'dataset',font,font)
    assert report['selected_files']==report['original_files']==2
    rows=[json.loads(s) for s in (tmp_path/'dataset/images.jsonl').read_text().splitlines()]
    assert len(rows)==4 and {r['ratio'] for r in rows}=={1,2}
    assert (tmp_path/'dataset/sample_000/reference.py').read_text()==source[0]['code']
    assert len({r['split'] for r in rows if r['source_id']=='long'})==1


def make_inputs(tmp_path):
    dataset=tmp_path/'data'
    dataset.mkdir()
    Image.new('RGB',(80,40),'white').save(dataset/'shot.png')
    (dataset/'source.py').write_text('x = 1\n')
    (dataset/'build_report.json').write_text(json.dumps({'checkout':str(tmp_path)}))
    rows=[{'sample_id':f'{name}_{ratio}','source_id':name,'ratio':ratio,'image':'shot.png','reference':'source.py'}
          for name in ['a','b'] for ratio in [1,2]]
    manifest=dataset/'images.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return manifest


def test_shards_cover_all_sources_and_resume_does_not_retry_failures(tmp_path,monkeypatch):
    manifest=make_inputs(tmp_path)
    monkeypatch.setattr(bench,'upstream',upstream_fake)
    calls=[]

    class Reader:
        def __init__(self,store):
            pass

        def read(self,image_id):
            calls.append(image_id)
            raise RuntimeError('record this failure')

    monkeypatch.setattr(bench,'OCRReader',Reader)
    records=[]
    for index in [0,1]:
        output=tmp_path/f'shard_{index}'
        result=bench.evaluate_codeocr(manifest,RunConfig(),output,('ocr',),shard_index=index,shard_count=2)
        assert len(result['records'])==2 and result['summary'][0]['cer_including_errors_mean']==1
        again=bench.evaluate_codeocr(manifest,RunConfig(),output,('ocr',),shard_index=index,shard_count=2,resume=True)
        assert again['records']==result['records']
        records.extend(result['records'])
    assert len(calls)==4
    assert {r['sample_id'] for r in records}=={'a_1','a_2','b_1','b_2'}


def test_resume_rejects_changed_inputs(tmp_path,monkeypatch):
    manifest=make_inputs(tmp_path)
    monkeypatch.setattr(bench,'upstream',upstream_fake)

    class Reader:
        def __init__(self,store):
            pass

        def read(self,image_id):
            return Reading(raw_text='x = 1\n')

    monkeypatch.setattr(bench,'OCRReader',Reader)
    bench.evaluate_codeocr(manifest,RunConfig(),tmp_path/'run',('ocr',))
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError,match='different dataset'):
        bench.evaluate_codeocr(manifest,RunConfig(),tmp_path/'run',('ocr',),resume=True)


@pytest.mark.parametrize('text',['    x = 1\n','中文✅','one\u0085two'])
def test_accelerated_raw_cer_matches_the_existing_definition(text):
    assert distance(text,'x = 1')==edit_distance(text,'x = 1')


def test_finalizer_requires_exact_coverage_and_ignores_worker_log_files(tmp_path,monkeypatch):
    project=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('full_supervisor',project/'scripts/run_codeocr_full.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    output=tmp_path/'run'
    dataset=tmp_path/'dataset'
    dataset.mkdir()
    output.mkdir()
    (dataset/'image.png').write_bytes(b'fixture image')
    (dataset/'reference.py').write_text('x = 1\n')
    rows=[{'sample_id':f'{index}_{ratio}','source_id':str(index),'image':'image.png','reference':'reference.py',
           'ratio':ratio,'image_sha256':hashlib.sha256(b'fixture image').hexdigest(),
           'source_sha256':hashlib.sha256(b'x = 1\n').hexdigest()} for index in range(100) for ratio in [1,2]]
    (dataset/'images.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (dataset/'build_report.json').write_text('{}')
    (output/'source_state.json').write_text('{}')
    (output/'source_snapshot.zip').write_bytes(b'snapshot fixture')
    (tmp_path/'CODEOCR_FULL_UPSTREAM.json').write_text('{}')
    config={'code':{'base_url':'fixture'},'vision':{'base_url':'fixture'}}
    expected=[]
    for row in rows:
        for reader in ['ocr','vlm']:
            expected.append({**row,'reader':reader,'status':'complete','metrics':{'cer':0,'wer':0,'line_sim':1},
                             'raw_cer':0,'hypothesis_parses':True,'reference_parses':True,'exact_match':True,'elapsed_seconds':.1})
    for index in [0,1]:
        folder=output/f'shard_{index}'
        (folder/'codeocr').mkdir(parents=True)
        (output/f'shard_{index}.log').write_text('not a worker directory')
        for name,value in [('config.json',config),('launcher_status.json',{'exit_code':0}),
                           ('final_model_stats.json',{'peak_cuda_memory_bytes':1024}),('launch.json',{})]:
            (folder/name).write_text(json.dumps(value))
        (folder/'codeocr/report.json').write_text(json.dumps({'complete':True,'records':expected[index::2]}))
    before=expected[-1]
    f=output/'shard_1/codeocr/report.json'
    f.write_text(json.dumps({'complete':True,'records':expected[1::2][:-1]}))
    with pytest.raises(ValueError,match='covered exactly once'):
        module.finalize(output,dataset,{'source_sha256':{}})
    f.write_text(json.dumps({'complete':True,'records':[*expected[1::2][:-1],before]}))
    report=module.finalize(output,dataset,{'source_sha256':{}})
    assert len(report['records'])==400
    assert (tmp_path/'docs/artifacts/codeocr_full_v4/verification.json').exists()
