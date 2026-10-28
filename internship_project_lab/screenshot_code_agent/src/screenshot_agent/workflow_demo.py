"""Deterministic demonstration with real sandbox execution; never a model score."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from openai.types.chat import ChatCompletion
from PIL import Image, ImageDraw, ImageFont

from .config import ImageInput, Task
from .models import APIModel
from .workflow import run_workflow_task
from .workflow_config import load_workflow_config


def patch(old,new):
    return f'<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE'


class DemoAPI(APIModel):
    """Scripted model interface. Tool calls below run in the real sandbox."""
    script = []

    def __init__(self,config,budget,events,**kwargs):
        self.config,self.budget,self.events,self.policy = config,budget,events,kwargs['policy']
        self.commands = iter(type(self).script)
        self.number = 0

    def complete(self,messages,tools=None,**kwargs):
        reserve = self.budget.reserve('code',self.config,messages,tools)
        self.budget.settle(self.config,reserve,SimpleNamespace(prompt_tokens=10,completion_tokens=10))
        self.number += 1
        item = next(self.commands)
        if isinstance(item,BaseException):
            raise item
        message = {'role':'assistant','content':item} if isinstance(item,str) else {
            'role':'assistant','content':'演示预设操作，实际执行工具。',
            'tool_calls':[{'id':f'demo_{self.number}','type':'function',
                           'function':{'name':item[0],'arguments':json.dumps(item[1])}}]}
        response = ChatCompletion.model_validate({'id':'demo','object':'chat.completion','created':0,
            'model':'scripted-demo-not-inference','choices':[{'index':0,'finish_reason':'stop','message':message}],
            'usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20}})
        self.events.emit('model_response',kind='scripted_demo',response=response.model_dump())
        return response


PLAN = json.dumps({'steps':[{'phase':'locate','description':'查找 discounted_price 的定义与测试'},
    {'phase':'reproduce','description':'运行已有单元测试，确认百分比折扣和小数金额的行为'},
    {'phase':'repair','description':'修复百分比计算，再通过测试反馈检查金额精度'}]},ensure_ascii=False)


def demo(output):
    folder = Path(output)/'demo'
    if folder.exists():
        folder = folder.with_name('demo_'+str(time.time_ns()))
    repo = folder/'input/repo'
    (repo/'tests').mkdir(parents=True)
    (repo/'pricing.py').write_text('def discounted_price(price, percent):\n    return price - percent\n')
    (repo/'tests/test_pricing.py').write_text('import unittest\nfrom pricing import discounted_price\n\n'
        'class PricingTest(unittest.TestCase):\n'
        '    def test_percent(self):\n        self.assertEqual(discounted_price(200, 10), 180)\n'
        '    def test_fraction(self):\n        self.assertEqual(discounted_price(80.5, 10), 72.45)\n')
    visible = 'FAIL: tests/test_pricing.py\nAssertionError: 190 != 180\n\n'
    visible += 'discounted_price(200, 10) should return 180\nSee pricing.py:2\n'
    reference = folder/'input/visible.txt'
    reference.write_text(visible)
    image = Image.new('RGB',(1020,280),'#172133')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',23)
    draw.multiline_text((28,28),visible,font=font,fill='#e6f0fa',spacing=13)
    screenshot = folder/'input/failure.png'
    image.save(screenshot)
    task = Task(task_id='discount_repair_demo',repo=str(repo),
        problem='根据报错截图修复折扣金额计算，保留小数金额并四舍五入到两位。',
        test_command='python3 -m unittest discover -s tests -v',protected_paths=['tests'],
        images=[ImageInput(image_id='error_screen',path=str(screenshot),reference=str(reference))])
    DemoAPI.script = [PLAN,('repo_search',{'query':'discounted_price pricing.py'}),
        ('read_file',{'file_path':'pricing.py'}),'已定位到金额计算函数。',
        ('run_tests',{}),'已复现两个失败测试。',
        ('apply_patch',{'file_path':'pricing.py','patch':patch('return price - percent','return int(price * (1 - percent / 100))')}),
        '先修复了百分比计算，等待独立测试。',
        json.dumps({'issues':['小数金额被 int 截断'],'next_action':'去掉整数转换，将结果四舍五入到两位。'},ensure_ascii=False),
        ('apply_patch',{'file_path':'pricing.py','patch':patch('return int(price * (1 - percent / 100))','return round(price * (1 - percent / 100), 2)')}),
        '已保留两位小数，等待最终验证。']
    config = load_workflow_config(Path(__file__).resolve().parents[2]/'configs/workflow-local.json')
    result = run_workflow_task(task,config,folder/'run',reader_name='oracle',counter=lambda m,t:100,
                              api_factory=DemoAPI,model_inference=False)
    return {'result':result,'report':str(folder/'run/report.html'),
            'delivery':str(folder/'run/delivery.zip'),'gpu_started':False,
            'note':'预设模型响应 + 实际沙箱工具和测试；用于演示完整流程，不代表真实模型成功率。'}
