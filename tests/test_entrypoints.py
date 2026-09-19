"""Verify entry points with Python -S: no model dependencies or training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ENTRY=Path(__file__).resolve().parents[1]/'baseline&finetuning&EDCP'
MODULES=['main','evaluate_pretrain']

def run_code(module,expression,arguments=(),environment=None):
    code=f'import {module} as app; import json,sys; '+expression
    env=os.environ.copy();env.pop('CROHN_DATA_ROOT',None)
    env.update(PYTHONDONTWRITEBYTECODE='1')
    if environment:env.update(environment)
    return subprocess.run([sys.executable,'-S','-c',code,*arguments],cwd=ENTRY,capture_output=True,text=True,env=env)

@pytest.mark.parametrize('module',MODULES)
def test_help_without_site_packages(module):
    result=subprocess.run([sys.executable,'-S',str(ENTRY/(module+'.py')),'--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--dataset_dir' in result.stdout and '--pretrained_backbone' in result.stdout

@pytest.mark.parametrize('module',MODULES)
def test_import_does_not_parse_args_or_load_models(module):
    result=run_code(module,"assert all(x not in sys.modules for x in ['torch','torchvision','numpy','utils']); print('safe import')",['--deliberately-unknown'])
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='safe import'

@pytest.mark.parametrize('module',MODULES)
@pytest.mark.parametrize('value,expected',[('False',False),('0',False),('True',True),('1',True)])
def test_explicit_boolean(module,value,expected):
    result=run_code(module,'print(json.dumps(app.parse_args(sys.argv[1:]).pretrained_backbone))',['--pretrained_backbone',value])
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout) is expected

@pytest.mark.parametrize('module',MODULES)
def test_invalid_boolean_rejected(module):
    result=run_code(module,'app.parse_args(sys.argv[1:])',['--pretrained_backbone','maybe'])
    assert result.returncode==2 and 'expected true/false' in result.stderr

@pytest.mark.parametrize('module',MODULES)
def test_environment_path_and_historical_default_true(module):
    result=run_code(module,'a=app.parse_args([]); print(json.dumps([a.dataset_dir,a.pretrained_backbone]))',environment={'CROHN_DATA_ROOT':'synthetic/folds'})
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==['synthetic/folds',True]

@pytest.mark.parametrize('module',MODULES)
def test_cli_path_overrides_environment(module):
    result=run_code(module,'print(json.dumps(app.parse_args(sys.argv[1:]).dataset_dir))',['--dataset_dir','override/folds'],environment={'CROHN_DATA_ROOT':'synthetic/folds'})
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)=='override/folds'
