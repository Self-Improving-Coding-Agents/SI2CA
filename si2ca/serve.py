"""Launch one explicit SGLang replica and wait until it is ready."""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

from si2ca.model_files import is_hf_id, parser_defaults, resolve_files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True)
    p.add_argument('--name',help='Served model ID; defaults to --model-path without abbreviation')
    p.add_argument('--gpus',required=True,help='Explicit physical GPU IDs, e.g. 0,1,2,3')
    p.add_argument('--tp',type=int,required=True)
    p.add_argument('--port',type=int,default=8151)
    p.add_argument('--context-length',type=int,help='Default: infer from model configuration')
    p.add_argument('--tool-call-parser',default='auto',help='auto, none, or an explicit SGLang parser name')
    p.add_argument('--reasoning-parser',default='auto',help='auto, none, or an explicit SGLang parser name')
    p.add_argument('--revision',help='Hugging Face branch, tag or commit')
    p.add_argument('--model-cache-dir',help='Hugging Face cache directory')
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--mem-fraction',type=float,default=.8)
    p.add_argument('--log',required=True)
    p.add_argument('--rocm',action='store_true')
    return launch(p.parse_args())


def launch(a):
    gpus=a.gpus.split(',')
    if len(gpus)!=a.tp or len(set(gpus))!=len(gpus) or not all(x.isdigit() for x in gpus):
        raise ValueError('One replica requires exactly tp distinct GPU IDs')
    if not 0<a.mem_fraction<1 or not 1<=a.port<55536:
        raise ValueError('mem-fraction must be in (0,1); port must leave room for auxiliary ports')
    source = getattr(a, 'model_id', None) or a.model_path
    local_override = a.model_path if source != a.model_path else None
    model = str(Path(a.model_path).expanduser().resolve()) if Path(a.model_path).expanduser().is_dir() else a.model_path
    if not Path(model).is_dir() and not is_hf_id(source):
        raise ValueError('Use an existing local model directory or a Hugging Face org/model ID')
    if a.context_length is not None and a.context_length < 1:
        raise ValueError('context-length must be positive')
    model = resolve_files(source, local_path=local_override, revision=getattr(a, 'revision', None),
        cache_dir=getattr(a, 'model_cache_dir', None), local_files_only=getattr(a, 'local_files_only', False))
    automatic = parser_defaults(source, model if Path(model).is_dir() else None)
    command=[sys.executable,'-m','sglang.launch_server','--model-path',str(model),
             '--served-model-name',a.name or source,'--host','127.0.0.1','--port',str(a.port),
             '--tp',str(a.tp),'--mem-fraction-static',str(a.mem_fraction),'--trust-remote-code']
    if a.context_length is not None:
        command += ['--context-length', str(a.context_length)]
    for key, default in zip(('tool_call_parser', 'reasoning_parser'), automatic):
        value = getattr(a, key, 'auto')
        value = default if value == 'auto' else value
        if value and value != 'none':
            command += ['--' + key.replace('_', '-'), value]
    print(shlex.join(command),flush=True)
    env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=a.gpus
    if a.rocm:
        env.update(HIP_VISIBLE_DEVICES=a.gpus,ROCR_VISIBLE_DEVICES=a.gpus,SGLANG_USE_AITER='0',
                   HSA_NO_SCRATCH_RECLAIM='1',HIP_FORCE_DEV_KERNARG='1')
    log=Path(a.log).resolve();log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as stream:
        process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(f'Started PID {process.pid}; log {log}',flush=True)
    try:
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError(f'Server exited with {process.returncode}; inspect {log}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health',timeout=5) as response:
                    if response.status==200:
                        print(f'Ready: http://127.0.0.1:{a.port}/v1 ; PID {process.pid}',flush=True)
                        return process
            except OSError:pass
            time.sleep(5)
        raise TimeoutError(f'Server not ready after 1200 seconds; see {log}')
    except BaseException:
        stop(process)
        raise


def stop(process):
    """Stop only the process group created by launch(), including its GPU workers."""
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        process.poll()  # Reap the parent, but also wait for its worker process group.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.1)
    try:
        os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


if __name__=='__main__':main()
