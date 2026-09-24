"""External CPU-only A5 tail assembly/binary validation; never initializes a worker."""
import inspect
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from pypto.runtime import RunConfig, device_runner
from models.deepseek_v4_1_flash.tp_ep_layer import l3_tp_ep_layer_tail as fn
os.environ['PYPTO_CODEGEN_MAX_WORKERS']='8'
os.environ['PYPTO_COMPILER_TIMEOUT']='120'
def bounded_pool(*args,**kwargs):
    kwargs['max_workers']=min(8,kwargs.get('max_workers',8))
    print('Binary compiler worker cap:',kwargs['max_workers'],flush=True)
    return ThreadPoolExecutor(*args,**kwargs)
device_runner.ThreadPoolExecutor=bounded_pool
args=[]
for name,p in inspect.signature(fn._func).parameters.items():
    a=p.annotation
    if not hasattr(a,'shape'):
        args.append(1)
        continue
    dims=[d if isinstance(d,int) else (17 if 'OUTPUT' in str(d) else 68) for d in a.shape]
    dtype={'bfloat16':torch.bfloat16,'fp32':torch.float32,'fp8e4m3fn':torch.float8_e4m3fn,'fp8e8m0':torch.float8_e8m0fnu,'int32':torch.int32,'int8':torch.int8,'uint8':torch.uint8}[str(a.dtype)]
    args.append(torch.empty(dims,dtype=dtype,device='meta'))
compiled=fn.compile(*args,config=RunConfig(platform='a5'))
print('A5 TAIL PTOAS PASS',compiled.output_dir,flush=True)
for cfg in sorted(Path(compiled.output_dir).glob('next_levels/*/kernel_config.py')):
    print('BUILD',cfg.parent,flush=True)
    device_runner._compile_and_assemble(cfg.parent,'a5')
    print('A5 TAIL BINARY PASS',cfg.parent,flush=True)