import inspect, torch
from pypto.runtime import RunConfig
from models.deepseek_v4_1_flash.tp_ep_layer import l3_tp_ep_layer_tail as fn
args=[]
for name,p in inspect.signature(fn._func).parameters.items():
 a=p.annotation
 if not hasattr(a,'shape'):
  args.append(1); continue
 dims=[d if isinstance(d,int) else (17 if 'OUTPUT' in str(d) else 68) for d in a.shape]
 dtype={'bfloat16':torch.bfloat16,'fp32':torch.float32,'fp8e4m3fn':torch.float8_e4m3fn,'fp8e8m0':torch.float8_e8m0fnu,'int32':torch.int32,'int8':torch.int8,'uint8':torch.uint8}.get(str(a.dtype))
 print(name,dims,str(a.dtype),flush=True)
 if dtype is None: raise ValueError(str(a.dtype))
 args.append(torch.empty(dims,dtype=dtype,device='meta'))
else:
 fn.compile(*args,config=RunConfig(platform='a5',codegen_only=True))
 print('CONCRETE L3 CODEGEN PASS')
