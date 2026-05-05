from __future__ import annotations
import os, sys, time, json, contextlib
import types
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IXI = os.path.join(REPO, 'IXI')
TM  = os.path.join(IXI, 'TransMorph')
BLC = os.path.join(IXI, 'Baseline_registration_methods')
BLT = os.path.join(IXI, 'Baseline_Transformers')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SZ  = (160,192,224)
WARMUP, REPEATS = 5, 20


def add(p):
    if p not in sys.path:
        sys.path.insert(0, p)

def purge(prefix):
    for k in list(sys.modules.keys()):
        if k == prefix or k.startswith(prefix + '.'):
            sys.modules.pop(k, None)

def ensure_ns(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    sys.modules[name] = mod

def install_nnunet_stubs():
    if 'nnunet.utilities.random_stuff' in sys.modules and 'nnunet.utilities.to_torch' in sys.modules:
        return
    ensure_ns('nnunet', '')
    ensure_ns('nnunet.utilities', '')
    m_rand = types.ModuleType('nnunet.utilities.random_stuff')
    m_torch = types.ModuleType('nnunet.utilities.to_torch')

    @contextlib.contextmanager
    def no_op():
        yield

    def maybe_to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.from_numpy(x)

    def to_cuda(x, gpu_id=None, non_blocking=True):
        if not isinstance(x, torch.Tensor):
            x = maybe_to_torch(x)
        if torch.cuda.is_available():
            if gpu_id is None or gpu_id == 'cpu':
                return x.cuda(non_blocking=non_blocking)
            return x.cuda(gpu_id, non_blocking=non_blocking)
        return x

    m_rand.no_op = no_op
    m_torch.maybe_to_torch = maybe_to_torch
    m_torch.to_cuda = to_cuda
    sys.modules['nnunet.utilities.random_stuff'] = m_rand
    sys.modules['nnunet.utilities.to_torch'] = m_torch

def bench(fn):
    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
    if DEV == 'cuda':
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts=[]
    with torch.no_grad():
        for _ in range(REPEATS):
            t0=time.perf_counter(); fn()
            if DEV=='cuda': torch.cuda.synchronize()
            ts.append(time.perf_counter()-t0)
    mem = float(torch.cuda.max_memory_allocated()/1e9) if DEV=='cuda' else 0.0
    return float(np.mean(ts)), float(np.std(ts, ddof=0)), mem

def nparam(m):
    return float(sum(p.numel() for p in m.parameters())/1e6)

res={}
x2 = torch.randn(1,2,*SZ,device=DEV)
x1 = torch.randn(1,1,*SZ,device=DEV)

# HypEReg-TransMorph
purge('models'); add(TM)
from models.TransMorph import CONFIGS as C_TM
import models.TransMorph as TMm
m = TMm.TransMorph(C_TM['TransMorph']).to(DEV).eval()
mu,sd,pm=bench(lambda: m(x2))
res['HypEReg-TransMorph']={'mean_s':mu,'std_s':sd,'peak_gb':pm,'params_M':nparam(m)}
res['TransMorph']={'mean_s':mu,'std_s':sd,'peak_gb':pm,'params_M':nparam(m)}

# Bayes single-pass
from models.TransMorph_Bayes import CONFIGS as C_B
import models.TransMorph_Bayes as TB
mb = TB.TransMorphBayes(C_B['TransMorphBayes']).to(DEV).eval()
mu,sd,pm=bench(lambda: mb(x2))
res['TransMorphBayes']={'mean_s':mu,'std_s':sd,'peak_gb':pm,'params_M':nparam(mb),
                        'note':'single-pass pure forward; MC eval uses 25 passes'}

# VoxelMorph
purge('models'); add(os.path.join(BLC,'VoxelMorph'))
from models import VxmDense_1
vxm = VxmDense_1(SZ).to(DEV).eval()
mu,sd,pm=bench(lambda: vxm(x2))
res['VoxelMorph-1']={'mean_s':mu,'std_s':sd,'peak_gb':pm,'params_M':nparam(vxm)}

# CycleMorph
purge('models'); purge('util'); add(os.path.join(BLC,'CycleMorph'))
from models.cycleMorph_model import CONFIGS as C_C, cycleMorph
cmobj=cycleMorph(); cmobj.initialize(C_C['Cycle-Morph-v0'])
cm=cmobj.netG_A.to(DEV).eval()
mu,sd,pm=bench(lambda: cm(x2))
res['CycleMorph']={'mean_s':mu,'std_s':sd,'peak_gb':pm,'params_M':nparam(cm)}

# MIDIR
purge('models'); purge('transformation'); add(os.path.join(BLC,'MIDIR'))
import models as MM
mm=MM.CubicBSplineNet(ndim=3).to(DEV).eval()
mu,sd,pm=bench(lambda: mm((x1,x1)))
res['MIDIR']={'mean_s':mu,'std_s':sd,'peak_gb':pm,'params_M':nparam(mm)}

# CoTr
purge('models'); purge('CoTr'); add(os.path.join(BLT,'models'))
ensure_ns('models', os.path.join(BLT, 'models'))
install_nnunet_stubs()
from CoTr.network_architecture.ResTranUnet import ResTranUnet as CoTr
cotr=CoTr().to(DEV).eval()
mu,sd,pm=bench(lambda: cotr(x2))
res['CoTr']={'mean_s':mu,'std_s':sd,'peak_gb':pm,'params_M':nparam(cotr)}

# nnFormer
purge('models'); purge('nnFormer'); add(os.path.join(BLT,'models'))
ensure_ns('models', os.path.join(BLT, 'models'))
from nnFormer.Swin_Unet_l_gelunorm import swintransformer as NN
nnf=NN().to(DEV).eval()
mu,sd,pm=bench(lambda: nnf(x2))
res['nnFormer']={'mean_s':mu,'std_s':sd,'peak_gb':pm,'params_M':nparam(nnf)}

# PVT
purge('models'); purge('PVT'); add(os.path.join(BLT,'models'))
ensure_ns('models', os.path.join(BLT, 'models'))
from PVT import CONFIGS as C_P, PVTVNetSkip
pvt=PVTVNetSkip(C_P['PVT-Net']).to(DEV).eval()
mu,sd,pm=bench(lambda: pvt(x2))
res['PVT']={'mean_s':mu,'std_s':sd,'peak_gb':pm,'params_M':nparam(pvt)}

res['SyN (ANTs)']={'mean_s':None,'std_s':None,'peak_gb':None,'params_M':None,
                   'note':'classical iterative optimizer'}

for k,v in res.items():
    if v.get('mean_s') is None:
        print(f'{k}: N/A')
    else:
        print(f"{k}: {v['mean_s']:.4f}s ±{v['std_s']:.4f}, {v['peak_gb']:.3f}GB, {v['params_M']:.3f}M")

out=os.path.join(IXI,'Results','profile_pure_forward_all.json')
os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,'w',encoding='utf-8') as f: json.dump(res,f,indent=2)
print('Saved',out)
