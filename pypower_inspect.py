import numpy as np
from pypower.api import runpf
from matpower_parser import load_case
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
case = load_case(os.path.join(BASE_DIR,'cases','case9.json'))
mpc = {'baseMVA': float(case.get('baseMVA',100)), 'bus': np.array(case.get('bus',[]),dtype=float), 'gen': np.array(case.get('gen',[]),dtype=float), 'branch': np.array(case.get('branch',[]),dtype=float)}
print('Calling runpf...')
out = runpf(mpc)
print('type(out)=', type(out))
try:
    import inspect
    print('signature runpf:', inspect.signature(runpf))
except Exception:
    pass
try:
    import pprint
    pprint.pprint(out)
except Exception as e:
    print('Could not pprint out:', e)
