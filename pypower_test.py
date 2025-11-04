import traceback
import json
import os
from pypower_adapter import pypower_available, run_pypower_pf
from matpower_parser import load_case

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
case_path = os.path.join(BASE_DIR, 'cases', 'case9.json')
print('case_path:', case_path)
try:
    case = load_case(case_path)
    print('Loaded case: baseMVA', case.get('baseMVA'), 'buses', len(case.get('bus', [])), 'branches', len(case.get('branch', [])))
except Exception as e:
    print('Failed to load case json:', e)
    traceback.print_exc()
    raise

print('pypower_available():', pypower_available())
try:
    res = run_pypower_pf(case)
    V = res.get('V')
    print('run_pypower_pf returned. V is None?', V is None)
    if V is not None:
        print('Voltage magnitudes (first 10):', [abs(v) for v in V[:10]])
        print('Voltage angles deg (first 10):', [float(__import__('numpy').angle(v, deg=True)) for v in V[:10]])
    else:
        raw = res.get('raw', {})
        print('Raw output keys:', list(raw.keys()))
        bus_raw = raw.get('bus')
        print('bus_raw type:', type(bus_raw))
        try:
            import numpy as _np
            print('bus_raw as numpy shape:', _np.asarray(bus_raw).shape)
            print('bus_raw first row:', _np.asarray(bus_raw)[0])
        except Exception as e:
            print('Could not inspect bus_raw:', e)
except Exception as e:
    print('PyPower run failed:')
    traceback.print_exc()
    raise
