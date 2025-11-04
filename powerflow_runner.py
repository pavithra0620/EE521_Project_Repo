import copy
import numpy as np
from typing import Tuple, List, Dict, Any

from matpower_adapter import run_matpower_pf, matlab_engine_available
from pypower_adapter import run_pypower_pf, pypower_available
from powerflow import newton_raphson


def compute_branch_flows(case: Dict[str, Any], V: np.ndarray) -> List[Dict[str, Any]]:
    flows = []
    for br in case.get('branch', []):
        if len(br) < 4:
            continue
        f = int(br[0]) - 1
        t = int(br[1]) - 1
        r = float(br[2]); x = float(br[3])
        rateA = float(br[5]) if len(br) > 5 else 0.0
        ratio = float(br[8]) if len(br) > 8 else 0.0
        status = int(br[10]) if len(br) > 10 else 1
        if status == 0:
            flows.append({'f': f+1, 't': t+1, 'S_from': 0.0, 'rateA': rateA, 'status': status})
            continue
        z = r + 1j * x
        y = 0+0j if abs(z) < 1e-12 else 1.0 / z
        a = ratio if (ratio not in (0.0, None)) else 1.0
        try:
            Vf = V[f]; Vt = V[t]
        except Exception:
            Vf = 1.0+0j; Vt = 1.0+0j
        if a == 0:
            a = 1.0
        I_from = (Vf / a - Vt) * y
        S_from = Vf * np.conj(I_from)
        flows.append({'f': f+1, 't': t+1, 'S_from': float(abs(S_from)), 'rateA': rateA, 'status': status})
    return flows


def run_pf_and_flows(case: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Prefer PyPower, then MATPOWER via MATLAB, then Python NR
    if pypower_available():
        try:
            pr = run_pypower_pf(case)
            V = pr.get('V')
            if V is not None:
                res = {'V': V}
                flows = compute_branch_flows(case, V)
                return res, flows
        except Exception:
            pass

    if matlab_engine_available():
        try:
            mp = run_matpower_pf(case)
            V = mp.get('V')
            if V is not None:
                res = {'V': V}
                flows = compute_branch_flows(case, V)
                return res, flows
        except Exception:
            pass

    # fallback to python NR solver
    res = newton_raphson(case, tol=1e-8, maxiter=200, damping=0.5)
    V = res['V']
    flows = compute_branch_flows(case, V)
    return res, flows
