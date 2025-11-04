from __future__ import annotations
from __future__ import annotations
import copy
import numpy as np
from typing import Dict, Any, List
from powerflow_runner import run_pf_and_flows


def severity_score(overload_count: int, max_loading_ratio: float | None, max_voltage_drop: float | None, weights=(1000.0, 100.0, 10.0)) -> float:
    """Compute a numeric severity score for a contingency.

    Higher is more severe. Weights default to emphasize overload count,
    then loading ratio excess, then voltage drop.
    """
    w_over, w_load, w_vdrop = weights
    load_excess = 0.0
    if max_loading_ratio is not None:
        load_excess = max(0.0, max_loading_ratio - 1.0)
    vdrop = max(0.0, max_voltage_drop or 0.0)
    return overload_count * w_over + load_excess * w_load + vdrop * w_vdrop


def single_contingency(case: Dict[str, Any], branch_idx: int, base_V=None) -> Dict[str, Any]:
    br = case['branch'][branch_idx]
    case2 = copy.deepcopy(case)
    # set branch out of service
    if len(case2['branch'][branch_idx]) > 10:
        case2['branch'][branch_idx][10] = 0
    else:
        while len(case2['branch'][branch_idx]) <= 10:
            case2['branch'][branch_idx].append(0)
        case2['branch'][branch_idx][10] = 0

    try:
        res2, flows2 = run_pf_and_flows(case2)
        V2 = res2.get('V')
        success = V2 is not None
    except Exception:
        flows2 = []
        V2 = None
        success = False

    # compute metrics
    # prepare rateA vector from case branches (fallback to inf when non-positive)
    rates = []
    for b in case.get('branch', []):
        rA = float(b[5]) if len(b) > 5 else 0.0
        rates.append(np.inf if (rA is None or rA <= 0) else float(rA))

    max_loading = None
    overload_count = 0
    max_pct_over = 0.0
    if flows2:
        branch_ratios = []
        for idx_f, f in enumerate(flows2):
            rate = rates[idx_f] if idx_f < len(rates) else np.inf
            s_from = float(f.get('S_from', 0.0) or 0.0)
            s_to = float(f.get('S_to', 0.0) or 0.0)
            # compute max of both ends against rate
            if rate and rate != np.inf and rate > 0:
                r_from = s_from / rate
                r_to = s_to / rate
                br_ratio = max(r_from, r_to)
                branch_ratios.append(br_ratio)
                if br_ratio > 1.0:
                    overload_count += 1
                    # percent over = max((s/rate -1)*100)
                    max_pct_over = max(max_pct_over, max((r_from - 1.0) * 100.0 if r_from > 1.0 else 0.0, (r_to - 1.0) * 100.0 if r_to > 1.0 else 0.0))
            else:
                # rate is infinite or zero meaning no limit -> skip
                continue
        if branch_ratios:
            max_loading = max(branch_ratios)

    # voltage metrics: count violations and maximum deviation beyond limits
    max_v_drop = None
    nVoltVio = 0
    maxVoltDev = 0.0
    # extract vmin/vmax from case bus entries if available (MATPOWER-like layout)
    vmins = []
    vmaxs = []
    for b in case.get('bus', []):
        # VMAX at index 11, VMIN at index 12 (0-based)
        vmaxs.append(float(b[11]) if len(b) > 11 else 0.0)
        vmins.append(float(b[12]) if len(b) > 12 else 0.0)
    vmins = np.array(vmins) if vmins else None
    vmaxs = np.array(vmaxs) if vmaxs else None
    # provide fallbacks similar to MATPOWER
    if vmins is None or np.all(vmins == 0):
        if vmins is None:
            vmins = np.array([])
        vmins = np.full(len(case.get('bus', [])), 0.90)
    if vmaxs is None or np.all(vmaxs == 0):
        if vmaxs is None:
            vmaxs = np.array([])
        vmaxs = np.full(len(case.get('bus', [])), 1.10)

    if base_V is not None and V2 is not None:
        try:
            mags_base = np.abs(base_V)
            mags_cont = np.abs(V2)
            L = min(len(mags_base), len(mags_cont), len(vmins), len(vmaxs))
            if L > 0:
                Vm = mags_cont[:L]
                vmin_arr = vmins[:L]
                vmax_arr = vmaxs[:L]
                vio = (Vm < vmin_arr) | (Vm > vmax_arr)
                nVoltVio = int(np.count_nonzero(vio))
                # max deviation beyond limits
                max_dev_upper = np.max(np.maximum(Vm - vmax_arr, 0.0)) if np.any(~np.isnan(Vm)) else 0.0
                max_dev_lower = np.max(np.maximum(vmin_arr - Vm, 0.0)) if np.any(~np.isnan(Vm)) else 0.0
                maxVoltDev = float(max(max_dev_upper, max_dev_lower, 0.0))
                # also compute max voltage drop relative to base (base - cont)
                max_v_drop = float(np.max(mags_base[:L] - mags_cont[:L]))
        except Exception:
            max_v_drop = None
            nVoltVio = 0
            maxVoltDev = 0.0

    # severity: match MATLAB approach (large penalty for non-convergence)
    try:
        iters = res2.get('iterations') if isinstance(res2, dict) else None
    except Exception:
        iters = None

    # weights similar to MATLAB defaults
    wV = 100.0
    wO = 1.0
    if not success:
        SI = 1e6
    else:
        SI = wV * maxVoltDev + wO * max_pct_over

    return {
        'branch': branch_idx + 1,
        'from': int(br[0]),
        'to': int(br[1]),
        'converged': bool(success),
        'iters': int(iters) if iters is not None and not np.isnan(iters) else None,
        'nVoltVio': int(nVoltVio),
        'maxVoltDev': float(maxVoltDev),
        'nOver': int(overload_count),
        'maxPctOver': float(max_pct_over),
        'Severity': float(SI),
    }


def run_n1_sweep_serial(case: Dict[str, Any], base_V=None, branches: list | None = None) -> List[Dict[str, Any]]:
    nb = len(case.get('branch', []))
    idxs = list(range(nb)) if branches is None else branches
    results = []
    for i in idxs:
        br = case['branch'][i]
        try:
            status = int(br[10]) if len(br) > 10 else 1
        except Exception:
            status = 1
        if status == 0:
            continue
        results.append(single_contingency(case, i, base_V=base_V))
    return results


def run_n1_sweep_parallel(case: Dict[str, Any], base_V=None, branches: list | None = None, max_workers: int | None = None):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    nb = len(case.get('branch', []))
    idxs = list(range(nb)) if branches is None else branches
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as exc:
        futures = {exc.submit(single_contingency, case, i, base_V): i for i in idxs}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                i = futures[fut]
                results.append({'branch': i+1, 'from': None, 'to': None, 'success': False, 'overload_count': 0, 'max_loading_ratio': None, 'max_voltage_drop': None, 'severity_score': 0.0})
    return results
    
