"""Adapter to run power flow using PyPower (Python port of MATPOWER).

Provides:
- pypower_available(): bool
- run_pypower_pf(case_dict): returns dict with key 'V' (numpy complex voltages) and raw outputs

The adapter expects a MATPOWER-style case dict (keys: baseMVA, bus, gen, branch).
"""
from __future__ import annotations
from typing import Dict, Any

def pypower_available() -> bool:
    try:
        import pypower  # type: ignore
        return True
    except Exception:
        return False


def run_pypower_pf(case: Dict[str, Any]) -> Dict[str, Any]:
    """Run power flow using PyPower and return results.

    Returns {'V': numpy.array(complex), 'raw': {'bus': bus, 'gen': gen, 'branch': branch, 'success': success}}
    """
    try:
        import numpy as np
        from pypower.api import runpf
    except Exception as e:
        raise RuntimeError('PyPower not available: %s' % e)

    # Build mpc dict expected by PyPower
    mpc = {}
    mpc['baseMVA'] = float(case.get('baseMVA', 100))
    # Ensure arrays are numeric
    import numpy as _np
    mpc['bus'] = _np.array(case.get('bus', []), dtype=float)
    mpc['gen'] = _np.array(case.get('gen', []), dtype=float)
    mpc['branch'] = _np.array(case.get('branch', []), dtype=float)

    # runpf returns (baseMVA, bus, gen, branch, success, et) in some versions; others return tuple
    # Call runpf with its default signature (different pypower versions accept different args)
    out = runpf(mpc)
    # Normalize outputs: try to pick bus, gen, branch and success from returned tuple or dict
    bus = None; gen = None; branch = None; success = None
    if isinstance(out, tuple) or isinstance(out, list):
        # PyPower often returns (results_dict, success_flag). Handle that case first.
        if len(out) >= 1 and isinstance(out[0], dict):
            results_dict = out[0]
            bus = results_dict.get('bus')
            gen = results_dict.get('gen')
            branch = results_dict.get('branch')
            success = bool(out[1]) if len(out) >= 2 else results_dict.get('success')
        else:
            # fallback heuristics: look for array-like items
            for item in out:
                if hasattr(item, 'shape') and item is not None:
                    try:
                        if getattr(item, 'shape', (0,))[1] >= 8 and bus is None:
                            bus = item
                            continue
                    except Exception:
                        pass
            # success might be last boolean
            if len(out) >= 1 and isinstance(out[-1], (bool, int)):
                success = bool(out[-1])
    elif isinstance(out, dict):
        bus = out.get('bus')
        gen = out.get('gen')
        branch = out.get('branch')
        success = out.get('success')

    if bus is None:
        # try to extract bus from first returned element
        try:
            bus = out[1]
        except Exception:
            pass

    if bus is None:
        raise RuntimeError('Could not parse PyPower runpf output')

    # Extract Vm (col 7) and Va (col 8) using MATPOWER column convention (1-based): Vm col 8, Va col 9
    try:
        import numpy as _np
        bus_arr = _np.asarray(bus, dtype=float)
        Vm = bus_arr[:, 7]
        Va = bus_arr[:, 8]
        V = Vm * _np.exp(1j * _np.deg2rad(Va))
    except Exception:
        # fallback: try to compute from bus columns if available
        V = None

    return {'V': V, 'raw': {'bus': bus, 'gen': gen, 'branch': branch, 'success': success}}
