"""MATPOWER adapter: try to run MATPOWER powerflow using MATLAB Engine or matlab CLI.

Behavior:
- If MATLAB Engine for Python is installed and MATLAB/MATPOWER are accessible, use the engine to
  run `runpf` on a temporary case and return results as a Python dict.
- If MATLAB Engine is not available, but a `matlab` executable is on PATH and MATPOWER is on MATLAB
  path, the CLI mode can be used (not implemented here). For now, adapter reports unavailable.

This adapter is best-effort: it will raise informative errors if MATLAB or MATPOWER are not found.
"""
from __future__ import annotations
import os
import tempfile
import json
import shutil
import subprocess
from typing import Dict, Any


def matlab_engine_available() -> bool:
    try:
        import matlab.engine  # type: ignore
        return True
    except Exception:
        return False


def run_matpower_pf_with_engine(case: Dict[str, Any], matpower_path: str | None = None) -> Dict[str, Any]:
    """Run MATPOWER `runpf` using MATLAB Engine and return results as a Python dict.

    Args:
        case: MATPOWER-style Python dict (fields: baseMVA, bus, gen, branch)
        matpower_path: optional path to MATPOWER installation; if provided, it will be added to MATLAB path.

    Returns:
        results dict with at least key 'V' (numpy array of complex voltages) and optionally other fields.
    """
    try:
        import matlab.engine  # type: ignore
        import numpy as np
    except Exception as e:
        raise RuntimeError('MATLAB Engine is not available: ' + str(e))

    eng = matlab.engine.start_matlab()
    # Add MATPOWER to path if provided
    if matpower_path:
        eng.addpath(matpower_path, nargout=0)

    # Create a temporary folder and write a MATPOWER .m case file
    tmpdir = tempfile.mkdtemp(prefix='mp_adapter_')
    try:
        case_name = 'mpcase_tmp'
        case_file = os.path.join(tmpdir, case_name + '.m')
        # Write a minimal .m file that defines mpc struct from JSON
        with open(case_file, 'w', encoding='utf-8') as f:
            f.write('function mpc = %s()\n' % case_name)
            f.write('%% Auto-generated MATPOWER case from Python adapter\n')
            # write baseMVA
            f.write("mpc.baseMVA = %g;\n" % float(case.get('baseMVA', 100)))
            # write bus matrix
            f.write('mpc.bus = [\n')
            for row in case.get('bus', []):
                # ensure numeric formatting
                f.write('  ' + ' '.join(str(float(x)) for x in row) + ';\n')
            f.write('];\n')
            # gen
            f.write('mpc.gen = [\n')
            for row in case.get('gen', []):
                f.write('  ' + ' '.join(str(float(x)) for x in row) + ';\n')
            f.write('];\n')
            # branch
            f.write('mpc.branch = [\n')
            for row in case.get('branch', []):
                f.write('  ' + ' '.join(str(float(x)) for x in row) + ';\n')
            f.write('];\n')
            f.write('end\n')

        # In MATLAB: add tmpdir to path, call runpf on case_name
        eng.addpath(tmpdir, nargout=0)
        # runpf returns [baseMVA, bus, gen, branch, success, et] when called as struct? Use runpf(mpc)
        # We'll call: results = runpf(%s) and then save its workspace variable 'results' to a .mat
        try:
            # Try calling runpf with a case struct by name
            res = eng.eval('results = runpf(%s); results' % case_name, nargout=1)
        except Exception as e:
            # Try call as runpf(%s())
            res = eng.eval('results = runpf(%s()); results' % case_name, nargout=1)

        # Convert MATLAB struct to JSON-compatible dict by using MATLAB's jsonencode
        eng.eval("json = jsonencode(results);", nargout=0)
        json_str = eng.eval('json', nargout=1)
        py = json.loads(json_str)

        # Extract voltages if present
        # MATPOWER's results struct often contains 'bus', 'branch', 'gen' and 'success'.
        V = None
        if 'bus' in py:
            # bus matrix: column 8 is Vm, 9 is Va (degrees) in MATPOWER; but indexing may vary
            try:
                bus = py['bus']
                Vm = [float(r[7]) for r in bus]
                Va = [float(r[8]) for r in bus]
                V = np.array([vm * np.exp(1j * np.deg2rad(va)) for vm, va in zip(Vm, Va)])
            except Exception:
                V = None

        out = {'raw': py, 'V': V}
        return out
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


def run_matpower_pf(case: Dict[str, Any], matpower_path: str | None = None) -> Dict[str, Any]:
    """High-level runner: prefer MATLAB Engine when available."""
    if matlab_engine_available():
        return run_matpower_pf_with_engine(case, matpower_path=matpower_path)
    raise RuntimeError('MATLAB Engine not available in this Python environment.')
