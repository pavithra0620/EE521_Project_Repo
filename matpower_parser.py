import json
import importlib.util
from typing import Dict, Any


def load_case(path: str) -> Dict[str, Any]:
    """Load a MATPOWER-style case from JSON (.json) or Python file (.py).

    The expected structure matches MATPOWER mpc: keys baseMVA, bus, gen, branch.
    """
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    elif path.endswith('.py'):
        spec = importlib.util.spec_from_file_location('mpc_module', path)
        mpc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mpc)
        # look for common names
        for name in ('mpc', 'case', 'get_mpc'):
            if hasattr(mpc, name):
                obj = getattr(mpc, name)
                # if callable, call it
                if callable(obj):
                    return obj()
                else:
                    return obj
        raise ValueError('No mpc-like object found in python file')
    else:
        raise ValueError('Unsupported case format. Provide .json or .py file')
