EE521 Power Flow Dashboard

This folder contains a self-contained copy of the Power Flow Dashboard used in the project. It includes the Streamlit dashboard, the powerflow solver, a MATPOWER-case parser, diagnostic scripts and an example case. Use this to push to GitHub or to run locally.

EE521 Power Flow Dashboard

This folder contains a self-contained copy of the Power Flow Dashboard used in the project. It includes the Streamlit dashboard, the powerflow solver, a MATPOWER-case parser, diagnostic scripts and an example case. Use this to push to GitHub or to run locally.

Files of interest
- `contingency_dashboard.py` — Streamlit dashboard UI (entry point used for contingency analysis).
- `powerflow.py` — Newton–Raphson polar solver, Ybus builder, analytic Jacobian.
- `matpower_parser.py` — Helper to load .json or .py MATPOWER cases.
- `smoke_test.py` — Simple harness to run the solver on a case and print diagnostics.
- `diag_pf.py` — Diagnostic script used during debugging (Jacobian, residual checks).
- `requirements.txt` — Minimal dependencies required to run the dashboard.
- `cases/case9.json` — Example MATPOWER-style case (IEEE 9-bus).

Quick start (Windows PowerShell)

1) Create a venv and install requirements (if you didn't already):

```powershell
cd C:\\lu_dashboard\\EE521_PF_Dashboard
python -m venv .venv
.\.venv\\Scripts\\pip.exe install -r requirements.txt
```

2) Start the dashboard:

```powershell
.\.venv\\Scripts\\python.exe -m streamlit run "C:\\lu_dashboard\\EE521_PF_Dashboard\\contingency_dashboard.py" --server.port 8501
```

3) Open http://localhost:8501 in your browser.

If you run into missing-package errors, install them in the venv, for example:

```powershell
.\.venv\\Scripts\\pip.exe install plotly
```

Notes
- PyPower is the preferred engine for MATPOWER parity. Install it with `python -m pip install pypower`.
- If you have MATLAB + MATPOWER and the MATLAB Engine for Python installed, the app can attempt to use MATPOWER via `matpower_adapter.py`.
- Parallel N-1 sweeps are implemented, but external engines (MATLAB Engine) may not be multiprocessing-safe.

Developer: test & validation

- Run a quick smoke runner to execute the N-1 sweeps on `cases/case9.json`:

```powershell
python "tools\\run_n1_smoke.py"
```

- Unit tests are under `tests/`. Run them with:

```powershell
python -m pytest -q
```

If any tests fail or the Streamlit app reports errors, open an issue or contact the maintainer.
