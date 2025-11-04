import json, os, sys
sys.path.insert(0, r'C:\lu_dashboard\EE521_PF_Dashboard')
from powerflow import newton_raphson

candidates = [r'C:\lu_dashboard\EE521_PF_Dashboard\cases\case9.json', r'C:\lu_dashboard\case.json', r'C:\lu_dashboard\mp_test.json']
case_path = None
for p in candidates:
    if os.path.exists(p):
        case_path = p
        break
if not case_path:
    print('No case file found')
    sys.exit(1)
with open(case_path,'rb') as f:
    data = f.read()
    # detect BOM/encoding common in Windows-saved JSONs
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        text = data.decode('utf-16')
    else:
        try:
            text = data.decode('utf-8')
        except Exception:
            text = data.decode('latin-1')
    mpc = json.loads(text)
# Convert dict-style MATPOWER JSON to list-of-lists format expected by local powerflow
def _to_list_case(mpc_in):
    m = dict(mpc_in)
    # bus
    bus_in = m.get('bus', [])
    bus_out = []
    if len(bus_in) > 0 and isinstance(bus_in[0], dict):
        for b in bus_in:
            bus_i = int(b.get('bus_i') or b.get('bus') or b.get('bus_i', 0))
            btype = int(b.get('type', 1))
            Pd = float(b.get('Pd', 0.0))
            Qd = float(b.get('Qd', 0.0))
            Gs = float(b.get('Gs', 0.0))
            Bs = float(b.get('Bs', 0.0))
            area = int(b.get('area', 1))
            Vm = float(b.get('Vm', 1.0))
            Va = float(b.get('Va', 0.0))
            baseKV = float(b.get('baseKV', 230.0))
            zone = int(b.get('zone', 1))
            Vmax = float(b.get('Vmax', 1.06))
            Vmin = float(b.get('Vmin', 0.94))
            bus_out.append([bus_i, btype, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin])
    else:
        bus_out = bus_in
    # gen
    gen_in = m.get('gen', [])
    gen_out = []
    if len(gen_in) > 0 and isinstance(gen_in[0], dict):
        for g in gen_in:
            bus_i = int(g.get('bus'))
            Pg = float(g.get('Pg', g.get('P', 0.0)))
            Qg = float(g.get('Qg', g.get('Q', 0.0)))
            Qmax = float(g.get('Qmax', 9999))
            Qmin = float(g.get('Qmin', -9999))
            Vg = float(g.get('Vg', 1.0))
            mbase = float(g.get('mbase', m.get('baseMVA', 100)))
            status = int(g.get('status', 1))
            Pmax = float(g.get('Pmax', 9999))
            Pmin = float(g.get('Pmin', -9999))
            gen_out.append([bus_i, Pg, Qg, Qmax, Qmin, Vg, mbase, status, Pmax, Pmin])
    else:
        gen_out = gen_in
    # branch
    br_in = m.get('branch', [])
    br_out = []
    if len(br_in) > 0 and isinstance(br_in[0], dict):
        for br in br_in:
            f = int(br.get('fbus', br.get('from'))) 
            t = int(br.get('tbus', br.get('to')))
            r = float(br.get('r', 0.0))
            x = float(br.get('x', 0.0))
            b = float(br.get('b', 0.0))
            rateA = float(br.get('rateA', 0.0))
            rateB = float(br.get('rateB', 0.0))
            rateC = float(br.get('rateC', 0.0))
            ratio = float(br.get('ratio', 0.0))
            angle = float(br.get('angle', 0.0))
            status = int(br.get('status', 1))
            angmin = float(br.get('angmin', -360))
            angmax = float(br.get('angmax', 360))
            br_out.append([f, t, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax])
    else:
        br_out = br_in
    out = {'baseMVA': m.get('baseMVA', 100), 'bus': bus_out, 'gen': gen_out, 'branch': br_out}
    return out

mpc = _to_list_case(mpc)
res = newton_raphson(mpc, tol=1e-6, maxiter=100)
print('Converged:', res.get('converged'))
print('Iterations:', res.get('iterations'))
Pmis = res.get('P_mismatch')
Qmis = res.get('Q_mismatch')
print('Max |P_mis| =', max(abs(x) for x in Pmis))
print('Max |Q_mis| =', max(abs(x) for x in Qmis))
print('Bus voltages (mag):')
for i,v in enumerate(res['V']):
    print(i+1, abs(v), angle := (v.real, v.imag))
