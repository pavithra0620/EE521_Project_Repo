import json, os, sys
sys.path.insert(0, r'C:\lu_dashboard\EE521_PF_Dashboard')
from powerflow import build_ybus, _build_jacobian, newton_raphson
import numpy as np

case_path = r'C:\lu_dashboard\EE521_PF_Dashboard\cases\case9.json'
if not os.path.exists(case_path):
    print('case9 not found')
    sys.exit(1)
with open(case_path,'r',encoding='utf-8') as f:
    mpc = json.load(f)
# convert dict to list of lists if necessary (simple conversion)
if isinstance(mpc.get('bus',[])[0], dict):
    bus_out = []
    for b in mpc['bus']:
        bus_out.append([b.get('bus_i',0), b.get('type',1), b.get('Pd',0), b.get('Qd',0), b.get('Gs',0), b.get('Bs',0), 1, b.get('Vm',1.0), b.get('Va',0.0), b.get('baseKV',230), 1, 1.06, 0.94])
    gen_out = []
    for g in mpc.get('gen',[]):
        gen_out.append([g.get('bus',1), g.get('Pg',0), g.get('Qg',0), g.get('Qmax',9999), g.get('Qmin',-9999), g.get('Vg',1.0), mpc.get('baseMVA',100), 1, g.get('Pmax',9999), g.get('Pmin',-9999)])
    br_out = []
    for br in mpc.get('branch',[]):
        br_out.append([br.get('fbus'), br.get('tbus'), br.get('r',0.0), br.get('x',0.0), br.get('b',0.0), br.get('rateA',0), br.get('rateB',0), br.get('rateC',0), br.get('ratio',0), br.get('angle',0), br.get('status',1), br.get('angmin',-360), br.get('angmax',360)])
    mpc2 = {'baseMVA': mpc.get('baseMVA',100), 'bus': bus_out, 'gen': gen_out, 'branch': br_out}
else:
    mpc2 = mpc

# initial V like in newton_raphson
bus = mpc2['bus']
V = np.array([complex(b[7] * np.cos(np.deg2rad(b[8])), b[7] * np.sin(np.deg2rad(b[8]))) for b in bus])
Y = build_ybus(mpc2)
# compute P_spec, Q_spec
P_spec = np.zeros(len(bus))
Q_spec = np.zeros(len(bus))
for i,b in enumerate(bus):
    Pd = float(b[2]); Qd = float(b[3])
    P_spec[i] -= Pd / mpc2.get('baseMVA',100)
    Q_spec[i] -= Qd / mpc2.get('baseMVA',100)
for g in mpc2.get('gen',[]):
    bi = int(g[0]) - 1
    P_spec[bi] += float(g[1]) / mpc2.get('baseMVA',100)
    Q_spec[bi] += float(g[2]) / mpc2.get('baseMVA',100)

S = V * np.conj(Y @ V)
Pcalc = S.real; Qcalc = S.imag
Pm = P_spec - Pcalc
Qm = Q_spec - Qcalc
print('Initial max |Pm|', np.max(np.abs(Pm)))
print('Initial max |Qm|', np.max(np.abs(Qm)))

# compute Jacobian
from powerflow import _build_jacobian
bus_types = {int(b[0]) - 1: ('Slack' if int(b[1])==3 else ('PV' if int(b[1])==2 else 'PQ')) for b in bus}
pq = [i for i,t in bus_types.items() if t=='PQ']
slack = [i for i,t in bus_types.items() if t=='Slack'][0]
J = _build_jacobian(V, Y, slack, pq)
print('Jacobian shape:', J.shape)
# show condition number
try:
    cond = np.linalg.cond(J)
    print('Jacobian cond:', cond)
except Exception as e:
    print('Could not compute cond:', e)
print('Some Jacobian entries (top-left 6x6):')
print(J[:6,:6])

# try one Newton step
import numpy.linalg as la
mis = np.hstack((Pm[[i for i in range(len(bus)) if i != slack]], Qm[pq] if len(pq)>0 else np.array([])))
print('Mismatch vector norm:', np.linalg.norm(mis, ord=np.inf))
try:
    dx = la.solve(J, -mis)
    print('dx norm', np.linalg.norm(dx))
    print('dx vector (first 12 entries):', dx[:12])
    # check linear residual: J dx + mis should be ~0
    res = J.dot(dx) + mis
    print('linear residual norm |J dx + mis|:', np.linalg.norm(res))
    # apply one Newton step to V and report new mismatch norm
    # build updated V (angles and PQ voltages)
    theta = np.angle(V)
    idx_ns = [i for i in range(len(bus)) if i != slack]
    for k, i in enumerate(idx_ns):
        theta[i] += dx[k]
    Vm = np.abs(V).copy()
    for k, i in enumerate(pq):
        Vm[i] += dx[len(idx_ns) + k]
    V1 = np.array([Vm[i] * np.exp(1j * theta[i]) for i in range(len(bus))])
    print('\nVm before:', np.abs(V))
    print('Vm after :', Vm)
    S1 = V1 * np.conj(Y @ V1)
    Pcalc1 = S1.real; Qcalc1 = S1.imag
    Pm1 = P_spec - Pcalc1
    Qm1 = Q_spec - Qcalc1
    mis1 = np.hstack((Pm1[[i for i in range(len(bus)) if i != slack]], Qm1[pq] if len(pq)>0 else np.array([])))
    print('After one Newton step mismatch norm:', np.linalg.norm(mis1, ord=np.inf))
    print('\nPer-bus Pcalc before -> after:')
    for i in range(len(bus)):
        print(f'bus {i+1}: Pcalc {Pcalc[i]:.6f} -> {Pcalc1[i]:.6f}, P_spec {P_spec[i]:.6f}')
    print('\nPer-bus Qcalc before -> after:')
    for i in range(len(bus)):
        print(f'bus {i+1}: Qcalc {Qcalc[i]:.6f} -> {Qcalc1[i]:.6f}, Q_spec {Q_spec[i]:.6f}')
except Exception as e:
    print('Solve failed:', e)

# run full NR
res = newton_raphson(mpc2, tol=1e-6, maxiter=50)
print('After NR - converged', res['converged'], 'iters', res['iterations'])
print('Final max |P_mis|', max(abs(x) for x in res['P_mismatch']))
print('Final max |Q_mis|', max(abs(x) for x in res['Q_mismatch']))
# try with damping alpha=0.5
res2 = newton_raphson(mpc2, tol=1e-6, maxiter=100, damping=0.5)
print('After NR with damping 0.5 - converged', res2['converged'], 'iters', res2['iterations'])
print('Final max |P_mis| (damped)', max(abs(x) for x in res2['P_mismatch']))
print('Final max |Q_mis| (damped)', max(abs(x) for x in res2['Q_mismatch']))
# sweep damping values to see stability
for alpha in [0.4, 0.3, 0.2, 0.1]:
    r = newton_raphson(mpc2, tol=1e-8, maxiter=200, damping=alpha)
    print(f'alpha={alpha} -> converged={r["converged"]} iters={r["iterations"]} maxP={max(abs(x) for x in r["P_mismatch"]):.6g} maxQ={max(abs(x) for x in r["Q_mismatch"]):.6g}')