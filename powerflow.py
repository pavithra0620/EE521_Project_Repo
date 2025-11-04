import numpy as np
from typing import Dict, Any, Tuple, List


def build_ybus(mpc: Dict[str, Any]) -> np.ndarray:
	"""Build nodal admittance matrix Ybus (nb x nb) from MATPOWER-style mpc dict.

	Expects mpc['bus'], mpc['branch']. Fields are read permissively to handle
	list-of-lists MATPOWER-style cases used in the project.
	"""
	bus = np.array(mpc['bus'])
	branch = np.array(mpc['branch']) if 'branch' in mpc else np.empty((0, 13))
	nb = bus.shape[0]
	Y = np.zeros((nb, nb), dtype=complex)
	# branches
	for br in branch:
		if len(br) < 4:
			continue
		f = int(br[0]) - 1
		t = int(br[1]) - 1
		r = float(br[2])
		x = float(br[3])
		b = float(br[4]) if len(br) > 4 else 0.0
		# tap at index 8 in MATPOWER (1-based) -> python idx 8
		tap = float(br[8]) if len(br) > 8 and br[8] not in (0, None, '') else 1.0
		status = int(br[10]) if len(br) > 10 else 1
		if status == 0:
			continue
		z = r + 1j * x
		if z == 0:
			y = 0
		else:
			y = 1 / z
		ysh = 1j * b / 2
		# off-diagonals
		Y[f, t] -= y / tap
		Y[t, f] -= y / np.conjugate(tap)
		# diagonals
		Y[f, f] += (y + ysh) / (tap * np.conjugate(tap))
		Y[t, t] += y + ysh
	# bus shunts (Gs, Bs) typical columns 4,5 (0-based: 4->Gs index 4?)
	for i in range(nb):
		b = bus[i]
		Gs = float(b[4]) if len(b) > 4 else 0.0
		Bs = float(b[5]) if len(b) > 5 else 0.0
		Y[i, i] += Gs + 1j * Bs
	return Y


def get_bus_types(mpc: Dict[str, Any]) -> Dict[int, str]:
	"""Return mapping index->type string: 1->PQ,2->PV,3->Slack.

	Uses MATPOWER bus type codes.
	"""
	bus = mpc['bus']
	type_map = {1: 'PQ', 2: 'PV', 3: 'Slack'}
	out = {}
	for i, b in enumerate(bus):
		code = int(b[1]) if len(b) > 1 else 1
		out[i] = type_map.get(code, 'PQ')
	return out


def _net_injections(mpc: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute P_spec and Q_spec per bus (pu, using baseMVA).

	P_spec = Pg - Pd, Q_spec = Qg - Qd
	"""
	base = float(mpc.get('baseMVA', 100.0))
	nb = len(mpc['bus'])
	P = np.zeros(nb)
	Q = np.zeros(nb)
	# loads: PD col idx 2, QD idx 3 (1-based MATPOWER)
	for i, b in enumerate(mpc['bus']):
		Pd = float(b[2]) if len(b) > 2 else 0.0
		Qd = float(b[3]) if len(b) > 3 else 0.0
		P[i] -= Pd / base
		Q[i] -= Qd / base
	# gens: bus, Pg, Qg typically at idx 0,1,2
	for g in mpc.get('gen', []):
		bi = int(g[0]) - 1
		Pg = float(g[1]) if len(g) > 1 else 0.0
		Qg = float(g[2]) if len(g) > 2 else 0.0
		P[bi] += Pg / base
		Q[bi] += Qg / base
	return P, Q


def _build_jacobian(V: np.ndarray, Y: np.ndarray, slack: int, pq: List[int]) -> np.ndarray:
	"""Assemble the reduced Jacobian in polar coordinates.

	Unknown ordering: [dTheta for non-slack buses..., dV for PQ buses...]
	Returns dense numpy array.
	"""
	nb = len(V)
	G = Y.real
	B = Y.imag
	Vm = np.abs(V)
	Va = np.angle(V)
	idx_ns = [i for i in range(nb) if i != slack]
	ntheta = len(idx_ns)
	npq = len(pq)
	# allocate
	H = np.zeros((ntheta, ntheta))
	N = np.zeros((ntheta, npq)) if npq > 0 else np.zeros((ntheta, 0))
	M = np.zeros((npq, ntheta)) if npq > 0 else np.zeros((0, ntheta))
	L = np.zeros((npq, npq)) if npq > 0 else np.zeros((0, 0))
	# helper idx maps
	idx_ns_map = {bus: k for k, bus in enumerate(idx_ns)}
	pq_map = {bus: k for k, bus in enumerate(pq)}
	# fill off-diagonals
	for i_idx, i in enumerate(idx_ns):
		for j_idx, j in enumerate(idx_ns):
			if i == j:
				continue
			angle = Va[i] - Va[j]
			H[i_idx, j_idx] = Vm[i] * Vm[j] * (G[i, j] * np.sin(angle) - B[i, j] * np.cos(angle))
	for i_idx, i in enumerate(idx_ns):
		for k_idx, k in enumerate(pq):
			if i == k:
				continue
			angle = Va[i] - Va[k]
			N[i_idx, k_idx] = Vm[i] * (G[i, k] * np.cos(angle) + B[i, k] * np.sin(angle))
	for i_idx, i in enumerate(pq):
		for j_idx, j in enumerate(idx_ns):
			if i == j:
				continue
			angle = Va[i] - Va[j]
			M[i_idx, j_idx] = -Vm[i] * Vm[j] * (G[i, j] * np.cos(angle) + B[i, j] * np.sin(angle))
	for i_idx, i in enumerate(pq):
		for k_idx, k in enumerate(pq):
			if i == k:
				continue
			angle = Va[i] - Va[k]
			L[i_idx, k_idx] = Vm[i] * (G[i, k] * np.sin(angle) - B[i, k] * np.cos(angle))
	# diagonals
	# H diag
	for i_idx, i in enumerate(idx_ns):
		s = 0.0
		for j in range(nb):
			if j == i:
				continue
			angle = Va[i] - Va[j]
			s += Vm[i] * Vm[j] * (G[i, j] * np.sin(angle) - B[i, j] * np.cos(angle))
		H[i_idx, idx_ns_map[i]] = -s
	# N diag
	for i_idx, i in enumerate(idx_ns):
		s = 0.0
		for j in range(nb):
			if j == i:
				continue
			angle = Va[i] - Va[j]
			s += Vm[j] * (G[i, j] * np.cos(angle) + B[i, j] * np.sin(angle))
		if i in pq_map:
			N[i_idx, pq_map[i]] = 2 * G[i, i] * Vm[i] + s
	# M diag
	for i_idx, i in enumerate(pq):
		s = 0.0
		for j in range(nb):
			if j == i:
				continue
			angle = Va[i] - Va[j]
			s += Vm[i] * Vm[j] * (G[i, j] * np.cos(angle) + B[i, j] * np.sin(angle))
		M[i_idx, idx_ns_map[i]] = -s
	# L diag
	for i_idx, i in enumerate(pq):
		s = 0.0
		for j in range(nb):
			if j == i:
				continue
			angle = Va[i] - Va[j]
			s += Vm[j] * (G[i, j] * np.sin(angle) - B[i, j] * np.cos(angle))
		L[i_idx, pq_map[i]] = -2 * B[i, i] * Vm[i] + s
	# assemble
	if npq > 0:
		top = np.hstack((H, N))
		bottom = np.hstack((M, L))
		J = np.vstack((top, bottom))
	else:
		J = H
	return J


def newton_raphson(mpc: Dict[str, Any], tol: float = 1e-8, maxiter: int = 50, damping: float = 1.0) -> Dict[str, Any]:
	"""Solve power flow with Newton-Raphson (polar formulation).

	Returns dict with keys: V, Ybus, converged, iterations, history, P_mismatch, Q_mismatch,
	mismatch (reduced), Jacobian (at final point)
	"""
	bus = mpc['bus']
	nb = len(bus)
	Y = build_ybus(mpc)
	P_spec, Q_spec = _net_injections(mpc)
	bus_types = get_bus_types(mpc)
	slack_idx = [i for i, t in bus_types.items() if t == 'Slack']
	if len(slack_idx) == 0:
		raise ValueError('No slack bus found')
	slack = slack_idx[0]
	pv = [i for i, t in bus_types.items() if t == 'PV']
	pq = [i for i, t in bus_types.items() if t == 'PQ']
	# initial voltage
	V = np.zeros(nb, dtype=complex)
	for i, b in enumerate(bus):
		Vm = float(b[7]) if len(b) > 7 else 1.0
		Va = float(b[8]) if len(b) > 8 else 0.0
		V[i] = Vm * np.exp(1j * np.deg2rad(Va))
	idx_ns = [i for i in range(nb) if i != slack]
	ntheta = len(idx_ns)
	npq = len(pq)
	history = []
	converged = False
	for it in range(maxiter):
		S = V * np.conj(Y @ V)
		Pcalc = S.real
		Qcalc = S.imag
		# mismatches f = P_spec - Pcalc, Q_spec - Qcalc
		fP = P_spec - Pcalc
		fQ = Q_spec - Qcalc
		fP_ns = fP[idx_ns]
		fQ_pq = fQ[pq] if npq > 0 else np.array([])
		f = np.hstack((fP_ns, fQ_pq)) if npq > 0 else fP_ns
		norm = np.linalg.norm(f, ord=np.inf)
		history.append(norm)
		if norm < tol:
			converged = True
			break
		J = _build_jacobian(V, Y, slack, pq)
		# solve J dx = -f
		try:
			# Newton update: J_calc * dx = f  where f = P_spec - Pcalc, Q_spec - Qcalc
			dx = np.linalg.solve(J, f)
		except np.linalg.LinAlgError:
			# singular Jacobian
			break
		# apply damping/line-search: attempt full step, if nonlinear residual increases,
		# back off by halving until improvement or minimum step reached.
		dx_full = damping * dx
		alpha = 1.0
		V_curr = V.copy()
		S_curr = V_curr * np.conj(Y @ V_curr)
		Pcalc_curr = S_curr.real
		Qcalc_curr = S_curr.imag
		fP_curr = P_spec - Pcalc_curr
		fQ_curr = Q_spec - Qcalc_curr
		f_curr = np.hstack((fP_curr[idx_ns], fQ_curr[pq])) if npq > 0 else fP_curr[idx_ns]
		norm_curr = np.linalg.norm(f_curr, ord=np.inf)
		accepted = False
		for _ in range(20):
			dx_try = alpha * dx_full
			dtheta = dx_try[0:ntheta]
			dV = dx_try[ntheta:] if npq > 0 else np.array([])
			theta = np.angle(V_curr)
			for k, i in enumerate(idx_ns):
				theta[i] += dtheta[k]
			Vm_try = np.abs(V_curr).copy()
			for k, i in enumerate(pq):
				Vm_try[i] += dV[k]
			V_try = np.array([Vm_try[i] * np.exp(1j * theta[i]) for i in range(nb)])
			S_try = V_try * np.conj(Y @ V_try)
			Pcalc_try = S_try.real
			Qcalc_try = S_try.imag
			fP_try = P_spec - Pcalc_try
			fQ_try = Q_spec - Qcalc_try
			f_try = np.hstack((fP_try[idx_ns], fQ_try[pq])) if npq > 0 else fP_try[idx_ns]
			norm_try = np.linalg.norm(f_try, ord=np.inf)
			if norm_try < norm_curr or alpha < 1e-4:
				# accept
				V = V_try
				accepted = True
				break
			alpha *= 0.5
		if not accepted:
			# if no improvement, keep current V (abort)
			V = V_curr
	# final Jacobian
	Jfinal = _build_jacobian(V, Y, slack, pq)
	# full mismatches per bus
	S = V * np.conj(Y @ V)
	Pcalc = S.real
	Qcalc = S.imag
	Pm = P_spec - Pcalc
	Qm = Q_spec - Qcalc
	result = {
		'V': V,
		'Ybus': Y,
		'converged': bool(converged),
		'iterations': int(it + 1),
		'history': history,
		'P_mismatch': Pm,
		'Q_mismatch': Qm,
		'mismatch': f,
		'Jacobian': Jfinal,
	}
	return result


def NR_polar_Jacobian_Solver(mpc: Dict[str, Any], tol: float = 1e-8, maxiter: int = 50, damping: float = 1.0) -> Dict[str, Any]:
	"""Alias for newton_raphson kept for compatibility with older code/tests."""
	return newton_raphson(mpc, tol=tol, maxiter=maxiter, damping=damping)
