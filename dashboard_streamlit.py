import os
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matpower_parser import load_case
from powerflow import build_ybus, newton_raphson, get_bus_types


# helper to create small demo MATPOWER-like cases programmatically
def make_dummy_case(n):
    """Create a simple MATPOWER-style case dict with n buses connected in a chain."""
    baseMVA = 100
    bus = []
    gen = []
    branch = []
    for i in range(1, n+1):
        # [bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin]
        btype = 1 if i != 1 else 3  # bus 1 slack (3), others PQ (1)
        Pd = 0.0
        Qd = 0.0
        bus.append([i, btype, Pd, Qd, 0.0, 0.0, 1, 1.0, 0.0, 230, 1, 1.06, 0.94])
    # single generator at bus 1
    gen.append([1, 100.0, 0.0, 100.0, -100.0, 1.0, baseMVA, 1, 200.0, 0.0])
    # chain branches
    for i in range(1, n):
        # [fbus,tbus,r,x,b, rateA,rateB,rateC,ratio,angle,status,angmin,angmax]
        r = 0.01
        x = 0.03
        b = 0.0
        branch.append([i, i+1, r, x, b, 0, 0, 0, 0, 0, 1, -360, 360])
    return {'baseMVA': baseMVA, 'bus': bus, 'gen': gen, 'branch': branch}


st.set_page_config(layout='wide', page_title='Power Flow Dashboard')

# Pastel CSS theme (subtle) -----------------------------------------------
st.markdown("""
<style>
body {background: #fffaf6}
.stApp { background: linear-gradient(180deg,#fffaf6 0%, #fff 100%); }
.stButton>button { background-color: #ffccf9; color: #4b2e83; }
.big-title {font-size:34px; font-weight:700; color:#7b2cbf}
.card {background: #ffffffcc; border-radius:12px; padding:12px; box-shadow: 0 6px 18px rgba(123,44,191,0.06);} 
.muted {color: #6b7280}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Power Flow Dashboard</div>', unsafe_allow_html=True)
st.markdown('A small interactive UI to load MATPOWER-style cases, compute Ybus, Jacobian, mismatches and run Newton-Raphson power flow.')


st.sidebar.header('Load case')
use_example = st.sidebar.checkbox('Use example case (case9)', value=True)
uploaded = st.sidebar.file_uploader('Or upload case (.json or .py)', type=['json','py'])
preset = st.sidebar.selectbox('Or choose a built-in demo case', ['None','case3_lmbd','case14','case57','case145','case2383p'])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# select case: preset takes priority, then example, then uploaded
if preset and preset != 'None':
    sizes = {'case3_lmbd':3, 'case14':14, 'case57':57, 'case145':145, 'case2383p':2383}
    n = sizes.get(preset, 14)
    case = make_dummy_case(n)
elif use_example:
    case = load_case(os.path.join(BASE_DIR, 'cases', 'case9.json'))
elif uploaded is not None:
    # save uploaded file to a temp location
    upath = os.path.join(BASE_DIR, 'uploaded_case')
    with open(upath, 'wb') as f:
        f.write(uploaded.getbuffer())
    case = load_case(upath)
else:
    st.sidebar.info('Select example, preset, or upload a case file')
    st.stop()

st.sidebar.header('Display options')
ybus_format = st.sidebar.radio('Ybus form', ['Polar','Rectangular'])


# compute
Ybus = build_ybus(case)
pf = newton_raphson(case)


# Top summary metrics -----------------------------------------------------
col_a, col_b, col_c = st.columns([2,1,1])
with col_a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Case summary')
    st.write(f"Base MVA: **{case.get('baseMVA', 100)}**")
    st.write(f"Buses: **{len(case['bus'])}**  •  Branches: **{len(case['branch'])}**")
    st.markdown('</div>', unsafe_allow_html=True)
with col_b:
    st.metric('Converged', '✅' if pf['converged'] else '❌', delta=f"{pf['iterations']} it")
with col_c:
    st.metric('Last mismatch (inf-norm)', f"{(pf['history'][-1] if len(pf['history'])>0 else None):.2e}" if pf.get('history') else 'n/a')


## Admittance matrix -----------------------------------------------------
st.markdown('---')
st.subheader('Admittance Matrix (Ybus)')
if ybus_format == 'Polar':
    mags = np.abs(Ybus)
    angs = np.angle(Ybus, deg=True)
    df = pd.DataFrame([[f"{mags[i,j]:.4f} ∠ {angs[i,j]:.2f}°" for j in range(Ybus.shape[1])] for i in range(Ybus.shape[0])])
else:
    df = pd.DataFrame([[f"{Ybus[i,j].real:.4f} + j{Ybus[i,j].imag:.4f}" for j in range(Ybus.shape[1])] for i in range(Ybus.shape[0])])
st.dataframe(df)

# provide download of Ybus as CSV (real,imag per cell)
try:
    import io
    # CSV: each row contains pairs re,im for each column
    buf = io.StringIO()
    for i in range(Ybus.shape[0]):
        row = []
        for j in range(Ybus.shape[1]):
            row.append(f"{Ybus[i,j].real:.8f}")
            row.append(f"{Ybus[i,j].imag:.8f}")
        buf.write(','.join(row) + '\n')
    st.download_button('Download Ybus (CSV)', data=buf.getvalue(), file_name='ybus.csv', mime='text/csv')
except Exception:
    pass


## Bus types --------------------------------------------------------------
st.markdown('---')
st.subheader('Bus Types')
bus_types = get_bus_types(case)
bus_df = pd.DataFrame([{'bus': i+1, 'type': t} for i, t in bus_types.items()])

# show cute badges for types
def type_emoji(t):
    return '🔌 Slack' if t=='Slack' else ('☀️ PV' if t=='PV' else '🔋 PQ')

bus_df['label'] = bus_df['type'].apply(type_emoji)
st.table(bus_df[['bus','label']].set_index('bus'))


## Power flow results ----------------------------------------------------
st.markdown('---')
st.subheader('Power Flow Results')
col1, col2 = st.columns([2,1])
with col1:
    st.write('Voltage solution')
    V = pf['V']
    results = pd.DataFrame([{ 'bus': i+1, 'V_mag': np.abs(V[i]), 'V_ang_deg': np.angle(V[i],deg=True)} for i in range(len(V))])
    st.table(results.set_index('bus'))
    # show convergence history
    if pf.get('history'):
        st.write('Convergence history (inf-norm of mismatch)')
        st.line_chart(pd.Series(pf['history']))
with col2:
    st.write('Mismatches')
    st.write('P mismatch (pu)')
    st.write(pd.Series(pf['P_mismatch']).round(6))
    if 'Q_mismatch' in pf:
        st.write('Q mismatch (pu)')
        st.write(pd.Series(pf['Q_mismatch']).round(6))

# show results CSV download
try:
    import io
    out_buf = io.StringIO()
    results.to_csv(out_buf, index=False)
    st.download_button('Download Results (CSV)', data=out_buf.getvalue(), file_name='results.csv', mime='text/csv')
except Exception:
    pass


## Jacobian ----------------------------------------------------------------
st.markdown('---')
st.subheader('Jacobian Matrix (reduced)')
J = pf.get('Jacobian')
nb = len(case['bus'])
# determine ordering for Jacobian rows/cols: theta (non-slack) then Vm (PQ buses)
bus_types = get_bus_types(case)
slack_idxs = [i for i,t in bus_types.items() if t == 'Slack']
slack_idx = slack_idxs[0] if len(slack_idxs)>0 else 0
pq = [i for i,t in bus_types.items() if t == 'PQ']
theta_order = [i for i in range(nb) if i != slack_idx]
V_order = pq
labels = [f'dθ_bus{(i+1)}' for i in theta_order] + [f'dV_bus{(i+1)}' for i in V_order]
if J is not None:
    try:
        Jdf = pd.DataFrame(J.round(6), index=labels, columns=labels)
        st.dataframe(Jdf)
        # download Jacobian CSV
        try:
            import io
            csv_buf = io.StringIO()
            Jdf.to_csv(csv_buf)
            st.download_button('Download Jacobian (CSV)', data=csv_buf.getvalue(), file_name='jacobian.csv', mime='text/csv')
        except Exception:
            pass
        # explain ordering
        with st.expander('Show Jacobian variable ordering'):
            st.write('Order of variables (rows and columns):')
            st.write(labels)
    except Exception:
        st.write('Jacobian matrix is large or could not be displayed')


## Network plot (cute) -----------------------------------------------------
st.markdown('---')
st.subheader('Network Visualization')
G = nx.Graph()
bus = case['bus']
for b in bus:
    G.add_node(int(b[0]))
for br in case['branch']:
    # handle branch status if present
    try:
        status = int(br[10]) if len(br) > 10 else 1
    except Exception:
        status = 1
    if status == 0:
        continue
    G.add_edge(int(br[0]), int(br[1]))
pos = nx.spring_layout(G, seed=42)

# prepare data
V = pf['V']
vmags = np.array([np.abs(V[i]) for i in range(len(V))])
nodes = list(G.nodes())

use_interactive = st.checkbox('Interactive visualization (hover tooltips)', value=True)
if use_interactive:
    # build Plotly traces for edges
    edge_x = []
    edge_y = []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#d3d3d3'), hoverinfo='none')

    node_x = []
    node_y = []
    hover_text = []
    sizes = []
    # scaling
    if len(vmags) > 0:
        cmin = float(vmags.min()); cmax = float(vmags.max())
    else:
        cmin = 1.0; cmax = 1.0
    for n in nodes:
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        idx = n-1
        Vm = float(np.abs(V[idx])) if 0 <= idx < len(V) else float('nan')
        Va = float(np.angle(V[idx], deg=True)) if 0 <= idx < len(V) else float('nan')
        try:
            Pd = case['bus'][idx][2]
            Qd = case['bus'][idx][3]
        except Exception:
            Pd = None; Qd = None
        btype = get_bus_types(case).get(idx, 'PQ')
        hover = f"Bus {n}<br>V = {Vm:.4f} pu ∠ {Va:.2f}°<br>Type: {btype}<br>Pd={Pd} Qd={Qd}"
        hover_text.append(hover)
        if cmax - cmin > 1e-12 and not np.isnan(Vm):
            size = 10 + 30 * ((Vm - cmin) / (cmax - cmin))
        else:
            size = 20
        sizes.append(size)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[str(n) for n in nodes],
        textposition='top center',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=vmags.tolist() if len(vmags)>0 else [1.0]*len(nodes),
            size=sizes,
            colorbar=dict(title='V (pu)'),
            line_width=1
        ),
        hovertext=hover_text
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))
    st.plotly_chart(fig, use_container_width=True)
else:
    # fallback static matplotlib
    fig, ax = plt.subplots(figsize=(8,5))
    node_colors = [vmags[n-1] if (n-1)>=0 and (n-1)<len(vmags) else 1.0 for n in nodes]
    if len(node_colors)>0:
        cmin = float(min(node_colors)); cmax = float(max(node_colors))
        if cmax-cmin > 1e-12:
            node_sizes = [300 + 1200*((c-cmin)/(cmax-cmin)) for c in node_colors]
        else:
            node_sizes = [600 for _ in node_colors]
    else:
        node_sizes = [600 for _ in nodes]
    nodes_draw = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, cmap=plt.cm.Pastel1, ax=ax, edgecolors='#7b2cbf')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#ffd6e8')
    nx.draw_networkx_labels(G, pos, ax=ax, font_color='#4b2e83')
    cb = fig.colorbar(nodes_draw, ax=ax)
    cb.set_label('Voltage magnitude (pu)')
    ax.set_axis_off()
    st.pyplot(fig)


st.markdown('---')
st.caption('Made with ♥︎ — you can download the CSVs above and run `streamlit run dashboard_streamlit.py` to host locally.')
