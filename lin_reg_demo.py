import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------
# Config & data
# ---------------------------
st.set_page_config(page_title="Linear Regression Demo", layout="wide")

@st.cache_data
def make_data(n_points: int = 40, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n_points)
    m_true, b_true = 1.2, -2.0
    noise_sigma = 1.0
    y = m_true * x + b_true + rng.normal(0, noise_sigma, size=x.shape)
    return x, y, noise_sigma

x, y, sigma = make_data()

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.markdown("## Controls")

# Sliders use session_state keys so we can update after GD
m = st.sidebar.slider("Slope (m)", min_value=-5.0, max_value=5.0, value=st.session_state.get("m", 1.0), step=0.05, key="m")
b = st.sidebar.slider("Intercept (b)", min_value=-10.0, max_value=10.0, value=st.session_state.get("b", 0.0), step=0.1, key="b")

# Gradient descent controls
lr = st.sidebar.slider("Learning rate (η)", min_value=0.0001, max_value=0.2, value=0.01, step=0.0001, format="%.4f")
max_iter = st.sidebar.slider("Max iterations", min_value=1, max_value=30, value=20, step=1)
delay = st.sidebar.slider("Delay per step (s)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

# Grid for chi^2 surface
res = st.sidebar.selectbox("χ² grid resolution", options=[51, 81, 121], index=1, help="Higher = smoother, slower")
m_min, m_max = -5.0, 5.0
b_min, b_max = -10.0, 10.0

run_gd = st.sidebar.button("Run gradient descent", use_container_width=True)
reset_path = st.sidebar.button("Reset GD path", use_container_width=True)

# ---------------------------
# Session state for GD path
# ---------------------------
# Each entry: (m, b, chi2)
if "gd_path" not in st.session_state:
    st.session_state.gd_path = []
if reset_path:
    st.session_state.gd_path = []

# ---------------------------
# Computations
# ---------------------------

def chi2_and_grad(mv: float, bv: float):
    r = y - (mv * x + bv)
    chi2_val = float(np.sum((r / sigma) ** 2))
    # gradients
    dchi_dm = float(-2.0 * np.sum(x * r / (sigma ** 2)))
    dchi_db = float(-2.0 * np.sum(r / (sigma ** 2)))
    return chi2_val, dchi_dm, dchi_db

# Prepare chi^2 grid
M = np.linspace(m_min, m_max, int(res))
B = np.linspace(b_min, b_max, int(res))
MM, BB = np.meshgrid(M, B)
x3 = x[None, None, :]
y3 = y[None, None, :]
CHI2 = np.sum(((y3 - (MM[..., None] * x3 + BB[..., None])) / sigma) ** 2, axis=-1)

min_idx = np.unravel_index(np.argmin(CHI2), CHI2.shape)
m_min_chi2 = float(MM[min_idx])
b_min_chi2 = float(BB[min_idx])
chi2_min = float(CHI2[min_idx])

# ---------------------------
# Figure builders
# ---------------------------

def build_figures(mv: float, bv: float, path):
    # Left: data + regression line
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data",
                                     hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>"))

    x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
    y_line = mv * x_line + bv
    chi2_val, _, _ = chi2_and_grad(mv, bv)

    fig_scatter.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="model y=mx+b", hoverinfo="skip"))

    fig_scatter.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="x",
        yaxis_title="y",
        title=f"Data & Model — m={mv:.3f}, b={bv:.3f},  χ²={chi2_val:.2f}"
    )

    # Right: chi^2 surface (contour) with current (m,b) and GD path
    fig_chi2 = go.Figure()
    fig_chi2.add_trace(
        go.Contour(
            x=M,
            y=B,
            z=CHI2,
            contours=dict(showlabels=True, labelfont=dict(size=10)),
            colorbar=dict(title="χ²"),
            hovertemplate="m=%{x:.2f}<br>b=%{y:.2f}<br>χ²=%{z:.2f}<extra></extra>",
            showscale=True,
        )
    )

    # Minimum marker
    fig_chi2.add_trace(
        go.Scatter(
            x=[m_min_chi2], y=[b_min_chi2], mode="markers+text", name="χ² min",
            text=["min"], textposition="top center"
        )
    )

    # GD path (if any)
    if path:
        mxs = [p[0] for p in path]
        bys = [p[1] for p in path]
        steps_txt = [str(i) for i in range(len(path))]
        fig_chi2.add_trace(
            go.Scatter(
                x=mxs, y=bys, mode="lines+markers+text", name="GD steps",
                text=steps_txt, textposition="top center"
            )
        )

    # Current parameter marker (large X)
    fig_chi2.add_trace(
        go.Scatter(
            x=[mv], y=[bv], mode="markers", name="current (m,b)",
            marker=dict(size=14, symbol="x"),
            hovertemplate="m=%{x:.3f}<br>b=%{y:.3f}<extra>current</extra>"
        )
    )

    fig_chi2.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="slope m",
        yaxis_title="intercept b",
        title=f"χ² Surface (min at m={m_min_chi2:.3f}, b={b_min_chi2:.3f}, χ²_min={chi2_min:.2f})"
    )

    return fig_scatter, fig_chi2

# ---------------------------
# Layout + initial render
# ---------------------------
col1, col2 = st.columns(2)
left_placeholder = col1.empty()
right_placeholder = col2.empty()

# Initial draw (with current sliders & existing path)
fig_left, fig_right = build_figures(m, b, st.session_state.gd_path)
left_placeholder.plotly_chart(fig_left, use_container_width=True)
right_placeholder.plotly_chart(fig_right, use_container_width=True)

# Explanatory math (LaTeX-rendered)
st.markdown("**Notes**")
st.latex(r"\chi^2(m,b) = \sum_i \left(\frac{y_i - (m x_i + b)}{\sigma}\right)^2")
st.markdown('''- We use a fixed noise level σ = 1.
- The ‘χ² min’ point is the grid minimum (a numerical approximation to the analytic least-squares solution).
- Use the sliders to set the initial (m,b). Press **Run gradient descent** to watch the steps on the χ² surface.''')

# ---------------------------
# Gradient descent loop (animated with delay)
# ---------------------------
if run_gd:
    # Start fresh from the slider-selected (m,b)
    path = []
    mv, bv = float(m), float(b)
    for it in range(max_iter):
        chi2_val, dchi_dm, dchi_db = chi2_and_grad(mv, bv)
        path.append((mv, bv, chi2_val))
        # Parameter update (gradient descent)
        mv = mv - lr * dchi_dm
        bv = bv - lr * dchi_db
        # Keep the raw GD trajectory so the plotted path reflects the true updates.

        # redraw with the current step
        fig_left, fig_right = build_figures(mv, bv, path)
        left_placeholder.plotly_chart(fig_left, use_container_width=True)
        right_placeholder.plotly_chart(fig_right, use_container_width=True)
        time.sleep(delay)

    # final append and persist path
    chi2_val, _, _ = chi2_and_grad(mv, bv)
    path.append((mv, bv, chi2_val))
    st.session_state.gd_path = path

    # update sliders to final params
    st.session_state.m = mv
    st.session_state.b = bv

    # final redraw after state update
    fig_left, fig_right = build_figures(mv, bv, path)
    left_placeholder.plotly_chart(fig_left, use_container_width=True)
    right_placeholder.plotly_chart(fig_right, use_container_width=True)
