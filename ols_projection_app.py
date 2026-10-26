"""
KL6000 Data Science -- Supplementary interactive app: OLS as Orthogonal Projection

Companion to Lecture 3, Section 3.2 and Seminar 3, Part 2, and to the video script
`video_script_ols_projection.md`. Reuses the exact toy example from that script
(n = 3, invented purely to be drawable -- never Ames Housing or Palmer Penguins,
which are reserved for student lab exercises).

Core idea shown interactively: y-hat = X * beta is constrained to lie on the plane
C(X). Dragging beta moves the candidate y-hat across that plane; the squared
distance to y (the RSS) is minimised at exactly one point -- the OLS estimate --
and X^T e is the zero vector at that point and only at that point.

Run with:  streamlit run app.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="OLS as Orthogonal Projection", layout="wide")

# ----------------------------------------------------------------------------
# Fixed design matrix: matches the video script exactly.
# x = (1, 2, 3); intercept + one predictor; n = 3, p = 2 -> C(X) is a plane in R^3.
# ----------------------------------------------------------------------------
X = np.column_stack([np.ones(3), np.array([1.0, 2.0, 3.0])])  # columns: intercept, x
DEFAULT_Y = np.array([1.0, 3.0, 2.0])
SCRIPT_ALT_BETA = np.array([1.3, 0.3])  # the "nearby but wrong" beta used in the script


def ols_beta(y: np.ndarray) -> np.ndarray:
    """Closed-form OLS estimate for the fixed X above."""
    return np.linalg.solve(X.T @ X, X.T @ y)


# ----------------------------------------------------------------------------
# Session state: y (data point) and beta (candidate point), both user-movable.
# ----------------------------------------------------------------------------
for key, default in [("y1", DEFAULT_Y[0]), ("y2", DEFAULT_Y[1]), ("y3", DEFAULT_Y[2])]:
    st.session_state.setdefault(key, float(default))

_default_beta = ols_beta(DEFAULT_Y)
st.session_state.setdefault("beta0", float(_default_beta[0]))
st.session_state.setdefault("beta1", float(_default_beta[1]))


def current_y() -> np.ndarray:
    return np.array([st.session_state.y1, st.session_state.y2, st.session_state.y3])


def snap_to_ols():
    b = ols_beta(current_y())
    st.session_state.beta0 = float(b[0])
    st.session_state.beta1 = float(b[1])


def reset_everything():
    st.session_state.y1, st.session_state.y2, st.session_state.y3 = DEFAULT_Y.tolist()
    snap_to_ols()


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
st.title("OLS as Orthogonal Projection")
st.caption(
    "Toy example: n = 3, x = (1, 2, 3) -- invented purely to be drawable, "
    "not Ames Housing or Palmer Penguins. Matches the worked example in the "
    "video script."
)

controls_col, plot_col, metrics_col = st.columns([1.1, 2.4, 1.3])

with controls_col:
    st.subheader("Candidate fit")
    st.caption("Move β to slide the candidate point across the plane C(X).")
    st.slider("β₀ (intercept)", -3.0, 5.0, step=0.02, key="beta0")
    st.slider("β₁ (slope)", -3.0, 5.0, step=0.02, key="beta1")
    st.button("Snap β to the OLS estimate", on_click=snap_to_ols, width="stretch")

    show_alt = st.checkbox(
        "Show the script's alternative point (β = 1.3, 0.3)",
        value=False,
        help="The 'nearby but wrong' point compared against OLS in the video script.",
    )
    show_basis = st.checkbox("Show the basis vectors of X", value=False)

    with st.expander("Data point y (advanced)"):
        st.caption("Defaults match the video script (y = 1, 3, 2).")
        st.slider("y₁", -2.0, 6.0, step=0.1, key="y1")
        st.slider("y₂", -2.0, 6.0, step=0.1, key="y2")
        st.slider("y₃", -2.0, 6.0, step=0.1, key="y3")
        st.button("Reset y and β to the script example", on_click=reset_everything)

# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
y = current_y()
beta = np.array([st.session_state.beta0, st.session_state.beta1])
yhat = X @ beta
e = y - yhat
rss = float(e @ e)
Xte = X.T @ e

beta_star = ols_beta(y)
yhat_star = X @ beta_star
e_star = y - yhat_star
rss_star = float(e_star @ e_star)

is_orthogonal = np.allclose(Xte, 0.0, atol=1e-6)

# Orthonormal basis for C(X) -- used ONLY to draw a legible plane patch.
# The raw columns of X are ~22 degrees apart, so a patch built directly from
# them renders as a near-invisible sliver; an orthonormal basis spans the
# identical plane and does not change y, yhat, or e, all of which are
# basis-independent quantities.
Q, _ = np.linalg.qr(X)
q1, q2 = Q[:, 0], Q[:, 1]
centre = X.mean(axis=1)
a_vals = np.linspace(-3.5, 3.5, 14)
b_vals = np.linspace(-3.5, 3.5, 14)
A, B = np.meshgrid(a_vals, b_vals)
Px = centre[0] + A * q1[0] + B * q2[0]
Py = centre[1] + A * q1[1] + B * q2[1]
Pz = centre[2] + A * q1[2] + B * q2[2]

# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Surface(
    x=Px, y=Py, z=Pz, opacity=0.35, showscale=False,
    colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
    name="C(X)", hoverinfo="skip",
))

if show_basis:
    for col, label in [(X[:, 0], "intercept col (1,1,1)"), (X[:, 1], "x col (1,2,3)")]:
        fig.add_trace(go.Scatter3d(
            x=[0, col[0]], y=[0, col[1]], z=[0, col[2]],
            mode="lines", line=dict(color="steelblue", width=5),
            name=label,
        ))

# Data point y
fig.add_trace(go.Scatter3d(
    x=[y[0]], y=[y[1]], z=[y[2]], mode="markers",
    marker=dict(size=7, color="firebrick"), name="y (data)",
))

# Candidate y-hat(beta) and residual
fig.add_trace(go.Scatter3d(
    x=[yhat[0]], y=[yhat[1]], z=[yhat[2]], mode="markers",
    marker=dict(size=7, color="royalblue"), name="y-hat(β) (candidate)",
))
fig.add_trace(go.Scatter3d(
    x=[y[0], yhat[0]], y=[y[1], yhat[1]], z=[y[2], yhat[2]],
    mode="lines", line=dict(color="black", width=4, dash="dash"),
    name="e = y - y-hat(β)",
))

# OLS point, shown faintly unless it coincides with the candidate
fig.add_trace(go.Scatter3d(
    x=[yhat_star[0]], y=[yhat_star[1]], z=[yhat_star[2]], mode="markers",
    marker=dict(size=9, color="darkgreen", symbol="diamond", opacity=0.55),
    name="y-hat (OLS)",
))

if show_alt:
    yhat_alt = X @ SCRIPT_ALT_BETA
    rss_alt = float(np.sum((y - yhat_alt) ** 2))
    fig.add_trace(go.Scatter3d(
        x=[yhat_alt[0]], y=[yhat_alt[1]], z=[yhat_alt[2]], mode="markers",
        marker=dict(size=7, color="darkorange", symbol="x"),
        name=f"script alt β=(1.3,0.3), RSS={rss_alt:.3f}",
    ))

fig.update_layout(
    scene=dict(
        xaxis_title="obs 1 axis", yaxis_title="obs 2 axis", zaxis_title="obs 3 axis",
        aspectmode="cube",
        camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9)),
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(x=0.01, y=0.99),
    height=640,
)

with plot_col:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption("Drag to rotate. The surface is C(X) -- every possible y-hat lives on it.")

# ----------------------------------------------------------------------------
# Live numeric panel
# ----------------------------------------------------------------------------
with metrics_col:
    st.subheader("Live numbers")
    st.metric("RSS(β)  =  ‖e‖²", f"{rss:.4f}")
    gap = rss - rss_star
    st.metric(
        "RSS at OLS (minimum possible)",
        f"{rss_star:.4f}",
        delta=None if gap < 1e-9 else f"{gap:.4f} above minimum",
        delta_color="inverse",  # higher RSS than the minimum is "bad" -> red, not green
    )

    st.markdown("**Xᵀe** (zero only at OLS):")
    c1, c2 = st.columns(2)
    c1.metric("row 1", f"{Xte[0]:.4f}")
    c2.metric("row 2", f"{Xte[1]:.4f}")

    if is_orthogonal:
        st.success("e ⟂ C(X) -- you're at the OLS point.")
    else:
        st.info("e is **not** orthogonal to C(X) yet -- RSS can still be reduced.")

    with st.expander("Why this is the whole story"):
        st.markdown(
            "Every point on the grey surface is some `X @ beta`. The squared "
            "distance from `y` to that point is the residual sum of squares. "
            "As you drag β, you are sliding a point across the plane; RSS is "
            "smallest, and Xᵀe is exactly zero, at one point only -- the "
            "orthogonal projection of y onto C(X). That orthogonality "
            "condition, written out in coordinates, *is* the normal "
            "equations your seminar derives."
        )

st.divider()
with st.expander("Normal equations for the current y"):
    XtX = X.T @ X
    Xty = X.T @ y
    st.write("Xᵀ X =")
    st.write(XtX)
    st.write("Xᵀ y =")
    st.write(Xty)
    st.write("β̂ = (Xᵀ X)⁻¹ Xᵀ y =")
    st.write(ols_beta(y))
