"""
IRLS: Fitting Logistic Regression by Iteratively Reweighted Least Squares
KL6000 Data Science — Week 4, Video 3 companion app

Same toy dataset as Video 1 (n = 40, x in [-4, 4], set.seed(42) in R).
The x, y arrays below were generated once in R and hardcoded here so the
app always matches the numbers quoted in the Video 1 and Video 3 scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="IRLS reweighting", layout="centered")

# ----------------------------------------------------------------------
# 1. Data (ported from R — see extract_data.R for provenance)
# ----------------------------------------------------------------------
x = np.array([
    -4.0000000000, -3.7948717949, -3.5897435897, -3.3846153846, -3.1794871795,
    -2.9743589744, -2.7692307692, -2.5641025641, -2.3589743590, -2.1538461538,
    -1.9487179487, -1.7435897436, -1.5384615385, -1.3333333333, -1.1282051282,
    -0.9230769231, -0.7179487179, -0.5128205128, -0.3076923077, -0.1025641026,
     0.1025641026,  0.3076923077,  0.5128205128,  0.7179487179,  0.9230769231,
     1.1282051282,  1.3333333333,  1.5384615385,  1.7435897436,  1.9487179487,
     2.1538461538,  2.3589743590,  2.5641025641,  2.7692307692,  2.9743589744,
     3.1794871795,  3.3846153846,  3.5897435897,  3.7948717949,  4.0000000000,
])
y = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1,
    0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
])
X = np.column_stack([np.ones_like(x), x])
N_ITERS = 6  # converges by iteration 6 (matches irls.R)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def irls_step(theta, X, y):
    eta = X @ theta
    p = sigmoid(eta)
    w = p * (1 - p)
    z = eta + (y - p) / w
    W = np.diag(w)
    theta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ z)
    return theta_new


@st.cache_data
def compute_trajectory():
    """theta, p, w at each state theta_0 (initial) ... theta_6 (converged)."""
    thetas = [np.array([0.0, 0.0])]
    for _ in range(N_ITERS):
        thetas.append(irls_step(thetas[-1], X, y))
    ps = [sigmoid(X @ th) for th in thetas]
    ws = [p * (1 - p) for p in ps]
    return thetas, ps, ws


thetas, ps, ws = compute_trajectory()

# Cross-check against the R/glm reference value baked into the video scripts
_glm_reference = np.array([0.0, 1.0997781261])
assert np.allclose(thetas[-1], _glm_reference, atol=1e-6), "IRLS did not converge to the expected glm() value"

# ----------------------------------------------------------------------
# 2. UI
# ----------------------------------------------------------------------
st.title("IRLS: watching logistic regression fit itself")
st.markdown(
    "Same forty points as the *\"Why Regression Fails for Classification\"* video "
    "(n = 40, x ∈ [−4, 4]). Logistic regression has no closed-form solution, so θ is "
    "found by **Iteratively Reweighted Least Squares**: at each step, every point is "
    "reweighted by wᵢ = pᵢ(1−pᵢ), and a weighted least squares problem is solved for a "
    "new θ. Move the slider to step through the algorithm."
)

if "iteration" not in st.session_state:
    st.session_state.iteration = 0

col_back, col_slider, col_fwd = st.columns([1, 6, 1])
with col_back:
    if st.button("◀", use_container_width=True, disabled=st.session_state.iteration == 0):
        st.session_state.iteration -= 1
with col_fwd:
    if st.button("▶", use_container_width=True, disabled=st.session_state.iteration == N_ITERS):
        st.session_state.iteration += 1
with col_slider:
    st.session_state.iteration = st.slider(
        "Iteration", min_value=0, max_value=N_ITERS, value=st.session_state.iteration,
        help="0 = initial state (θ = (0,0)) before any update. 6 = converged.",
    )

k = st.session_state.iteration
theta_k, p_k, w_k = thetas[k], ps[k], ws[k]
theta_display = np.where(np.abs(theta_k) < 1e-9, 0.0, theta_k)  # avoid "-0.0000" from float noise

label = "Initial state — before any update" if k == 0 else (
    "Converged" if k == N_ITERS else f"After iteration {k}"
)
status = "🟢 Converged" if k == N_ITERS else "🟡 In progress"
st.markdown(f"**{label}** &nbsp;|&nbsp; θ = ({theta_display[0]:.4f}, {theta_display[1]:.4f}) &nbsp;|&nbsp; {status}")

# ----------------------------------------------------------------------
# 3. Panel A — data, current fitted curve, weight-scaled points
# ----------------------------------------------------------------------
grid_x = np.linspace(-5, 5, 400)
curve = sigmoid(theta_k[0] + theta_k[1] * grid_x)

fig1, ax1 = plt.subplots(figsize=(6.5, 4))
ax1.axhspan(1.0, 1.15, color="0.88", zorder=0)
ax1.axhspan(-0.15, 0.0, color="0.88", zorder=0)
ax1.plot(grid_x, curve, color="#3b6ea5", linewidth=2, zorder=2)
sizes = 15 + 300 * w_k          # min size ~15, max (w=0.25) ~90
alphas = 0.25 + 0.75 * (w_k / 0.25)
for xi, yi, s, a in zip(x, y, sizes, alphas):
    ax1.scatter(xi, yi, s=s, alpha=min(a, 1.0), color="#b23a2e", zorder=3, edgecolors="none")
ax1.set_xlim(-5, 5)
ax1.set_ylim(-0.15, 1.15)
ax1.set_xlabel("x")
ax1.set_ylabel("y  /  fitted p(x)")
ax1.set_title("Point size/opacity = current weight wᵢ = pᵢ(1−pᵢ)")
ax1.grid(alpha=0.3)
st.pyplot(fig1, clear_figure=True)

# ----------------------------------------------------------------------
# 4. Panel B — weight profile (fixed axes so sharpening is visible)
# ----------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6.5, 3))
ax2.plot(x, w_k, color="#b23a2e", marker="o", markersize=4, linewidth=1.5)
ax2.axhline(0.25, color="0.75", linestyle=":", linewidth=1)
ax2.set_xlim(-4.3, 4.3)
ax2.set_ylim(0, 0.27)
ax2.set_xlabel("x")
ax2.set_ylabel("wᵢ = pᵢ(1−pᵢ)")
ax2.set_title("Weight profile at this iteration  (max possible weight = 0.25, dotted)")
ax2.grid(alpha=0.3)
st.pyplot(fig2, clear_figure=True)

# ----------------------------------------------------------------------
# 5. Panel C — convergence trace (θ₁, the slope, across iterations)
# ----------------------------------------------------------------------
slope_trace = [th[1] for th in thetas]
fig3, ax3 = plt.subplots(figsize=(6.5, 2.3))
ax3.plot(range(N_ITERS + 1), slope_trace, color="0.6", linewidth=1.5, zorder=1)
ax3.axhline(slope_trace[-1], color="0.8", linestyle=":", linewidth=1, zorder=0)
ax3.scatter(range(N_ITERS + 1), slope_trace, color="0.6", s=25, zorder=2)
ax3.scatter([k], [slope_trace[k]], color="#3b6ea5", s=90, zorder=3)
ax3.set_xlim(-0.3, N_ITERS + 0.3)
ax3.set_ylim(0, 1.25)
ax3.set_xticks(range(N_ITERS + 1))
ax3.set_xlabel("iteration")
ax3.set_ylabel("θ₁")
ax3.set_title(f"Convergence of θ₁  (final value ≈ {slope_trace[-1]:.4f})")
ax3.grid(alpha=0.3)
st.pyplot(fig3, clear_figure=True)

st.caption(
    "Verified against R: this implementation converges to θ = (0, 1.099778), "
    "matching `glm(y ~ x, family = binomial)` to 6 decimal places."
)
