"""
Streamlit app: Multivariate Normal Playground


This demo lets you interactively change the mean vector and covariance (via
standard deviations and correlation), visualize generated samples, overlay
PDF contours, and inspect sample vs. true statistics. It also includes the
multivariate normal density equation and study tips.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import streamlit as st

st.set_page_config(page_title="Multivariate Normal Playground", layout="wide")

# ---------------------- Title & Equation ----------------------
st.title("Multivariate Normal: Mean & Covariance Playground")

st.markdown(
    "Use the controls to change the distribution parameters. We'll generate samples, "
    "draw PDF contours, and compare true vs. empirical statistics."
)

st.latex(r"""
    p(\mathbf{x};\, \boldsymbol{\mu},\, \Sigma)
    = \frac{1}{(2\pi)^{d/2} \, |\Sigma|^{1/2}}
      \exp\!\left(-\tfrac{1}{2}\,(\mathbf{x}-\boldsymbol{\mu})^{\top}\,\Sigma^{-1}\,(\mathbf{x}-\boldsymbol{\mu})\right)
""")

st.caption(
    "Here we focus on **d = 2** so you can see the geometry. Σ must be symmetric positive definite."
)

# ---------------------- Sidebar Controls ----------------------
with st.sidebar:
    st.header("Parameters")
    seed = st.number_input("Random seed", value=0, step=1)
    np.random.seed(int(seed))

    n = st.slider("Samples (N)", min_value=100, max_value=20000, value=2000, step=100)

    st.subheader("Mean (μ)")
    mu_x = st.slider("μ₁", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    mu_y = st.slider("μ₂", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

    st.subheader("Covariance (Σ) via σ and ρ")
    sigma_x = st.slider("σ₁ (std along x₁)", min_value=0.05, max_value=5.0, value=1.0, step=0.05)
    sigma_y = st.slider("σ₂ (std along x₂)", min_value=0.05, max_value=5.0, value=1.0, step=0.05)
    rho     = st.slider("ρ (correlation)", min_value=-0.99, max_value=0.99, value=0.0, step=0.01)

    st.subheader("Overlays")
    show_contours   = st.checkbox("Show PDF contours", True)
    show_axes       = st.checkbox("Show principal axes", True)
    show_ellipses   = st.checkbox("Show confidence ellipses (1σ, 2σ)", True)
    show_sample_ell = st.checkbox("Show sample (empirical) ellipse", False)

# True parameters
mu = np.array([mu_x, mu_y])
Sigma = np.array([
    [sigma_x**2, rho * sigma_x * sigma_y],
    [rho * sigma_x * sigma_y, sigma_y**2],
])

# Diagnostics on Σ
try:
    det_S = float(np.linalg.det(Sigma))
    cond_S = float(np.linalg.cond(Sigma))
except np.linalg.LinAlgError:
    det_S = float("nan"); cond_S = float("inf")

if det_S <= 0 or not np.isfinite(cond_S):
    st.error("Σ is not positive definite. Adjust σ's or ρ.")

# ---------------------- Sample Generation ----------------------
# Parameterization via (σ1, σ2, ρ) guarantees PSD as long as |ρ|<1 and σ>0.
samples = np.random.multivariate_normal(mean=mu, cov=Sigma, size=int(n))

# Empirical statistics
xbar = samples.mean(axis=0)
S_emp = np.cov(samples, rowvar=False)

# Eigen-decomposition for geometry
lam, U = np.linalg.eigh(Sigma)  # ascending eigenvalues
order = np.argsort(lam)[::-1]   # descending for plotting width/height
lam_sorted = lam[order]
U_sorted = U[:, order]

# ---------------------- Plots ----------------------
col_plot, col_stats = st.columns([2.0, 1.0])

with col_plot:
    st.subheader("Samples & Geometry (2D)")
    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter of samples
    ax.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.35, linewidths=0)
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    # Contours of the true PDF
    if show_contours:
        # Grid scaled to ±4 stds along each axis
        gx = np.linspace(mu_x - 4 * sigma_x, mu_x + 4 * sigma_x, 160)
        gy = np.linspace(mu_y - 4 * sigma_y, mu_y + 4 * sigma_y, 160)
        X, Y = np.meshgrid(gx, gy)
        pos = np.dstack((X, Y))
        rv = multivariate_normal(mean=mu, cov=Sigma)
        Z = rv.pdf(pos)
        cs = ax.contour(X, Y, Z, levels=8)
        ax.clabel(cs, inline=1, fmt="%.3f", fontsize=8)

    # Principal axes (eigenvectors), length scaled by sqrt(eigenvalue)
    if show_axes:
        for val, vec in zip(lam_sorted, U_sorted.T):
            scale = 2.0 * np.sqrt(val)  # approx 2σ length
            ax.arrow(mu[0], mu[1], scale * vec[0], scale * vec[1],
                     head_width=0.08, length_includes_head=True)
            ax.arrow(mu[0], mu[1], -scale * vec[0], -scale * vec[1],
                     head_width=0.08, length_includes_head=True)

    # Confidence ellipses from true Σ
    if show_ellipses:
        angle_deg = np.degrees(np.arctan2(U_sorted[1, 0], U_sorted[0, 0]))
        for k in (1, 2):  # ~39% and ~86% mass in 2D, not the 68/95 rule
            e = Ellipse(xy=mu,
                        width=2 * k * np.sqrt(lam_sorted[0]),
                        height=2 * k * np.sqrt(lam_sorted[1]),
                        angle=angle_deg,
                        fill=False, linewidth=2, alpha=0.9)
            ax.add_patch(e)

    # Empirical ellipse using sample covariance
    if show_sample_ell:
        lamS, US = np.linalg.eigh(S_emp)
        orderS = np.argsort(lamS)[::-1]
        lamS = lamS[orderS]; US = US[:, orderS]
        angleS = np.degrees(np.arctan2(US[1, 0], US[0, 0]))
        eS = Ellipse(xy=xbar,
                     width=2 * np.sqrt(lamS[0]),
                     height=2 * np.sqrt(lamS[1]),
                     angle=angleS,
                     fill=False, linestyle=":", linewidth=2, alpha=0.9)
        ax.add_patch(eS)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.grid(True, linestyle=":", linewidth=0.5)
    st.pyplot(fig, use_container_width=True)

with col_stats:
    st.subheader("True parameters")
    st.write("**μ (mean):**", mu)
    st.write("**Σ (covariance):**")
    st.write(Sigma)
    st.write(f"|Σ| (determinant): {det_S:.6f}")
    st.write(f"cond(Σ): {cond_S:.2f}")

    st.divider()
    st.subheader("Empirical (from samples)")
    st.write("x̄ (sample mean):", xbar)
    st.write("S (sample covariance):")
    st.write(S_emp)

    # Average log-likelihood per sample under the true model
    rv = multivariate_normal(mean=mu, cov=Sigma)
    avg_ll = float(rv.logpdf(samples).mean())
    st.write(f"Avg log-likelihood per sample: {avg_ll:.3f}")

    # Conditioning warn

with st.expander("Study tips & guided exercises"):
    st.markdown(
        """
- **Correlation vs. covariance**: with fixed σ₁ and σ₂, changing ρ rotates/tilts the ellipse; σ’s set the radii.
- **Elliptical contours** come from constant Mahalanobis distance.
- **PCA link**: eigenvectors of Σ are the principal directions; eigenvalues are variances along those directions.
- **Near-singular Σ** (|ρ|→1 or tiny σ): one eigenvalue → 0; contours collapse toward a line; |Σ| → 0.
- **Sampling noise**: increase N to see x̄ → μ and S → Σ. Compare true vs. sample ellipse.
- **Parameterization**: (σ₁, σ₂, ρ) ⇒ PSD Σ when |ρ|<1 and σ>0.
        """
    )
    st.markdown("**Key formulas:**")
    st.latex(r"(x-\mu)^{\top}\,\Sigma^{-1}\,(x-\mu) = c")
    st.latex(r"\Sigma = \begin{bmatrix}\sigma_1^2 & \rho\,\sigma_1\sigma_2 \\ \rho\,\sigma_1\sigma_2 & \sigma_2^2\end{bmatrix}")

    st.markdown("**Try these:**")
    st.markdown("1) Set σ₁ = σ₂ = 1 and sweep ρ from −0.9 to +0.9. Note the rotation.")
    st.markdown("2) Fix ρ = 0 and vary σ₁ vs. σ₂. Which axis stretches?")
    st.markdown("3) Push ρ to 0.98 and reduce one σ to 0.1. Watch |Σ| and cond(Σ).")
    st.markdown("4) Double N and watch (x̄, S) approach (μ, Σ).")
    st.markdown("5) Toggle the sample ellipse for small N and compare.")


with st.expander("Theory: quick reference"):
    st.markdown("**PDF (d=2)**")
    st.latex(r"p(\mathbf{x}) = \frac{1}{2\pi\sqrt{|\Sigma|}}\,\exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\top}\,\Sigma^{-1}\,(\mathbf{x}-\boldsymbol{\mu})\right)")

    st.markdown("**Build Σ from (σ₁, σ₂, ρ)**")
    st.latex(r"\Sigma = \begin{bmatrix}\sigma_1^2 & \rho\,\sigma_1\sigma_2 \\ \rho\,\sigma_1\sigma_2 & \sigma_2^2\end{bmatrix}")

    st.markdown("**Eigen-decomposition**")
    st.latex(r"\Sigma = U\,\mathrm{diag}(\lambda_1,\lambda_2)\,U^{\top}")

    st.markdown("**Confidence/content ellipses**")
    st.latex(r"(x-\mu)^{\top}\,\Sigma^{-1}\,(x-\mu) \le c \;\Longleftrightarrow\; \Pr(\chi^2_2 \le c)")
    st.markdown("Examples: c=1 → ≈39.3% mass; c=4 → ≈86.5% mass in 2D.")

