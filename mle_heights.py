"""
KL6000 Data Science — Interactive Likelihood Function Demonstration
====================================================================

Companion app for Lecture 2 (Probability, MLE & Bayesian Inference).

Demonstrates the likelihood function for the mean of a Gaussian using the
"Student Heights" running example from the lecture notes:

    x_i ~ N(mu, sigma^2),   i = 1, ..., n

The student controls a *guess* for (mu, sigma) via sliders and sees:

  1. The guessed Normal pdf plotted against the observed data (rug plot).
  2. The resulting log-likelihood value, ell(mu, sigma) = sum_i log p(x_i | mu, sigma).
  3. A profile log-likelihood curve ell(mu) (sigma held at its current guessed
     value), shifted so its peak at the sample MLE (mu_hat = x_bar) sits at 0.

Run with:
    streamlit run app.py
"""

import numpy as np
import streamlit as st
from scipy.stats import norm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Fixed "ground truth" data-generating parameters.
# These match the worked example in lecture2_mle.tex ("Estimating Mean
# Student Height": mu0 = 175 cm, sigma = 8 cm).
# ---------------------------------------------------------------------------
TRUE_MEAN = 175.0
TRUE_SD = 8.0
MAX_N = 200
INITIAL_SEED = 42

# ---------------------------------------------------------------------------
# Fixed ranges — these never change, regardless of n or slider values, so
# the axes never rescale/flicker as the student explores.
# ---------------------------------------------------------------------------
MU_SLIDER_MIN, MU_SLIDER_MAX = 150.0, 200.0
SIGMA_SLIDER_MIN, SIGMA_SLIDER_MAX = 2.0, 20.0

PDF_X_MIN, PDF_X_MAX = 140.0, 210.0      # x-range, top panel (data + pdf)
PDF_Y_MAX = 1.0 / (SIGMA_SLIDER_MIN * np.sqrt(2 * np.pi)) * 1.08  # headroom above tallest possible pdf

MU_GRID_MIN, MU_GRID_MAX = 150.0, 200.0  # x-range, bottom panel (candidate means)
LOGLIK_Y_MIN, LOGLIK_Y_MAX = -40.0, 2.0  # y-range, bottom panel (relative log-lik)

N_GRID_POINTS = 400


def log_likelihood(x: np.ndarray, mu: float, sigma: float) -> float:
    """Gaussian log-likelihood ell(mu, sigma) = sum_i log p(x_i | mu, sigma)."""
    return float(np.sum(norm.logpdf(x, loc=mu, scale=sigma)))


def profile_log_likelihood(x: np.ndarray, mu_grid: np.ndarray, sigma: float) -> np.ndarray:
    """ell(mu) over a grid of candidate means, sigma held fixed."""
    n = x.size
    sum_x = np.sum(x)
    sum_x2 = np.sum(x ** 2)
    # Expand sum((x_i - mu)^2) = sum(x_i^2) - 2*mu*sum(x_i) + n*mu^2
    # to evaluate the whole grid in closed form without a Python loop.
    sse = sum_x2 - 2 * mu_grid * sum_x + n * mu_grid ** 2
    return -0.5 * n * np.log(2 * np.pi * sigma ** 2) - sse / (2 * sigma ** 2)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="The Likelihood Function", layout="wide")

st.title("The Likelihood Function: Estimating a Mean")
st.markdown(
    "We observe student heights x\u2081, \u2026, x\u2099 ~ N(\u03bc, \u03c3\u00b2) "
    "(the running example from Lecture 2). Adjust your **guess** for \u03bc and "
    "\u03c3 below and watch how well the resulting Normal curve explains the "
    "observed data — and how that is reflected in the log-likelihood."
)

# --- Sidebar controls ------------------------------------------------------
st.sidebar.header("Controls")

if "pool" not in st.session_state:
    rng = np.random.default_rng(INITIAL_SEED)
    st.session_state.pool = rng.normal(TRUE_MEAN, TRUE_SD, MAX_N)

if st.sidebar.button("🎲 Draw a new sample"):
    rng = np.random.default_rng()  # fresh entropy, no fixed seed
    st.session_state.pool = rng.normal(TRUE_MEAN, TRUE_SD, MAX_N)

n_obs = st.sidebar.slider("Number of observations, n", min_value=2, max_value=MAX_N, value=20, step=1)

mu_guess = st.sidebar.slider(
    "Guessed mean, \u03bc (cm)", min_value=MU_SLIDER_MIN, max_value=MU_SLIDER_MAX,
    value=160.0, step=0.5,
)
sigma_guess = st.sidebar.slider(
    "Guessed standard deviation, \u03c3 (cm)", min_value=SIGMA_SLIDER_MIN, max_value=SIGMA_SLIDER_MAX,
    value=12.0, step=0.5,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"True (unknown to the model) parameters used to generate the data: "
    f"\u03bc = {TRUE_MEAN:.0f} cm, \u03c3 = {TRUE_SD:.0f} cm."
)

data = st.session_state.pool[:n_obs]
mu_hat = float(np.mean(data))

# --- Top panel: data, guessed pdf, log-likelihood score ---------------------
col_plot, col_score = st.columns([3, 1])

with col_plot:
    fig1, ax1 = plt.subplots(figsize=(7.5, 3.2))
    x_grid = np.linspace(PDF_X_MIN, PDF_X_MAX, N_GRID_POINTS)
    ax1.plot(
        x_grid, norm.pdf(x_grid, mu_guess, sigma_guess),
        lw=2, color="tab:blue",
        label=f"guessed pdf: N({mu_guess:.1f}, {sigma_guess:.1f}\u00b2)",
    )
    ax1.plot(
        data, np.zeros_like(data), marker="|", linestyle="None",
        markersize=18, markeredgewidth=1.5, color="black", label="observations (rug)",
    )
    ax1.set_xlim(PDF_X_MIN, PDF_X_MAX)
    ax1.set_ylim(0, PDF_Y_MAX)
    ax1.set_xlabel("Height (cm)")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax1.set_title(f"Data (n = {n_obs}) and guessed Normal density")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

with col_score:
    ll_current = log_likelihood(data, mu_guess, sigma_guess)
    st.metric("Log-likelihood, \u2113(\u03bc, \u03c3)", f"{ll_current:,.2f}")
    st.caption(
        "The raw likelihood L(\u03bc,\u03c3) = \u220f\u1d62 p(x\u1d62|\u03bc,\u03c3) "
        "underflows to numerically 0 for even moderate n — this is exactly the "
        "numerical-stability argument from Lecture 2 for working with \u2113 "
        "instead of L. Higher (less negative) \u2113 means a better-fitting guess."
    )
    st.markdown(f"**Sample mean x\u0304:** {mu_hat:.2f} cm")
    st.markdown(f"**Your guess \u03bc:** {mu_guess:.1f} cm")

# --- Bottom panel: profile log-likelihood curve -----------------------------
st.markdown("### Profile Log-Likelihood for \u03bc")
st.markdown(
    "Holding \u03c3 fixed at your current guessed value, this shows "
    "\u2113(\u03bc) \u2212 \u2113(\u03bc\u0302) across a range of candidate means, where "
    "\u03bc\u0302 = x\u0304 is the MLE for the *current* sample. "
    "The axes are fixed throughout — as n grows, watch the curve narrow "
    "sharply around x\u0304 (greater curvature = a more precise estimate; "
    "this is the Hessian/precision argument from Lecture 2)."
)

mu_grid = np.linspace(MU_GRID_MIN, MU_GRID_MAX, N_GRID_POINTS)
ll_grid = profile_log_likelihood(data, mu_grid, sigma_guess)
ll_at_mle = log_likelihood(data, mu_hat, sigma_guess)
relative_ll = ll_grid - ll_at_mle

fig2, ax2 = plt.subplots(figsize=(10, 3.5))
ax2.plot(mu_grid, relative_ll, lw=2, color="tab:blue", label="log-likelihood \u2113(\u03bc) \u2212 \u2113(\u03bc\u0302)")
ax2.axvline(mu_hat, color="tab:green", linestyle=":", lw=1.5, label=f"\u03bc\u0302 = x\u0304 = {mu_hat:.1f}")
ax2.axvline(mu_guess, color="tab:red", linestyle="--", lw=1.5, label=f"current guess \u03bc = {mu_guess:.1f}")

marker_y = ll_current - ll_at_mle
marker_y_clipped = max(marker_y, LOGLIK_Y_MIN)
ax2.plot(mu_guess, marker_y_clipped, "o", color="tab:red", markersize=9, zorder=5)

ax2.set_xlim(MU_GRID_MIN, MU_GRID_MAX)
ax2.set_ylim(LOGLIK_Y_MIN, LOGLIK_Y_MAX)
ax2.set_xlabel("Candidate mean, \u03bc (cm)")
ax2.set_ylabel("Relative log-likelihood")
ax2.legend(loc="lower center", fontsize=9, framealpha=0.9, ncol=3)
ax2.axhline(0, color="grey", lw=0.5)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

if marker_y < LOGLIK_Y_MIN:
    st.caption(
        f"Note: your current guess has relative log-likelihood "
        f"{marker_y:,.1f}, below the plot's fixed lower limit "
        f"({LOGLIK_Y_MIN:.0f}), so its marker is pinned to the bottom edge."
    )
