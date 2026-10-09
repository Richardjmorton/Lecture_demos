# app.py
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import t

import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Linear Model Demo", layout="wide")

# ---- Sidebar controls -------------------------------------------------------
st.sidebar.header("Data & Model Controls")

plot_lib = st.sidebar.selectbox("Plotting library", ["Plotly", "Matplotlib"], index=0)

# Data generation
n = st.sidebar.slider("Number of points (n)", 20, 2000, 200, step=10)
x_min, x_max = st.sidebar.slider("x range", -10.0, 10.0, (-3.0, 3.0), step=0.1)
beta0 = st.sidebar.slider("True intercept (β₀)", -5.0, 5.0, 1.0, step=0.1)
beta1 = st.sidebar.slider("True slope (β₁)", -5.0, 5.0, 2.0, step=0.1)

noise_type = st.sidebar.selectbox(
    "Noise distribution",
    ["Normal", "Poisson", "Exponential"],
    index=0,
    help="This is for simulating the data noise; the fit is standard OLS in all cases."
)

# Noise parameters
if noise_type == "Normal":
    sigma = st.sidebar.slider("Normal σ", 0.0, 5.0, 1.0, step=0.1)
elif noise_type == "Poisson":
    lam = st.sidebar.slider("Poisson λ", 0.1, 10.0, 2.0, step=0.1)
else:  # Exponential
    scale = st.sidebar.slider("Exponential scale (θ)", 0.1, 10.0, 1.0, step=0.1)

center_noise = st.sidebar.checkbox(
    "Center noise to zero mean (recommended)",
    value=True,
    help="Subtracts the distribution mean so the noise has mean ~ 0. Keeps OLS unbiased."
)

seed = st.sidebar.number_input("Random seed", min_value=0, value=123, step=1)

# Model & intervals
st.sidebar.header("Intervals")
conf_level = st.sidebar.slider("Confidence level", 0.80, 0.99, 0.95, step=0.01)
alpha = 1.0 - conf_level
show_ci = st.sidebar.toggle("Show confidence interval (mean response)", value=True)
show_pi = st.sidebar.toggle("Show prediction interval (new observation)", value=True)

fit_intercept = st.sidebar.checkbox("Fit intercept", value=True)

# Plot options
st.sidebar.header("Plot Options")
show_true_line = st.sidebar.checkbox("Show true line", value=True)
point_size = st.sidebar.slider("Point size", 10, 200, 40)

# ---- Data generation ---------------------------------------------------------
rng = np.random.default_rng(seed)
x = rng.uniform(x_min, x_max, size=n)
x = np.sort(x)

# True mean function
mu = beta0 + beta1 * x

# Draw noise
if noise_type == "Normal":
    eps = rng.normal(0.0, sigma, size=n)
    mean_shift = 0.0
elif noise_type == "Poisson":
    eps = rng.poisson(lam, size=n).astype(float)
    mean_shift = lam
else:  # Exponential
    eps = rng.exponential(scale=scale, size=n)
    mean_shift = scale

if center_noise:
    eps = eps - mean_shift

y = mu + eps

df = pd.DataFrame({"x": x, "y": y, "mu_true": mu})

# ---- OLS fit ----------------------------------------------------------------
# Design matrix
if fit_intercept:
    X = np.column_stack([np.ones_like(x), x])
else:
    X = x.reshape(-1, 1)

# OLS closed form
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
beta_hat = XtX_inv @ (X.T @ y)

# Predictions on data points
y_hat = X @ beta_hat
resid = y - y_hat
df_resid = len(y) - X.shape[1]
s2 = (resid @ resid) / df_resid
s = np.sqrt(s2)

# Grid for smooth line + intervals
grid = np.linspace(x_min, x_max, 400)
if fit_intercept:
    X0 = np.column_stack([np.ones_like(grid), grid])
else:
    X0 = grid.reshape(-1, 1)

y0_hat = X0 @ beta_hat
se_mean = np.sqrt(np.sum(X0 @ XtX_inv * X0, axis=1) * s2)
se_pred = np.sqrt(se_mean**2 + s2)

tcrit = t.ppf(1 - alpha / 2.0, df_resid)

ci_lower = y0_hat - tcrit * se_mean
ci_upper = y0_hat + tcrit * se_mean
pi_lower = y0_hat - tcrit * se_pred
pi_upper = y0_hat + tcrit * se_pred

curve = pd.DataFrame(
    {
        "x": grid,
        "yhat": y0_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pi_lower": pi_lower,
        "pi_upper": pi_upper,
    }
)

# ---- Layout ------------------------------------------------------------------
st.title("Interactive Linear Model Demo")
st.caption("Generate data with different noise, fit OLS, toggle confidence & prediction intervals.")

left, right = st.columns([2, 1])

with left:
    if plot_lib == "Plotly":
        fig = go.Figure()

        # Prediction interval band (lighter)
        if show_pi:
            fig.add_traces([
                go.Scatter(x=curve["x"], y=curve["pi_upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
                go.Scatter(x=curve["x"], y=curve["pi_lower"], mode="lines", fill="tonexty", line=dict(width=0), name="Prediction Interval", hoverinfo="skip", opacity=0.2),
            ])

        # Confidence interval band (darker)
        if show_ci:
            fig.add_traces([
                go.Scatter(x=curve["x"], y=curve["ci_upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
                go.Scatter(x=curve["x"], y=curve["ci_lower"], mode="lines", fill="tonexty", line=dict(width=0), name="Confidence Interval", hoverinfo="skip", opacity=0.35),
            ])

        # Fitted line
        fig.add_trace(go.Scatter(x=curve["x"], y=curve["yhat"], mode="lines", name="Fitted line"))

        # True line (optional)
        if show_true_line:
            fig.add_trace(go.Scatter(x=grid, y=beta0 + beta1 * grid, mode="lines", name="True line", line=dict(dash="dash")))

        # Points
        fig.add_trace(go.Scatter(
            x=df["x"], y=df["y"], mode="markers", name="Data", marker=dict(size=point_size/4), hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
        ))

        fig.update_layout(height=520, xaxis_title="x", yaxis_title="y", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    else:  # Matplotlib
        fig, ax = plt.subplots(figsize=(8, 5))

        # Prediction interval
        if show_pi:
            ax.fill_between(curve["x"], curve["pi_lower"], curve["pi_upper"], alpha=0.15, label="Prediction Interval")

        # Confidence interval
        if show_ci:
            ax.fill_between(curve["x"], curve["ci_lower"], curve["ci_upper"], alpha=0.35, label="Confidence Interval")

        # Fitted line
        ax.plot(curve["x"], curve["yhat"], linewidth=2, label="Fitted line")

        # True line
        if show_true_line:
            ax.plot(grid, beta0 + beta1 * grid, linestyle="--", label="True line")

        # Points
        ax.scatter(df["x"], df["y"], s=point_size, alpha=0.7, label="Data")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", linewidth=0.5)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

with right:
    st.subheader("Model Summary")
    param_names = ["β₀ (intercept)", "β₁ (slope)"] if fit_intercept else ["β₁ (slope)"]
    for name, val in zip(param_names, beta_hat):
        st.write(f"**{name}:** {val:.4f}")
    st.write(f"**Residual std. error (σ̂):** {s:.4f}")
    st.write(f"**Degrees of freedom:** {df_resid}")

    with st.expander("Download generated data"):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            file_name="simulated_linear_data.csv",
            mime="text/csv",
        )

st.markdown(
    """
**Notes**
- Intervals use standard OLS+t. Exact under Normal noise; large-n reasonable otherwise.
- With Poisson/Exponential, centering the noise avoids bias in the mean.
- Confidence interval = uncertainty of **E[y|x]**. Prediction interval = uncertainty of a **new y**.
"""
)

