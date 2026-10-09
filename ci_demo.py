# app.py
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm, t

import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Normal Mean CI: Coverage Simulator", layout="wide")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Sampling & CI Settings")

mu_true = st.sidebar.number_input("True mean (μ)", value=0.0, step=0.1)
sigma_true = st.sidebar.slider("True σ (for data generation)", 0.1, 10.0, 1.0, step=0.1)
n = st.sidebar.slider("Sample size (n)", 2, 2000, 30, step=1)

estimate_sigma = st.sidebar.checkbox("Estimate σ (use t-interval)", value=True,
                                     help="Unchecked: use known-σ z-interval with the true σ.")
conf_level = st.sidebar.slider("Confidence level", 0.80, 0.99, 0.95, step=0.01)
alpha = 1 - conf_level

R = st.sidebar.slider("Replications (number of samples)", 10, 3000, 400, step=10)
show_noncover = st.sidebar.checkbox("Show non-covering intervals", value=True)

seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

plot_lib = st.sidebar.selectbox("Plotting library", ["Plotly", "Matplotlib"], index=0)

st.sidebar.markdown("---")
if "run_id" not in st.session_state:
    st.session_state.run_id = 0
if st.sidebar.button("Resample"):
    st.session_state.run_id += 1

# ---------------- Simulation ----------------
rng = np.random.default_rng(seed + st.session_state.run_id)

# Draw R samples, compute CI per replication
# For efficiency: vectorize over R with shape (R, n)
X = rng.normal(loc=mu_true, scale=sigma_true, size=(R, n))
xbar = X.mean(axis=1)
s = X.std(axis=1, ddof=1)

se_known = sigma_true / np.sqrt(n)
se_unknown = s / np.sqrt(n)

if estimate_sigma:
    crit = t.ppf(1 - alpha / 2, df=n - 1)
    half_width = crit * se_unknown
else:
    crit = norm.ppf(1 - alpha / 2)
    half_width = crit * se_known

lower = xbar - half_width
upper = xbar + half_width
covers = (lower <= mu_true) & (mu_true <= upper)

df_intervals = pd.DataFrame({
    "rep": np.arange(1, R + 1),
    "lower": lower,
    "upper": upper,
    "xbar": xbar,
    "covers": covers
})

# Running (cumulative) coverage estimate
cum_cov = np.cumsum(covers) / np.arange(1, R + 1)

# Optionally hide non-covering intervals in the strip
if not show_noncover:
    df_show = df_intervals[df_intervals["covers"]].copy()
else:
    df_show = df_intervals.copy()

# ---------------- Layout ----------------
st.title("Normal Mean CI: Coverage Simulator")
st.caption("Build intuition that “95%” means long-run coverage. Compare known-σ (z) vs estimated-σ (t) intervals.")

left, right = st.columns([3, 2])

# ---------------- CI strip plot ----------------
with left:
    st.subheader("CI Strip Across Replications")

    if plot_lib == "Plotly":
        fig = go.Figure()

        # Vertical line at true mean
        fig.add_shape(type="line", x0=mu_true, x1=mu_true, y0=0, y1=max(1, len(df_show)),
                      line=dict(dash="dash"), name="μ (true)")

        # Two traces: cover vs miss, to color differently
        show_df_cover = df_show[df_show["covers"]]
        show_df_miss = df_show[~df_show["covers"]]

        # helper to add horizontal segments
        def add_segments(sub_df, name, opacity, line_width, color=None):
            if sub_df.empty:
                return
            for _, r in sub_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[r["lower"], r["upper"]],
                    y=[r["rep"], r["rep"]],
                    mode="lines",
                    line=dict(width=line_width) if color is None else dict(width=line_width, color=color),
                    opacity=opacity,
                    name=name,
                    hovertemplate=f"rep={int(r['rep'])}<br>CI=[{r['lower']:.3f}, {r['upper']:.3f}]<br>x̄={r['xbar']:.3f}<extra></extra>",
                    showlegend=False  # legend gets noisy with many segments
                ))

        add_segments(show_df_cover, "Covers μ", opacity=0.6, line_width=6)
        add_segments(show_df_miss, "Misses μ", opacity=1.0, line_width=6)

        # A small marker at x̄ to show center
        fig.add_trace(go.Scatter(
            x=df_show["xbar"], y=df_show["rep"], mode="markers",
            marker=dict(size=4, symbol="x"),
            name="x̄",
            hoverinfo="skip",
            showlegend=False
        ))

        # Axes & layout
        fig.update_yaxes(title="replication", range=[R + 1, 0])  # latest at top
        fig.update_xaxes(title="value")
        # reasonable height scaling
        height = min(800, max(300, int(0.6 * len(df_show)) + 200))
        fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(9, 6))
        # vertical line at true mean
        ax.axvline(mu_true, linestyle="--")

        cover_df = df_show[df_show["covers"]]
        miss_df = df_show[~df_show["covers"]]

        for _, r in cover_df.iterrows():
            ax.plot([r["lower"], r["upper"]], [r["rep"], r["rep"]], alpha=0.6, linewidth=3)
        for _, r in miss_df.iterrows():
            ax.plot([r["lower"], r["upper"]], [r["rep"], r["rep"]], alpha=1.0, linewidth=3)

        ax.scatter(df_show["xbar"], df_show["rep"], s=6)
        ax.set_ylim(R + 1, 0)
        ax.set_xlabel("value")
        ax.set_ylabel("replication")
        ax.grid(True, linestyle=":", linewidth=0.5)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

# ---------------- Running coverage plot ----------------
with right:
    st.subheader("Running Coverage vs Nominal")

    nominal = np.full(R, conf_level)
    rep_idx = np.arange(1, R + 1)

    if plot_lib == "Plotly":
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rep_idx, y=cum_cov, mode="lines", name="Running coverage"))
        fig2.add_trace(go.Scatter(x=rep_idx, y=nominal, mode="lines", name="Nominal", line=dict(dash="dash")))
        fig2.update_layout(height=350, xaxis_title="replication", yaxis_title="coverage",
                           yaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        ax2.plot(rep_idx, cum_cov, label="Running coverage")
        ax2.plot(rep_idx, nominal, linestyle="--", label="Nominal")
        ax2.set_xlabel("replication")
        ax2.set_ylabel("coverage")
        ax2.set_ylim(0, 1)
        ax2.grid(True, linestyle=":", linewidth=0.5)
        ax2.legend(loc="best")
        st.pyplot(fig2, clear_figure=True, use_container_width=True)

    # Quick stats
    st.markdown("**Summary**")
    final_cov = cum_cov[-1]
    miss_count = int((~covers).sum())
    st.write(f"Final running coverage: **{final_cov:.3f}** (nominal {conf_level:.3f})")
    st.write(f"Missed intervals: **{miss_count}** out of {R}")

# ---------------- Notes / Takeaways ----------------
st.markdown(
    """
**Takeaways**
- Coverage approaches the nominal level as **n** and/or **R** grow.
- Miscoverage is random — you’ll see misses both above and below μ.
- Using **t** (when σ is estimated) inflates the half-width vs known-σ **z**; the gap shrinks as **n** increases.
"""
)

