import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="Coin Toss MLE Demo", layout="wide")
st.title("Coin Toss MLE (Binomial)")
st.caption("Interactively explore how the likelihood changes with data and how the MLE emerges as the sample proportion.")

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    p_true = st.slider("True probability of Heads (p)", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    n = st.slider("Number of tosses (n)", min_value=1, max_value=1000, value=50, step=1)

    seed = st.number_input("Random seed (optional)", value=0, min_value=0, step=1)

    colb1, colb2 = st.columns(2)
    with colb1:
        do_resample = st.button("Resample data", use_container_width=True)
    with colb2:
        clear_data = st.button("Clear data", use_container_width=True)

    st.markdown("---")
    view_mode = st.radio("Plot view", ["Likelihood (relative)", "Log-likelihood"], index=0)
    show_data_points = st.checkbox("Show simulated sequence (first 200)", value=False)

# -----------------------------
# Session state for data
# -----------------------------
if "tosses" not in st.session_state:
    st.session_state.tosses = None
    st.session_state.current_params = (None, None)

# Clear data if requested
if clear_data:
    st.session_state.tosses = None
    st.session_state.current_params = (None, None)

# Resample logic: resample on button OR if parameters changed and we have no data yet
params_changed = st.session_state.current_params != (p_true, n)
if do_resample or st.session_state.tosses is None or params_changed:
    rng = np.random.default_rng(seed if seed != 0 else None)
    # 1 = Head, 0 = Tail
    st.session_state.tosses = rng.binomial(1, p_true, size=n).astype(int)
    st.session_state.current_params = (p_true, n)

# Use data from state
x = st.session_state.tosses
n = len(x)
k = int(x.sum())
p_hat = k / n if n > 0 else np.nan

# -----------------------------
# Left: data summary; Right: likelihood plot
# -----------------------------
left, right = st.columns([0.4, 0.6])

with left:
    st.subheader("Data summary")
    st.metric("Tosses (n)", n)
    st.metric("Heads (k)", k)
    st.metric("Sample proportion (MLE)", f"{p_hat:.4f}")
    st.latex(r"\hat{p} = k/n")

    st.write(":orange[True p] vs :blue[MLE]")
    fig_counts, ax_counts = plt.subplots()
    ax_counts.bar(["Tails (0)", "Heads (1)"] , [n - k, k])
    ax_counts.set_ylabel("Count")
    st.pyplot(fig_counts, use_container_width=True)

    if show_data_points:
        st.write("**First 200 simulated outcomes:**")
        shown = x[:200]
        st.code(" ".join(map(str, shown.tolist())))

with right:
    st.subheader("Likelihood for p under Binomial(n, p)")

    # p-grid (avoid exact 0 or 1 for numerical stability)
    p_grid = np.linspace(1e-6, 1 - 1e-6, 2000)

    # Log-likelihood (up to additive constant): k*log(p) + (n-k)*log(1-p)
    logL = k * np.log(p_grid) + (n - k) * np.log(1 - p_grid)

    fig_like, ax_like = plt.subplots()

    if view_mode == "Likelihood (relative)":
        # Normalize likelihood to peak at 1 for visibility
        logL_shift = logL - logL.max()
        L_rel = np.exp(logL_shift)
        ax_like.plot(p_grid, L_rel)
        ax_like.set_ylabel("Relative Likelihood (scaled to 1 at MLE)")
    else:
        logL_shift = logL - logL.max()
        ax_like.plot(p_grid, logL_shift)
        ax_like.set_ylim(-100,1)
        ax_like.set_ylabel("Log-likelihood")

    ax_like.set_xlabel("p (probability of Heads)")

    # Vertical lines for true p and MLE
    ax_like.axvline(p_true, linestyle="--", linewidth=1)
    ax_like.axvline(p_hat, linestyle=":", linewidth=1)

    # Annotate
    ax_like.annotate("True p", xy=(p_true, ax_like.get_ylim()[1]), xytext=(5, -20), textcoords="offset points",
                     ha="left", va="top", rotation=90)
    ax_like.annotate("MLE = k/n", xy=(p_hat, ax_like.get_ylim()[1]), xytext=(5, -20), textcoords="offset points",
                     ha="left", va="top", rotation=90)

    st.pyplot(fig_like, use_container_width=True)

# -----------------------------
# Math & tips
# -----------------------------
with st.expander("Show math & tips"):
    st.markdown("**Model.** Assume i.i.d. Bernoulli(p) trials, total n tosses, k heads.")
    st.latex(r"L(p) = p^{k} (1-p)^{n-k}")
    st.latex(r"\ell(p) = k\,\log p + (n-k)\,\log(1-p)")
    st.latex(r"\hat{p} = k/n")
    st.markdown("Boundary cases: if k=0 or k=n, treat the MLE as 0 or 1 respectively.")
    st.markdown("Learning tips:")
    st.markdown("- Increase n to see the likelihood peak sharpen.\n- Compare the vertical dashed line (true p) and dotted line (MLE).\n- Switch to log-likelihood to avoid underflow and show additivity.")

st.caption("Pro tip: Use 'Resample data' after changing controls to generate a fresh dataset with the current parameters.")
