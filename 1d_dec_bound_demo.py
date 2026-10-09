import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="1D Decision Boundary Demo", layout="wide")

st.title("Interactive 1D Decision Boundary (Two Gaussians)")
st.caption("Tweak means, variances, and class priors. See how the Bayes-optimal boundary moves and how the Bayes error changes.")

# --- Controls ---
with st.sidebar:
    st.header("Class 1 (blue)")
    mu1 = st.slider("Mean μ₁", -10.0, 10.0, -1.0, 0.1)
    sigma1 = st.slider("Std σ₁", 0.05, 5.0, 1.0, 0.05)

    st.header("Class 2 (orange)")
    mu2 = st.slider("Mean μ₂", -10.0, 10.0, 1.0, 0.1)
    sigma2 = st.slider("Std σ₂", 0.05, 5.0, 1.0, 0.05)

    st.header("Priors")
    pi1 = st.slider("π₁ (Class 1 prior)", 0.0, 1.0, 0.5, 0.01)
    pi2 = 1.0 - pi1
    st.text(f"π₂ (Class 2 prior): {pi2:.2f}")

    st.header("Options")
    show_samples = st.checkbox("Draw random samples", value=False)
    n_samples = st.slider("Samples per refresh", 50, 2000, 400, 50, disabled=not show_samples)
    show_background = st.checkbox("Color classification regions", value=True)

# --- Utilities ---

def normal_pdf(x, mu, sigma):
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Discriminant: g(x) = log ( π₁ f₁(x) / (π₂ f₂(x)) )
# Decision: Class 1 if g(x) > 0 else Class 2.

def g_log_ratio(x):
    f1 = normal_pdf(x, mu1, sigma1)
    f2 = normal_pdf(x, mu2, sigma2)
    # avoid divide-by-zero by adding tiny epsilon
    eps = 1e-300
    return np.log((pi1 * (f1 + eps)) / (pi2 * (f2 + eps)))

# Closed-form roots for Gaussians (may yield 0, 1, or 2 real roots)
# Derived from setting g(x)=0 and expanding quadratics.

def decision_boundaries():
    a = 1.0 / (2.0 * sigma2 ** 2) - 1.0 / (2.0 * sigma1 ** 2)
    b = (mu1 / (sigma1 ** 2)) - (mu2 / (sigma2 ** 2))
    c = (mu2 ** 2) / (2.0 * sigma2 ** 2) - (mu1 ** 2) / (2.0 * sigma1 ** 2) \
        + 0.5 * np.log((sigma2 ** 2) / (sigma1 ** 2)) + np.log(pi2 / pi1)

    roots = []
    if abs(a) < 1e-14:
        # Linear case
        if abs(b) > 1e-14:
            roots = [ -c / b ]
        else:
            roots = []  # Degenerate; distributions identical up to scaling of priors
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0:
            r1 = (-b - np.sqrt(disc)) / (2.0 * a)
            r2 = (-b + np.sqrt(disc)) / (2.0 * a)
            roots = [r1, r2]
    # Filter unique & sorted
    roots = sorted(list({round(float(r), 12) for r in roots}))
    return roots

# Domain to plot: expand around means by a few stds
x_min = min(mu1 - 5 * sigma1, mu2 - 5 * sigma2) - 1.0
x_max = max(mu1 + 5 * sigma1, mu2 + 5 * sigma2) + 1.0
x = np.linspace(x_min, x_max, 2000)

f1 = normal_pdf(x, mu1, sigma1)
f2 = normal_pdf(x, mu2, sigma2)
post_ratio = g_log_ratio(x)

# Bayes error (integral of min-class posterior numerator)
# P(error) = ∫ min(π₁ f₁(x), π₂ f₂(x)) dx
bayes_error = np.trapz(np.minimum(pi1 * f1, pi2 * f2), x)

# Find decision boundary crossings numerically (fallback)
roots_cf = decision_boundaries()

# Determine predicted class over grid
pred_class1 = post_ratio > 0

# --- Plotting ---
fig, ax = plt.subplots(figsize=(9, 5))

# Optional background classification regions
if show_background:
    # color where Class 1 is chosen
    ax.fill_between(x, 0, np.maximum(f1, f2), where=pred_class1, alpha=0.15)
    ax.fill_between(x, 0, np.maximum(f1, f2), where=~pred_class1, alpha=0.15)

# Plot PDFs
ax.plot(x, f1, label="Class 1 PDF")
ax.plot(x, f2, label="Class 2 PDF")

# Vertical lines for decision boundaries
for r in roots_cf:
    if x_min <= r <= x_max:
        ax.axvline(r, linestyle="--")
        ax.text(r, ax.get_ylim()[1] * 0.9, f"x = {r:.3f}", rotation=90, va="top", ha="right")

ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title("Class-Conditional PDFs and Decision Boundary")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig, use_container_width=True)

# --- Right column: metrics & samples ---
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Numbers that matter")
    st.metric("Bayes Error (theoretical lower bound)", f"{bayes_error:.4f}")
    if roots_cf:
        st.write("**Decision boundary roots (g(x)=0):** ", ", ".join(f"{r:.6f}" for r in roots_cf))
    else:
        st.write("**Decision boundary roots (g(x)=0):** none in closed form (or degenerate)")

with col2:
    if show_samples:
        # Draw samples according to priors
        n1 = np.random.binomial(n_samples, pi1)
        n2 = n_samples - n1
        s1 = np.random.normal(mu1, sigma1, size=n1)
        s2 = np.random.normal(mu2, sigma2, size=n2)
        samples = np.concatenate([s1, s2])
        labels = np.array([1] * n1 + [2] * n2)
        # Predict using current boundary
        def predict(xs):
            return (g_log_ratio(xs) > 0).astype(int) + 1
        preds = predict(samples)
        acc = np.mean(preds == labels)

        # Scatter plot on top of PDFs
        fig2, ax2 = plt.subplots(figsize=(9, 3))
        ax2.plot(x, f1, label="Class 1 PDF")
        ax2.plot(x, f2, label="Class 2 PDF")
        # Jittered y for visibility
        y_scat = np.zeros_like(samples) + max(f1.max(), f2.max()) * 0.05
        ax2.scatter(samples[labels==1], y_scat[labels==1], marker="o", s=12, label="Samples C1")
        ax2.scatter(samples[labels==2], y_scat[labels==2], marker="x", s=16, label="Samples C2")
        for r in roots_cf:
            if x_min <= r <= x_max:
                ax2.axvline(r, linestyle="--")
        ax2.set_xlabel("x")
        ax2.set_ylabel("Density / markers")
        ax2.set_title(f"Random samples (n={n_samples}) • Empirical accuracy: {acc:.3f}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, use_container_width=True)
    else:
        st.info("Enable 'Draw random samples' in the sidebar to visualize empirical accuracy vs the Bayes error.")

st.markdown("---")
with st.expander("What am I looking at? (quick refresher)"):
    st.write(
        r"""
        **Bayes decision rule** chooses Class 1 when \( \log \frac{\pi_1 f_1(x)}{\pi_2 f_2(x)} > 0 \) and Class 2 otherwise.
        For two Gaussians, this yields up to two decision boundaries (a quadratic equation in \(x\)) when variances differ, and a single linear boundary when variances are equal.

        The **Bayes error** shown is the theoretical minimum error achievable by any classifier given the model: \(\int \min(\pi_1 f_1(x), \pi_2 f_2(x))\,dx\).
        """
    )

st.caption("Tip: set σ₁ = σ₂ to see the boundary snap to the midpoint shifted by the log-prior term.")
