# app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Interactive Polynomial Fitting", layout="wide")

# ---------- Sidebar controls ----------
st.sidebar.title("Controls")

# Data controls
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
n_samples = st.sidebar.slider("Number of data points (N)", min_value=15, max_value=400, value=60, step=1)
x_range = st.sidebar.slider("X range (symmetric)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
true_deg = st.sidebar.slider("True polynomial degree (data generator)", 0, 10, 3, step=1)
noise = st.sidebar.slider("Noise σ", min_value=0.0, max_value=3.0, value=0.4, step=0.05)

# Model controls
model_deg = st.sidebar.slider("Model polynomial degree", 0, 20, 5, step=1)
use_ridge = st.sidebar.checkbox("Use Ridge regularization", value=False)
alpha = st.sidebar.slider("Ridge alpha (λ)", 1e-6, 1000.0, 1.0, step=0.1, format="%.6f", disabled=not use_ridge)

test_size = st.sidebar.slider("Test split proportion", 0.1, 0.9, 0.3, step=0.05)

# Data persistence & regeneration
st.sidebar.markdown("---")
regen = st.sidebar.button("Regenerate dataset")

# ---------- Helper functions ----------
def random_poly_coeffs(degree: int, rng: np.random.Generator) -> np.ndarray:
    """
    Create random polynomial coefficients c_0 + c_1 x + ... + c_degree x^degree,
    scaled to keep magnitudes reasonable across degrees.
    """
    if degree < 0:
        return np.array([0.0])
    # Draw coefficients and damp higher orders
    coeffs = rng.normal(0, 1.0, size=degree + 1)
    # Dampen growth of high-order terms for readability
    for k in range(2, degree + 1):
        coeffs[k] /= (k ** 1.5)
    # Ensure there's some signal
    if np.allclose(coeffs, 0):
        coeffs[0] = 1.0
    return coeffs

def eval_poly(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate polynomial given coefficients in ascending order."""
    # np.polyval expects descending order; reverse
    return np.polyval(coeffs[::-1], x)

def pretty_poly(coeffs: np.ndarray, precision=2) -> str:
    """Human-readable polynomial string."""
    terms = []
    for k, c in enumerate(coeffs):
        if abs(c) < 10**(-precision):  # skip tiny
            continue
        coef = f"{c:+.{precision}f}"
        if k == 0:
            terms.append(f"{coef}")
        elif k == 1:
            terms.append(f"{coef}·x")
        else:
            terms.append(f"{coef}·x^{k}")
    return " ".join(terms) if terms else "0"

# ---------- Generate or restore dataset ----------
if "data_state" not in st.session_state or regen:
    st.session_state.data_state = {}

rng = np.random.default_rng(seed)
state = st.session_state.data_state

# Recompute data when key-generating params change or when regen pressed
key = (seed, n_samples, x_range, true_deg, noise)
if state.get("key") != key:
    coeffs_true = random_poly_coeffs(true_deg, rng)
    X = rng.uniform(-x_range, x_range, size=n_samples)
    y_clean = eval_poly(coeffs_true, X)
    y = y_clean + rng.normal(0, noise, size=n_samples)

    state["key"] = key
    state["X"] = X
    state["y"] = y
    state["y_clean"] = y_clean
    state["coeffs_true"] = coeffs_true

X = state["X"]
y = state["y"]
y_clean = state["y_clean"]
coeffs_true = state["coeffs_true"]

# ---------- Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X.reshape(-1, 1), y, test_size=test_size, random_state=seed
)

# ---------- Fit model ----------
poly = PolynomialFeatures(degree=model_deg, include_bias=True)
Xtr_poly = poly.fit_transform(X_train)
Xte_poly = poly.transform(X_test)

model = (Ridge(alpha=alpha, fit_intercept=False) if use_ridge else
         LinearRegression(fit_intercept=False))
model.fit(Xtr_poly, y_train)

ytr_pred = model.predict(Xtr_poly)
yte_pred = model.predict(Xte_poly)

mse_train = mean_squared_error(y_train, ytr_pred)
mse_test = mean_squared_error(y_test, yte_pred)

# ---------- Dense grid for curves ----------
x_dense = np.linspace(-x_range, x_range, 600)
X_dense_poly = poly.transform(x_dense.reshape(-1, 1))
y_dense_pred = model.predict(X_dense_poly)
y_dense_true = eval_poly(coeffs_true, x_dense)

# ---------- Layout ----------
left, right = st.columns([3, 2])

with left:
    st.subheader("Data & Fits")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_train, y_train, label="Train data", alpha=0.75)
    ax.scatter(X_test, y_test, label="Test data", alpha=0.75, marker="x")
    ax.plot(x_dense, y_dense_true, label="True function", linewidth=2)
    ax.plot(x_dense, y_dense_pred, label="Model prediction", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Polynomial Regression Demo")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig, use_container_width=True)

with right:
    st.subheader("Metrics")
    st.metric("Train MSE", f"{mse_train:.4f}")
    st.metric("Test MSE", f"{mse_test:.4f}")

    st.markdown("### True polynomial")
    st.code(pretty_poly(coeffs_true, precision=3))

    # Coefficients of fitted model in polynomial basis
    st.markdown("### Fitted coefficients (model basis)")
    # Align to ascending powers like our pretty function (poly outputs include bias first)
    fitted_coeffs = model.coef_ if hasattr(model, "coef_") else np.zeros(Xtr_poly.shape[1])
    # sklearn returns shape (n_features,) for coef_ and intercept_ separately; we set fit_intercept=False
    # So bias term is part of coef_
    powers = poly.powers_.sum(axis=1)  # total power per feature
    # Sort features by power (should already be ascending)
    lines = []
    for p, c in zip(powers, fitted_coeffs):
        lines.append(f"x^{int(p)} : {c:+.5f}")
    st.code("\n".join(lines))

# ---------- Tips ----------
with st.expander("What to try"):
    st.markdown(
        """
- Increase **Model degree** beyond the **True degree** to see **overfitting** (train MSE ↓, test MSE ↑).
- Crank up **Noise σ** and watch generalization degrade.
- Toggle **Ridge** and sweep **alpha (λ)** to stabilize high-degree fits.
- Reduce **N** to see variance explode on complex models.
"""
    )
