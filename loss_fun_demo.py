import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

st.set_page_config(page_title="Loss Function Demo", layout="wide")

st.title("What you optimize is what you get.")
st.markdown("**Generate synthetic data and see how different loss functions fit the data!**")

# Initialize session state
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False

# True line parameters
TRUE_SLOPE = 1.2
TRUE_INTERCEPT = 0.5

def generate_data(n_points, noise_type, seed):
    """Generate synthetic data with known true line"""
    np.random.seed(seed)
    x = np.linspace(0, 10, n_points)
    y_true = TRUE_SLOPE * x + TRUE_INTERCEPT
    
    if noise_type == "Normal":
        noise = np.random.normal(0, 1.0, n_points)
    else:  # Student-t
        noise = np.random.standard_t(df=3, size=n_points)
    
    y = y_true + noise
    return x, y, y_true

# Sidebar controls
st.sidebar.header("Data Generation")

# Number of data points slider
n_points = st.sidebar.slider("Number of data points", 5, 50, 20, 1)

# Noise type dropdown
noise_type = st.sidebar.selectbox("Noise distribution", ["Normal", "Student-t (heavy tails)"])

# Random seed controls
col1, col2 = st.sidebar.columns([2, 1])
with col1:
    if st.sidebar.button("New Random Seed", width='stretch'):
        st.session_state.random_seed = np.random.randint(0, 10000)
        st.session_state.data_generated = False
        st.rerun()
with col2:
    st.sidebar.text(f"Seed: {st.session_state.random_seed}")

# Generate data button
if st.sidebar.button("Generate Data", width='stretch') or not st.session_state.data_generated:
    x_data, y_data, y_true = generate_data(n_points, noise_type, st.session_state.random_seed)
    st.session_state.x_data = x_data
    st.session_state.y_data = y_data
    st.session_state.y_true = y_true
    st.session_state.data_generated = True
    st.rerun()

# Add outliers button
col1, col2 = st.sidebar.columns(2)
if col1.button("Add Outlier"):
   # Random x position
    new_x = np.random.uniform(st.session_state.x_data.min(), st.session_state.x_data.max())
    # True y value at this position
    y_true_at_x = TRUE_SLOPE * new_x + TRUE_INTERCEPT
    # Random large offset (positive or negative with equal probability)
    y_range = st.session_state.y_data.max() - st.session_state.y_data.min()
    mag = np.random.uniform(0.6,1.2)
    offset_magnitude = y_range * mag
    offset_sign = np.random.choice([-1, 1])  # 50% chance positive or negative
    new_y = y_true_at_x + offset_sign * offset_magnitude
    
    st.session_state.x_data = np.append(st.session_state.x_data, new_x)
    st.session_state.y_data = np.append(st.session_state.y_data, new_y)
    st.session_state.y_true = np.append(st.session_state.y_true, y_true_at_x)
    st.rerun()


if col2.button("Remove Last"):
    if len(st.session_state.x_data) > 0:
        st.session_state.x_data = st.session_state.x_data[:-1]
        st.session_state.y_data = st.session_state.y_data[:-1]
        st.session_state.y_true = st.session_state.y_true[:-1]
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("Loss Functions")

# Loss function selection
show_l2 = st.sidebar.checkbox("L2 (Squared Loss)", value=True)
show_l1 = st.sidebar.checkbox("L1 (Absolute Loss)", value=True)
show_huber = st.sidebar.checkbox("Huber Loss", value=True)
show_quantile = st.sidebar.checkbox("Quantile (10th percentile)", value=False)
show_true_line = st.sidebar.checkbox("Show True Line", value=True)

# Huber delta parameter
if show_huber:
    huber_delta = st.sidebar.slider("Huber delta", 0.1, 5.0, 1.0, 0.1)

# Define loss functions
def predict(params, x):
    """Linear model: y = mx + b"""
    m, b = params
    return m * x + b

def l2_loss(params, x, y):
    """Squared loss"""
    pred = predict(params, x)
    return np.sum((y - pred)**2)

def l1_loss(params, x, y):
    """Absolute loss"""
    pred = predict(params, x)
    return np.sum(np.abs(y - pred))

def huber_loss(params, x, y, delta=1.0):
    """Huber loss"""
    pred = predict(params, x)
    residual = np.abs(y - pred)
    return np.sum(np.where(
        residual <= delta,
        0.5 * residual**2,
        delta * (residual - 0.5 * delta)
    ))

def quantile_loss(params, x, y, tau=0.1):
    """Quantile loss"""
    pred = predict(params, x)
    residual = y - pred
    return np.sum(np.where(residual >= 0, tau * residual, (tau - 1) * residual))

# Fit models
def fit_model(loss_fn, x, y, loss_name):
    """Fit linear model using given loss function"""
    if len(x) < 2:
        return None, None
    
    # Initial guess from simple linear regression
    if len(x) > 1:
        coeffs = np.polyfit(x, y, 1)
        init_params = [coeffs[0], coeffs[1]]
    else:
        init_params = [1.0, 0.0]
    
    # Use different methods for different loss functions
    # L1 and quantile are not smooth, so use derivative-free methods
    if loss_name in ["L1", "Quantile"]:
        method = 'Nelder-Mead'
        options = {'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
    else:
        method = 'BFGS'
        options = {'maxiter': 1000}
    
    # Optimize
    result = minimize(loss_fn, init_params, args=(x, y), method=method, options=options)
    
    if result.success or result.fun < np.inf:
        return result.x, result.fun
    else:
        # Try Powell method as backup
        result = minimize(loss_fn, init_params, args=(x, y), method='Powell', 
                         options={'maxiter': 2000})
        if result.success or result.fun < np.inf:
            return result.x, result.fun
        return None, None

# Main plotting area
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    if len(st.session_state.x_data) > 0:
        ax.scatter(st.session_state.x_data, st.session_state.y_data, 
                   s=100, c='black', alpha=0.6, label='Data', zorder=5)
        
        # Fit and plot models
        x_line = np.linspace(st.session_state.x_data.min() - 1, 
                            st.session_state.x_data.max() + 1, 100)
        
        # Show true line if toggled
        if show_true_line:
            y_true_line = TRUE_SLOPE * x_line + TRUE_INTERCEPT
            ax.plot(x_line, y_true_line, 'k--', linewidth=2.5, 
                   label=f'True Line (y={TRUE_SLOPE}x+{TRUE_INTERCEPT})', 
                   alpha=0.4, zorder=3)
        
        results = []
        
        if show_l2:
            params, loss_val = fit_model(l2_loss, st.session_state.x_data, 
                                        st.session_state.y_data, "L2")
            if params is not None:
                y_line = predict(params, x_line)
                ax.plot(x_line, y_line, 'r-', linewidth=2, label='L2 (Squared)', alpha=0.8)
                results.append({"Loss": "L2", "Slope": f"{params[0]:.3f}", 
                               "Intercept": f"{params[1]:.3f}", "Loss Value": f"{loss_val:.3f}"})
        
        if show_l1:
            params, loss_val = fit_model(l1_loss, st.session_state.x_data, 
                                        st.session_state.y_data, "L1")
            if params is not None:
                y_line = predict(params, x_line)
                ax.plot(x_line, y_line, 'b-', linewidth=2, label='L1 (Absolute)', alpha=0.8)
                results.append({"Loss": "L1", "Slope": f"{params[0]:.3f}", 
                               "Intercept": f"{params[1]:.3f}", "Loss Value": f"{loss_val:.3f}"})
        
        if show_huber:
            params, loss_val = fit_model(lambda p, x, y: huber_loss(p, x, y, huber_delta),
                                        st.session_state.x_data, 
                                        st.session_state.y_data, "Huber")
            if params is not None:
                y_line = predict(params, x_line)
                ax.plot(x_line, y_line, 'g-', linewidth=2, label='Huber', alpha=0.8)
                results.append({"Loss": "Huber", "Slope": f"{params[0]:.3f}", 
                               "Intercept": f"{params[1]:.3f}", "Loss Value": f"{loss_val:.3f}"})
        
        if show_quantile:
            params, loss_val = fit_model(lambda p, x, y: quantile_loss(p, x, y, 0.1),
                                        st.session_state.x_data, 
                                        st.session_state.y_data, "Quantile")
            if params is not None:
                y_line = predict(params, x_line)
                ax.plot(x_line, y_line, 'm--', linewidth=2, label='Quantile (10%)', alpha=0.8)
                results.append({"Loss": "Quantile", "Slope": f"{params[0]:.3f}", 
                               "Intercept": f"{params[1]:.3f}", "Loss Value": f"{loss_val:.3f}"})
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Fitted Lines vs Data', fontsize=14, pad=20)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Display fitted parameters
    if len(st.session_state.x_data) > 0 and len(results) > 0:
        st.markdown("### Fitted Parameters")
        df_results = pd.DataFrame(results)
        # Add comparison to true values
        df_results['Slope Error'] = (df_results['Slope'].astype(float) - TRUE_SLOPE).abs()
        df_results['Slope Error'] = df_results['Slope Error'].apply(lambda x: f"{x:.3f}")
        st.dataframe(df_results, hide_index=True, width='stretch')
        st.caption(f"True slope: {TRUE_SLOPE}, True intercept: {TRUE_INTERCEPT}")

with col2:
    st.markdown("### Loss Function Shapes")
    st.markdown("How each loss penalizes residuals:")
    
    # Plot loss function shapes
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    residuals = np.linspace(-3, 3, 200)
    
    if show_l2:
        ax2.plot(residuals, residuals**2, 'r-', linewidth=2, label='L2: r²', alpha=0.8)
    if show_l1:
        ax2.plot(residuals, np.abs(residuals), 'b-', linewidth=2, label='L1: |r|', alpha=0.8)
    if show_huber:
        huber_vals = np.where(
            np.abs(residuals) <= huber_delta,
            0.5 * residuals**2,
            huber_delta * (np.abs(residuals) - 0.5 * huber_delta)
        )
        ax2.plot(residuals, huber_vals, 'g-', linewidth=2, label=f'Huber (δ={huber_delta})', alpha=0.8)
    if show_quantile:
        quantile_vals = np.where(residuals >= 0, 0.1 * residuals, -0.9 * residuals)
        ax2.plot(residuals, quantile_vals, 'm--', linewidth=2, label='Quantile (τ=0.1)', alpha=0.8)
    
    ax2.set_xlabel('Residual (y - ŷ)', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.set_title('Loss vs Residual', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 6)
    
    st.pyplot(fig2)
    
    # Explanations
    st.markdown("### Key Insights")
    st.markdown("""
    - **L2 (red)**: Heavily penalizes large errors → pulled by outliers
    - **L1 (blue)**: Linear penalty → more robust to outliers
    - **Huber (green)**: L2 for small errors, L1 for large → compromise
    - **Quantile (magenta)**: Asymmetric penalty → fits below/above data
    """)

# Instructions
st.markdown("---")
st.markdown("""
### Instructions
1. **Set number of data points** using the slider (5-50)
2. **Choose noise distribution**: Normal (light tails) or Student-t (heavy tails)
3. **Click "Generate Data"** to create synthetic data with known true line
4. **Click "New Random Seed"** to try different random realizations
5. **Add Outliers** to see how L2 gets pulled while L1 stays robust
6. **Toggle loss functions** to compare fitted lines
7. **Adjust Huber delta** to see the transition between L2 and L1 behavior
8. **Try Quantile loss** to fit the 10th percentile (line below the data)
9. **Show/Hide True Line** to see how close each loss gets to the ground truth

**Teaching Point**: What you optimize is what you get! Each loss function has a different notion of "best fit."
- With **Normal noise**, L2 performs best (maximum likelihood under Gaussian assumption)
- With **Student-t noise** (heavy tails), L1 and Huber are more robust
- **Quantile regression** fits a specific percentile, not the mean
""")

st.info("💡 Try Student-t noise with several outliers and watch how L2 (squared loss) gets pulled toward them while L1 (absolute loss) remains more stable and closer to the true line!")