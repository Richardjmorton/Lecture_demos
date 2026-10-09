import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, CubicSpline, BSpline, make_interp_spline, make_lsq_spline
from scipy.interpolate import splrep, splev
import pandas as pd

st.set_page_config(page_title="Polynomial vs Spline Demo", layout="wide")

st.title("🎯 Polynomial vs Spline Fitting: Interactive Demo")
st.markdown("""
This demo illustrates why splines are often preferred over high-degree polynomials for data fitting.
Explore different scenarios to see the issues with polynomials and the benefits of splines.
""")

# Sidebar for controls
st.sidebar.header("Controls")

# Example selection
example = st.sidebar.selectbox(
    "Select Example",
    ["Runge's Function", "Noisy Data", "Custom Points"]
)

# Number of data points
n_points = st.sidebar.slider("Number of Data Points", 5, 20, 11)

# Polynomial degree
poly_degree = st.sidebar.slider("Polynomial Degree", 1, 15, 10)

# Spline parameters
spline_type = st.sidebar.selectbox("Spline Type", ["Cubic Spline", "B-Spline", "Smoothing Spline"])
if spline_type == "Smoothing Spline":
    smoothing = st.sidebar.slider("Smoothing Factor (s)", 0.0, 10.0, 0.0, 0.1)
    st.sidebar.caption("Higher values = smoother fit")
elif spline_type == "B-Spline":
    n_knots = st.sidebar.slider("Number of Internal Knots", 0, 15, 5)
    st.sidebar.caption("More knots = more flexibility")
else:  # Cubic Spline
    st.sidebar.info("Cubic Spline interpolates through all data points")

# Add noise option
add_noise = st.sidebar.checkbox("Add Noise to Data", value=False)
if add_noise:
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1, 0.01)
else:
    noise_level = 0.0

# Generate data based on example
def runge_function(x):
    """Runge's function - classic example of polynomial interpolation problems"""
    return 1 / (1 + 25 * x**2)

def generate_data(example, n_points, noise_level):
    if example == "Runge's Function":
        x = np.linspace(-1, 1, n_points)
        y = runge_function(x)
    elif example == "Noisy Data":
        x = np.linspace(0, 10, n_points)
        y = np.sin(x) + 0.5 * np.sin(3*x)
    else:  # Custom Points
        # Generate some interesting data
        x = np.linspace(0, 2*np.pi, n_points)
        y = np.sin(x) + 0.3 * np.sin(5*x)
    
    if noise_level > 0:
        y = y + np.random.normal(0, noise_level, len(y))
    
    return x, y

# Generate data
np.random.seed(42)  # For reproducibility
x_data, y_data = generate_data(example, n_points, noise_level)

# Create fine grid for plotting
x_fine = np.linspace(x_data.min()-2, x_data.max()+2, 500)

# Fit polynomial
try:
    poly_coeffs = np.polyfit(x_data, y_data, poly_degree)
    poly_fit = np.polyval(poly_coeffs, x_fine)
    poly_success = True
except np.linalg.LinAlgError:
    poly_success = False
    st.error("Polynomial fitting failed due to numerical instability!")

# Fit spline
try:
    if spline_type == "Cubic Spline":
        cs = CubicSpline(x_data, y_data, bc_type='natural')
        spline_fit = cs(x_fine)
    elif spline_type == "B-Spline":
        # Create knot vector with internal knots
        k = 3  # cubic
        # Number of basis functions = n_knots + k + 1
        # This must be <= number of data points for least squares
        max_knots = len(x_data) - k - 1
        
        if n_knots > max_knots:
            st.warning(f"Reduced knots from {n_knots} to {max_knots} (max for {len(x_data)} data points)")
            n_knots_used = max_knots
        else:
            n_knots_used = n_knots
        
        if n_knots_used > 0:
            # Place internal knots evenly
            internal_knots = np.linspace(x_data.min(), x_data.max(), n_knots_used + 2)[1:-1]
        else:
            internal_knots = []
        
        # Create full knot vector with proper multiplicity at boundaries
        t = np.concatenate([
            [x_data.min()] * (k + 1),
            internal_knots,
            [x_data.max()] * (k + 1)
        ])
        
        # Use least-squares B-spline with custom knots
        from scipy.interpolate import make_lsq_spline
        spl = make_lsq_spline(x_data, y_data, t, k=k)
        spline_fit = spl(x_fine)
    else:  # Smoothing Spline
        spl = UnivariateSpline(x_data, y_data, s=smoothing)
        spline_fit = spl(x_fine)
    
    spline_success = True
except Exception as e:
    spline_success = False
    st.error(f"Spline fitting failed: {str(e)}")

# Create two columns for plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔴 Polynomial Fit")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    # Plot true function if available
    if example == "Runge's Function":
        x_true = np.linspace(-1, 1, 500)
        y_true = runge_function(x_true)
        ax1.plot(x_true, y_true, 'g--', linewidth=2, alpha=0.5, label='True Function')
    
    # Plot data points
    ax1.plot(x_data, y_data, 'ko', markersize=8, label='Data Points', zorder=5)
    
    # Plot polynomial fit
    if poly_success:
        ax1.plot(x_fine, poly_fit, 'r-', linewidth=2, label=f'Polynomial (degree {poly_degree})')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Polynomial Degree: {poly_degree}", fontsize=14, fontweight='bold')
    
    # Set reasonable y-limits to show oscillations
    if poly_success:
        y_range = np.ptp(y_data)
        y_center = np.mean(y_data)
        ax1.set_ylim([y_center - 3*y_range, y_center + 3*y_range])
    
    st.pyplot(fig1)
    plt.close()
    
    # Show polynomial coefficients
    if poly_success:
        with st.expander("📊 Polynomial Coefficients"):
            st.write(f"**Condition Number**: {np.linalg.cond(np.vander(x_data)):.2e}")
            st.caption("High condition numbers (>10^10) indicate numerical instability")

with col2:
    st.subheader("🔵 Spline Fit")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Plot true function if available
    if example == "Runge's Function":
        x_true = np.linspace(-1, 1, 500)
        y_true = runge_function(x_true)
        ax2.plot(x_true, y_true, 'g--', linewidth=2, alpha=0.5, label='True Function')
    
    # Plot data points
    ax2.plot(x_data, y_data, 'ko', markersize=8, label='Data Points', zorder=5)
    
    # Plot spline fit
    if spline_success:
        ax2.plot(x_fine, spline_fit, 'b-', linewidth=2, label=f'{spline_type}')
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"{spline_type}", fontsize=14, fontweight='bold')
    
    # Use same y-limits as polynomial plot for comparison
    if poly_success:
        y_range = np.ptp(y_data)
        y_center = np.mean(y_data)
        ax2.set_ylim([y_center - 3*y_range, y_center + 3*y_range])
    
    st.pyplot(fig2)
    plt.close()

# Comparison plot
st.subheader("📊 Direct Comparison")
fig3, ax3 = plt.subplots(figsize=(12, 6))

# Plot true function if available
if example == "Runge's Function":
    x_true = np.linspace(-1, 1, 500)
    y_true = runge_function(x_true)
    ax3.plot(x_true, y_true, 'g--', linewidth=2, alpha=0.5, label='True Function')

# Plot data points
ax3.plot(x_data, y_data, 'ko', markersize=10, label='Data Points', zorder=5)

# Plot both fits
if poly_success:
    ax3.plot(x_fine, poly_fit, 'r-', linewidth=2, alpha=0.7, label=f'Polynomial (degree {poly_degree})')
if spline_success:
    ax3.plot(x_fine, spline_fit, 'b-', linewidth=2, alpha=0.7, label=f'{spline_type}')

ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_title("Polynomial vs Spline: Side-by-Side Comparison", fontsize=14, fontweight='bold')

if poly_success:
    y_range = np.ptp(y_data)
    y_center = np.mean(y_data)
    ax3.set_ylim([y_center - 3*y_range, y_center + 3*y_range])

st.pyplot(fig3)
plt.close()

# Calculate and display metrics
if poly_success and spline_success:
    st.subheader("📈 Quantitative Comparison")
    
    # Interpolation error at data points
    poly_interp_error = np.mean(np.abs(np.polyval(poly_coeffs, x_data) - y_data))
    
    if spline_type == "Cubic Spline":
        spline_interp_error = np.mean(np.abs(cs(x_data) - y_data))
    elif spline_type == "B-Spline":
        spline_interp_error = np.mean(np.abs(spl(x_data) - y_data))
    else:
        spline_interp_error = np.mean(np.abs(spl(x_data) - y_data))
    
    # Display metrics
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("Polynomial MAE at Data Points", f"{poly_interp_error:.4f}")
    
    with col_b:
        st.metric("Spline MAE at Data Points", f"{spline_interp_error:.4f}")
    
    with col_c:
        # Estimate oscillation magnitude (std of second derivative as proxy)
        poly_diff2 = np.diff(poly_fit, n=2)
        spline_diff2 = np.diff(spline_fit, n=2)
        
        poly_osc = np.std(poly_diff2)
        spline_osc = np.std(spline_diff2)
        
        st.metric("Oscillation Ratio (Poly/Spline)", f"{poly_osc/spline_osc:.2f}x")

# Educational notes
st.subheader("📚 Key Observations")

with st.expander("🔍 What to look for"):
    st.markdown("""
    **Runge's Phenomenon (Polynomial Problems):**
    - Wild oscillations near the boundaries, especially with high-degree polynomials
    - Oscillations get worse as you increase the degree
    - Even with evenly-spaced data points!
    
    **Spline Benefits:**
    - Smooth interpolation without spurious oscillations
    - Local control: moving one data point doesn't affect distant regions
    - Numerically stable even with many points
    
    **Try this:**
    1. Set example to "Runge's Function"
    2. Increase polynomial degree to 15
    3. Compare with a cubic spline - notice how the spline stays well-behaved
    4. Add noise and see how the polynomial amplifies it
    """)

with st.expander("💡 Tips for your own data"):
    st.markdown("""
    **When to use splines:**
    - You need smooth interpolation between data points
    - You have noisy data (use smoothing splines)
    - You need local control over the fit
    - Numerical stability is important
    
    **Spline parameters:**
    - More knots = more flexibility (but watch for overfitting)
    - Smoothing factor: higher = smoother fit (for noisy data)
    - Cubic splines are usually a good default
    
    **For spectral analysis (like your work):**
    - Use splines when you have features at different scales
    - Place knots where the spectrum changes rapidly
    - Consider log-spacing for frequency data
    """)

# Footer
st.markdown("---")
st.caption("Adjust the controls in the sidebar to explore different scenarios. Try increasing the polynomial degree to see Runge's phenomenon in action!")