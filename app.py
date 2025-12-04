"""
OptiML - Interactive Bayesian Optimization Dashboard
=====================================================
A beautiful, easy-to-use UI for Bayesian optimization.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from optiml import BayesianOptimizer, Space, Real, Integer, Categorical
from optiml.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement

# Page configuration
st.set_page_config(
    page_title="OptiML - Bayesian Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 0.5rem;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎯 OptiML</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Bayesian Optimization Made Simple</p>', unsafe_allow_html=True)

# Initialize session state
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'running' not in st.session_state:
    st.session_state.running = False

# Sidebar configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Problem type
    problem_type = st.selectbox(
        "🧪 Problem Type",
        ["Built-in Test Function", "Custom Expression"],
        help="Choose a built-in test function or define your own"
    )
    
    if problem_type == "Built-in Test Function":
        test_function = st.selectbox(
            "📊 Test Function",
            ["Quadratic (2D)", "Rosenbrock (2D)", "Rastrigin (2D)", "Ackley (2D)", "Sphere (3D)"],
            help="Select a standard optimization test function"
        )
    else:
        custom_expr = st.text_area(
            "📝 Custom Expression",
            value="-(x1 - 2)**2 - (x2 - 3)**2",
            help="Use x1, x2, x3... for variables. Use numpy functions with 'np.' prefix."
        )
    
    st.markdown("---")
    st.markdown("### 🔧 Optimization Settings")
    
    n_iterations = st.slider("Number of Iterations", 10, 100, 30, 5)
    n_initial = st.slider("Initial Random Samples", 3, 20, 5)
    
    maximize = st.toggle("Maximize (vs Minimize)", value=True)
    
    acquisition_fn = st.selectbox(
        "🎲 Acquisition Function",
        ["Expected Improvement", "Upper Confidence Bound", "Probability of Improvement"]
    )
    
    if acquisition_fn == "Expected Improvement":
        xi = st.slider("Exploration (ξ)", 0.0, 0.5, 0.01, 0.01)
    elif acquisition_fn == "Upper Confidence Bound":
        kappa = st.slider("Exploration (κ)", 0.1, 5.0, 2.0, 0.1)
    
    random_seed = st.number_input("Random Seed", 0, 9999, 42)

# Define test functions
def get_test_function(name):
    """Get test function and its search space."""
    if name == "Quadratic (2D)":
        def func(params):
            x, y = params
            return -(x - 2)**2 - (y - 3)**2
        space = Space([Real(-5, 10, name="x"), Real(-5, 10, name="y")])
        true_opt = (2, 3, 0) if True else None
        return func, space, "Maximum at (2, 3) with value 0"
    
    elif name == "Rosenbrock (2D)":
        def func(params):
            x, y = params
            return -((1 - x)**2 + 100 * (y - x**2)**2)
        space = Space([Real(-2, 2, name="x"), Real(-1, 3, name="y")])
        return func, space, "Maximum at (1, 1) with value 0"
    
    elif name == "Rastrigin (2D)":
        def func(params):
            x, y = params
            A = 10
            return -(A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y)))
        space = Space([Real(-5.12, 5.12, name="x"), Real(-5.12, 5.12, name="y")])
        return func, space, "Maximum at (0, 0) with value 0"
    
    elif name == "Ackley (2D)":
        def func(params):
            x, y = params
            a, b, c = 20, 0.2, 2 * np.pi
            sum1 = x**2 + y**2
            sum2 = np.cos(c * x) + np.cos(c * y)
            return -(-a * np.exp(-b * np.sqrt(sum1 / 2)) - np.exp(sum2 / 2) + a + np.e)
        space = Space([Real(-5, 5, name="x"), Real(-5, 5, name="y")])
        return func, space, "Maximum at (0, 0) with value 0"
    
    elif name == "Sphere (3D)":
        def func(params):
            return -sum(p**2 for p in params)
        space = Space([Real(-5, 5, name="x"), Real(-5, 5, name="y"), Real(-5, 5, name="z")])
        return func, space, "Maximum at (0, 0, 0) with value 0"

def get_custom_function(expr, n_dims=2):
    """Create function from custom expression."""
    def func(params):
        local_vars = {'np': np}
        for i, p in enumerate(params):
            local_vars[f'x{i+1}'] = p
        return eval(expr, {"__builtins__": {}}, local_vars)
    
    space = Space([Real(-10, 10, name=f"x{i+1}") for i in range(n_dims)])
    return func, space, f"Custom: {expr}"

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🚀 Run Optimization")
    
    if st.button("▶️ Start Optimization", use_container_width=True):
        # Get function and space
        if problem_type == "Built-in Test Function":
            objective, space, description = get_test_function(test_function)
        else:
            objective, space, description = get_custom_function(custom_expr)
        
        # Create acquisition function
        if acquisition_fn == "Expected Improvement":
            acq = ExpectedImprovement(xi=xi)
        elif acquisition_fn == "Upper Confidence Bound":
            acq = UpperConfidenceBound(kappa=kappa)
        else:
            acq = ProbabilityOfImprovement(xi=0.01)
        
        # Create optimizer
        optimizer = BayesianOptimizer(
            space,
            acquisition=acq,
            n_initial=n_initial,
            maximize=maximize,
            random_state=int(random_seed)
        )
        
        # Progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        history = []
        best_so_far = float('-inf') if maximize else float('inf')
        
        for i in range(n_iterations):
            params = optimizer.suggest()
            value = objective(params)
            optimizer.tell(params, value)
            
            if (maximize and value > best_so_far) or (not maximize and value < best_so_far):
                best_so_far = value
            
            history.append({
                'iteration': i + 1,
                'params': params,
                'value': value,
                'best': best_so_far
            })
            
            progress_bar.progress((i + 1) / n_iterations)
            status_text.text(f"Iteration {i+1}/{n_iterations} | Current: {value:.6f} | Best: {best_so_far:.6f}")
        
        st.session_state.optimizer = optimizer
        st.session_state.history = history
        st.session_state.result = optimizer.get_result()
        st.session_state.space = space
        st.session_state.description = description
        
        status_text.empty()
        progress_bar.empty()
        st.success("✅ Optimization complete!")
        st.rerun()

with col2:
    if st.session_state.history:
        result = st.session_state.result
        st.markdown("### 🏆 Best Result")
        
        st.metric(
            label="Best Value",
            value=f"{result.y_best:.6f}",
            delta=f"Found in {result.n_iterations} iterations"
        )
        
        st.markdown("**Best Parameters:**")
        for i, (dim, val) in enumerate(zip(st.session_state.space.dimensions, result.x_best)):
            name = dim.name or f"x{i+1}"
            if isinstance(val, (int, np.integer)):
                st.code(f"{name} = {val}")
            else:
                st.code(f"{name} = {val:.6f}")

# Visualization section
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📈 Optimization Progress")
    
    tab1, tab2, tab3 = st.tabs(["📊 Convergence", "🗺️ Search Space", "📋 History"])
    
    with tab1:
        df = pd.DataFrame(st.session_state.history)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Objective Value per Iteration", "Best Value Found"),
            horizontal_spacing=0.1
        )
        
        # Scatter plot of all evaluations
        fig.add_trace(
            go.Scatter(
                x=df['iteration'],
                y=df['value'],
                mode='markers',
                name='Evaluations',
                marker=dict(
                    size=10,
                    color=df['value'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Value", x=0.45)
                )
            ),
            row=1, col=1
        )
        
        # Best value convergence
        fig.add_trace(
            go.Scatter(
                x=df['iteration'],
                y=df['best'],
                mode='lines+markers',
                name='Best So Far',
                line=dict(color='#764ba2', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=1, col=2)
        fig.update_yaxes(title_text="Objective Value", row=1, col=1)
        fig.update_yaxes(title_text="Best Value", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        space = st.session_state.space
        if len(space) == 2:
            # 2D visualization
            df = pd.DataFrame(st.session_state.history)
            x_vals = [h['params'][0] for h in st.session_state.history]
            y_vals = [h['params'][1] for h in st.session_state.history]
            values = [h['value'] for h in st.session_state.history]
            iterations = [h['iteration'] for h in st.session_state.history]
            
            fig = go.Figure()
            
            # Plot all points
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=iterations,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Iteration")
                ),
                text=[str(i) for i in iterations],
                textposition="top center",
                textfont=dict(size=8),
                hovertemplate="<b>Iteration %{text}</b><br>" +
                              f"{space[0].name or 'x1'}: %{{x:.4f}}<br>" +
                              f"{space[1].name or 'x2'}: %{{y:.4f}}<br>" +
                              "Value: %{customdata:.4f}<extra></extra>",
                customdata=values
            ))
            
            # Highlight best point
            result = st.session_state.result
            fig.add_trace(go.Scatter(
                x=[result.x_best[0]],
                y=[result.x_best[1]],
                mode='markers',
                marker=dict(size=25, color='red', symbol='star'),
                name='Best',
                hovertemplate=f"<b>Best Found</b><br>Value: {result.y_best:.4f}<extra></extra>"
            ))
            
            fig.update_layout(
                title="Search Space Exploration",
                xaxis_title=space[0].name or "x1",
                yaxis_title=space[1].name or "x2",
                height=500,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif len(space) == 3:
            # 3D visualization
            x_vals = [h['params'][0] for h in st.session_state.history]
            y_vals = [h['params'][1] for h in st.session_state.history]
            z_vals = [h['params'][2] for h in st.session_state.history]
            values = [h['value'] for h in st.session_state.history]
            
            fig = go.Figure(data=[go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=8,
                    color=values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Value")
                )
            )])
            
            fig.update_layout(
                title="3D Search Space",
                scene=dict(
                    xaxis_title=space[0].name or "x1",
                    yaxis_title=space[1].name or "x2",
                    zaxis_title=space[2].name or "x3"
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Search space visualization available for 2D and 3D problems.")
    
    with tab3:
        df = pd.DataFrame([
            {
                'Iteration': h['iteration'],
                **{st.session_state.space[i].name or f'x{i+1}': 
                   f"{h['params'][i]:.6f}" if isinstance(h['params'][i], float) else str(h['params'][i])
                   for i in range(len(h['params']))},
                'Value': f"{h['value']:.6f}",
                'Best': f"{h['best']:.6f}"
            }
            for h in st.session_state.history
        ])
        
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="optimization_results.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎯 <strong>OptiML</strong> - Advanced Bayesian Optimization</p>
    <p style="font-size: 0.9rem;">Built with ❤️ using Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)
