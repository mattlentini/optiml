"""
OptiML Web Server - Flask API backend
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
import os

from optiml import BayesianOptimizer, Space, Real, Integer, Categorical
from optiml.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement

app = Flask(__name__, static_folder='static')
CORS(app)

# Store optimizer state
optimizers = {}

# Test functions
TEST_FUNCTIONS = {
    'quadratic': {
        'name': 'Quadratic (2D)',
        'description': 'Simple bowl function. Optimum at (2, 3)',
        'space': [
            {'type': 'real', 'name': 'x', 'low': -5, 'high': 10},
            {'type': 'real', 'name': 'y', 'low': -5, 'high': 10}
        ],
        'func': lambda p: -(p[0] - 2)**2 - (p[1] - 3)**2
    },
    'rosenbrock': {
        'name': 'Rosenbrock (2D)',
        'description': 'Classic banana function. Optimum at (1, 1)',
        'space': [
            {'type': 'real', 'name': 'x', 'low': -2, 'high': 2},
            {'type': 'real', 'name': 'y', 'low': -1, 'high': 3}
        ],
        'func': lambda p: -((1 - p[0])**2 + 100 * (p[1] - p[0]**2)**2)
    },
    'rastrigin': {
        'name': 'Rastrigin (2D)',
        'description': 'Multi-modal with many local optima. Optimum at (0, 0)',
        'space': [
            {'type': 'real', 'name': 'x', 'low': -5.12, 'high': 5.12},
            {'type': 'real', 'name': 'y', 'low': -5.12, 'high': 5.12}
        ],
        'func': lambda p: -(20 + (p[0]**2 - 10*np.cos(2*np.pi*p[0])) + (p[1]**2 - 10*np.cos(2*np.pi*p[1])))
    },
    'ackley': {
        'name': 'Ackley (2D)',
        'description': 'Complex surface with global minimum. Optimum at (0, 0)',
        'space': [
            {'type': 'real', 'name': 'x', 'low': -5, 'high': 5},
            {'type': 'real', 'name': 'y', 'low': -5, 'high': 5}
        ],
        'func': lambda p: -(-20*np.exp(-0.2*np.sqrt(0.5*(p[0]**2+p[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*p[0])+np.cos(2*np.pi*p[1]))) + np.e + 20)
    },
    'himmelblau': {
        'name': 'Himmelblau (2D)',
        'description': 'Four identical local minima. Multiple optima exist',
        'space': [
            {'type': 'real', 'name': 'x', 'low': -5, 'high': 5},
            {'type': 'real', 'name': 'y', 'low': -5, 'high': 5}
        ],
        'func': lambda p: -((p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2)
    }
}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/functions', methods=['GET'])
def get_functions():
    """Get available test functions."""
    return jsonify({
        name: {'name': f['name'], 'description': f['description'], 'space': f['space']}
        for name, f in TEST_FUNCTIONS.items()
    })

@app.route('/api/optimize', methods=['POST'])
def optimize():
    """Run full optimization."""
    data = request.json
    
    func_name = data.get('function', 'quadratic')
    n_iterations = data.get('iterations', 30)
    n_initial = data.get('initial', 5)
    maximize = data.get('maximize', True)
    acquisition = data.get('acquisition', 'ei')
    seed = data.get('seed', 42)
    
    # Get function and create space
    func_data = TEST_FUNCTIONS[func_name]
    dimensions = []
    for dim in func_data['space']:
        if dim['type'] == 'real':
            dimensions.append(Real(dim['low'], dim['high'], name=dim['name']))
        elif dim['type'] == 'integer':
            dimensions.append(Integer(dim['low'], dim['high'], name=dim['name']))
    
    space = Space(dimensions)
    
    # Create acquisition function
    if acquisition == 'ei':
        acq = ExpectedImprovement(xi=0.01)
    elif acquisition == 'ucb':
        acq = UpperConfidenceBound(kappa=2.0)
    else:
        acq = ProbabilityOfImprovement(xi=0.01)
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        space,
        acquisition=acq,
        n_initial=n_initial,
        maximize=maximize,
        random_state=seed
    )
    
    # Run optimization
    objective = func_data['func']
    history = []
    best_value = float('-inf') if maximize else float('inf')
    
    for i in range(n_iterations):
        params = optimizer.suggest()
        value = float(objective(params))
        optimizer.tell(params, value)
        
        if (maximize and value > best_value) or (not maximize and value < best_value):
            best_value = value
        
        history.append({
            'iteration': i + 1,
            'params': [float(p) if isinstance(p, (np.floating, float)) else int(p) for p in params],
            'value': value,
            'best': best_value
        })
    
    result = optimizer.get_result()
    
    return jsonify({
        'success': True,
        'history': history,
        'best': {
            'params': [float(p) if isinstance(p, (np.floating, float)) else int(p) for p in result.x_best],
            'value': float(result.y_best)
        },
        'paramNames': [d.name for d in space.dimensions]
    })

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    print("\n🎯 OptiML Server running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
