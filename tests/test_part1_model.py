import pytest
import numpy as np
from scipy import sparse
from backend.optimization_model import OptimizationModel

def test_valid_model_creation():
    A = sparse.csr_matrix([[1.0, 2.0], [3.0, 4.0]])
    b_l = np.array([0.0, 0.0])
    b_u = np.array([10.0, 20.0])
    c = np.array([1.0, 2.0])
    x_l = np.array([0.0, 0.0])
    x_u = np.array([5.0, 5.0])
    integ = np.array([0, 0])
    
    model = OptimizationModel(A, b_l, b_u, c, x_l, x_u, integ)
    assert model.n_variables == 2
    assert model.n_constraints == 2
    assert model.nnz == 4
    assert model.problem_type == "LP"

def test_invalid_dimension_mismatch():
    A = sparse.csr_matrix([[1.0, 2.0]])
    b_l = np.array([0.0])
    b_u = np.array([10.0])
    c = np.array([1.0, 2.0, 3.0])  # Mismatch: 3 items for 2 variables
    x_l = np.array([0.0, 0.0])
    x_u = np.array([5.0, 5.0])
    integ = np.array([0, 0])
    
    with pytest.raises((ValueError, AssertionError)):
        m = OptimizationModel(A, b_l, b_u, c, x_l, x_u, integ)
        m.validate()

def test_invalid_bounds_lower_greater_than_upper():
    A = sparse.eye(2, format="csr")
    b_l = np.array([10.0, 0.0])
    b_u = np.array([5.0, 0.0])  # Invalid: 10.0 > 5.0
    c = np.array([1.0, 1.0])
    x_l = np.array([0.0, 0.0])
    x_u = np.array([5.0, 5.0])
    integ = np.array([0, 0])
    
    with pytest.raises((ValueError, AssertionError)):
        m = OptimizationModel(A, b_l, b_u, c, x_l, x_u, integ)
        m.validate()
