import numpy as np
from scipy import sparse
from backend.optimization_model import OptimizationModel

def test_regression_zero_row_matrix():
    A = sparse.csr_matrix([[1.0, 0.0], [0.0, 0.0]])
    b = np.array([1.0, 0.0])
    c = np.array([1.0, 1.0])
    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])
    integ = np.array([0, 0])
    model = OptimizationModel(A, b, b, c, xl, xu, integ)
    assert model.n_constraints == 2
