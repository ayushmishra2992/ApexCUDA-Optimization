import pytest
import numpy as np
from scipy import sparse
from backend.optimization_model import OptimizationModel
from backend.indioptima import solve

def test_dispatcher_routes_lp():
    A = sparse.eye(2, format="csr")
    b = np.array([1.0, 1.0])
    c = np.array([1.0, 1.0])
    xl = np.array([0.0, 0.0])
    xu = np.array([5.0, 5.0])
    integ = np.array([0, 0])
    lp = OptimizationModel(A, b, b, c, xl, xu, integ)
    res = solve(lp, backend="cpu")
    assert res is not None

def test_dispatcher_rejects_invalid_backend():
    A = sparse.eye(2, format="csr")
    b = np.array([1.0, 1.0])
    c = np.array([1.0, 1.0])
    xl = np.array([0.0, 0.0])
    xu = np.array([5.0, 5.0])
    integ = np.array([0, 0])
    model = OptimizationModel(A, b, b, c, xl, xu, integ)
    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        solve(model, backend="unsupported_quantum_gpu")
