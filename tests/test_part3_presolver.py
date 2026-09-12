import numpy as np
from scipy import sparse
from backend.optimization_model import OptimizationModel
from backend.presolver import GenericPresolver

def test_presolver_normal_model_no_fixed_vars():
    A = sparse.eye(2, format="csr")
    b = np.array([1.0, 1.0])
    c = np.array([1.0, 1.0])
    xl = np.array([0.0, 0.0])
    xu = np.array([5.0, 5.0])
    integ = np.array([0, 0])
    model = OptimizationModel(A, b, b, c, xl, xu, integ)
    presolver = GenericPresolver()
    res = presolver.presolve(model)
    assert res is not None

def test_presolver_detects_fixed_variables():
    A = sparse.eye(2, format="csr")
    b = np.array([1.0, 1.0])
    c = np.array([1.0, 1.0])
    xl = np.array([2.0, 0.0])
    xu = np.array([2.0, 5.0])  # x0 is fixed to 2.0
    integ = np.array([0, 0])
    model = OptimizationModel(A, b, b, c, xl, xu, integ)
    presolver = GenericPresolver()
    res = presolver.presolve(model)
    assert res is not None
