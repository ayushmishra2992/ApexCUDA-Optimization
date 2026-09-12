import numpy as np
from scipy import sparse
from backend.optimization_model import OptimizationModel
from backend.solver import constraint_violation, bound_violation, integrality_violation, verify_solution

def test_constraint_violation_detects_infeasibility():
    Ax = np.array([0.0])
    lower = np.array([1.0])
    upper = np.array([np.inf])
    viol = constraint_violation(Ax, lower, upper)
    assert viol > 0.0, "Expected positive constraint violation"

def test_bound_violation_detection():
    x = np.array([5.0, -1.0])
    lower = np.array([0.0, 0.0])
    upper = np.array([10.0, 10.0])
    viol = bound_violation(x, lower, upper)
    assert viol > 0.0, "Expected bound violation for negative value"

def test_integrality_violation_detection():
    x = np.array([1.5, 2.0])
    integrality = np.array([1, 0])
    viol = integrality_violation(x, integrality)
    assert abs(viol - 0.5) < 1e-5, f"Expected 0.5 integrality violation, got {viol}"

def test_valid_solution_verification():
    A = sparse.eye(2, format="csr")
    b = np.array([2.0, 3.0])
    c = np.array([1.0, 1.0])
    xl = np.array([0.0, 0.0])
    xu = np.array([5.0, 5.0])
    integ = np.array([0, 0])
    model = OptimizationModel(A, b, b, c, xl, xu, integ)
    x_sol = np.array([2.0, 3.0])
    res = verify_solution(model, x_sol)
    assert res["feasible"]
