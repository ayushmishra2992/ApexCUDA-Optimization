import os
import pytest
from backend.mps_parser import MPSParser
from backend.solver import ADMMSolver

def test_admm_tiny_test_mps():
    path = "examples/tiny_test.mps"
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
    model = MPSParser().parse(path)
    solver = ADMMSolver(max_iterations=500)
    result = solver.solve(model)
    assert result is not None
    assert len(result.solution) == model.n_variables
    assert result.status in ["FEASIBLE", "OPTIMAL"]

def test_admm_tiny_max_mps():
    path = "examples/tiny_max.mps"
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
    model = MPSParser().parse(path)
    solver = ADMMSolver(max_iterations=500)
    result = solver.solve(model)
    assert result.status in ["FEASIBLE", "OPTIMAL"]
