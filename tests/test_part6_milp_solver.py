import os
import pytest
from backend.mps_parser import MPSParser
from backend.milp_solver import MILPSolver

def test_milp_solver_fractional_milp():
    path = "examples/fractional_milp.mps"
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
    model = MPSParser().parse(path)
    solver = MILPSolver(max_nodes=100)
    result = solver.solve(model)
    assert result is not None
    assert len(result.solution) == model.n_variables
