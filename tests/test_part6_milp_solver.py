import os
import numpy as np
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

    assert result.status == "OPTIMAL"

    assert np.allclose(
        result.solution,
        np.round(result.solution),
        atol=1e-6
    )

    assert np.isclose(
        result.objective,
        1.0,
        atol=1e-6
    )

    assert result.constraint_violation <= 1e-6
    assert result.bound_violation <= 1e-6

    assert result.iterations > 1