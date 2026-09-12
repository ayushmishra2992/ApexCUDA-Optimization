import os
import pytest
from backend.mps_parser import MPSParser

EXAMPLES_DIR = "examples"

def test_tiny_test_mps():
    path = os.path.join(EXAMPLES_DIR, "tiny_test.mps")
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
    model = MPSParser().parse(path)
    assert model.n_variables == 2
    assert model.n_constraints >= 1

def test_tiny_milp_mps():
    path = os.path.join(EXAMPLES_DIR, "tiny_milp.mps")
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
    model = MPSParser().parse(path)
    assert model.n_variables == 2
    assert model.problem_type == "MILP"

def test_tiny_max_mps_objective_sense():
    """Bug detection: MPS parser should properly identify MAX sense on tiny_max.mps."""
    path = os.path.join(EXAMPLES_DIR, "tiny_max.mps")
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
    model = MPSParser().parse(path)
    assert model.objective_sense.lower() == "max", (
        f"BUG in MPSParser: Expected objective_sense='max' for tiny_max.mps, got '{model.objective_sense}'"
    )

def test_fractional_milp_mps():
    path = os.path.join(EXAMPLES_DIR, "fractional_milp.mps")
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
    model = MPSParser().parse(path)
    assert model.problem_type == "MILP"
