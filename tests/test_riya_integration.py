from pathlib import Path

from backend.core.indioptima import solve
from backend.formats.mps_parser import MPSParser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def test_mps_parser_to_indioptima_lp():
    """Verify an MPS LP can flow through parsing into the solver."""
    mps_path = EXAMPLES_DIR / "tiny_test.mps"

    model = MPSParser().parse(str(mps_path))

    assert model.problem_type == "LP"
    assert model.n_variables > 0
    assert model.n_constraints > 0
    assert model.nnz > 0

    result = solve(
        model,
        backend="cpu",
        max_iterations=10,
    )

    assert result is not None
    assert result.problem_type == "LP"
    assert result.backend == "cpu"
    assert result.solution is not None


def test_mps_parser_to_indioptima_milp():
    """Verify an MPS MILP can flow through parsing into the solver."""
    mps_path = EXAMPLES_DIR / "tiny_milp.mps"

    model = MPSParser().parse(str(mps_path))

    assert model.problem_type == "MILP"
    assert model.n_integer_variables > 0
    assert model.n_variables > 0
    assert model.n_constraints > 0
    assert model.nnz > 0

    result = solve(
        model,
        backend="cpu",
        max_iterations=10,
    )

    assert result is not None
    assert result.problem_type == "MILP"
    assert result.backend == "cpu"
    assert result.solution is not None