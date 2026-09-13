import os
import pytest

from backend.formats.mps_parser import MPSParser


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
    """
    Verify that the MPS parser correctly identifies MAX objective sense.
    """

    path = os.path.join(EXAMPLES_DIR, "tiny_max.mps")

    if not os.path.exists(path):
        pytest.skip(f"{path} not found")

    model = MPSParser().parse(path)

    assert model.objective_sense.lower() == "max", (
        "BUG in MPSParser: Expected objective_sense='max' "
        f"for tiny_max.mps, got '{model.objective_sense}'"
    )


def test_fractional_milp_mps():
    path = os.path.join(EXAMPLES_DIR, "fractional_milp.mps")

    if not os.path.exists(path):
        pytest.skip(f"{path} not found")

    model = MPSParser().parse(path)

    assert model.problem_type == "MILP"


def test_mps_missing_objective_row_raises_error(tmp_path):
    """
    A valid optimization model must contain an objective row.
    The parser should reject an MPS file that has no N-type row.
    """

    path = tmp_path / "missing_objective.mps"

    path.write_text(
        """NAME TEST
ROWS
 L CON1
COLUMNS
 X1 CON1 1
RHS
 RHS1 CON1 10
BOUNDS
 LO BND1 X1 0
ENDATA
"""
    )

    with pytest.raises(ValueError, match="objective"):
        MPSParser().parse(str(path))