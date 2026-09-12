import time

from backend.formats.mps_parser import MPSParser
from backend.core.indioptima import solve


def run_benchmark(
    filepath,
    backend="cpu",
    rho=1.0,
    max_iterations=100
):
    """
    Parse and solve an MPS optimization model.

    Returns a dictionary containing model statistics,
    solver results, and timing information.
    """

    parser = MPSParser()

    parse_start = time.time()
    model = parser.parse(filepath)
    parse_time = time.time() - parse_start

    solve_start = time.time()
    try:
        result = solve(
            model,
            backend=backend,
            rho=rho,
            max_iterations=max_iterations
        )

    except NotImplementedError as exc:
        return {
            "file": filepath,
            "problem_type": model.problem_type,
            "objective_sense": model.objective_sense,
            "variables": model.n_variables,
            "constraints": model.n_constraints,
            "nonzeros": model.nnz,
            "integer_variables": model.n_integer_variables,
            "continuous_variables": model.n_continuous_variables,
            "parse_time": parse_time,
            "solve_time": None,
            "status": "NOT_SUPPORTED",
            "message": str(exc),
            "objective": None,
            "iterations": None,
            "constraint_violation": None,
            "bound_violation": None,
            "backend": backend,
        }
    solve_time = time.time() - solve_start

    return {
        "file": filepath,
        "problem_type": model.problem_type,
        "objective_sense": model.objective_sense,
        "variables": model.n_variables,
        "constraints": model.n_constraints,
        "nonzeros": model.nnz,
        "integer_variables": model.n_integer_variables,
        "continuous_variables": model.n_continuous_variables,
        "parse_time": parse_time,
        "solve_time": solve_time,
        "status": result.status,
        "objective": result.objective,
        "iterations": result.iterations,
        "constraint_violation": result.constraint_violation,
        "bound_violation": result.bound_violation,
        "backend": result.backend,
    }

def run_benchmarks(
    filepaths,
    backend="cpu",
    rho=1.0,
    max_iterations=100
):
    """
    Run the benchmark runner on multiple optimization models.

    Returns a list of benchmark result dictionaries.
    """

    results = []

    for filepath in filepaths:
        result = run_benchmark(
            filepath,
            backend=backend,
            rho=rho,
            max_iterations=max_iterations
        )
        results.append(result)

    return results