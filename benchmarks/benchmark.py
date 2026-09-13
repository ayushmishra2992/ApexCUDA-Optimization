import time

from backend.formats.mps_parser import MPSParser
from backend.core.indioptima import solve


def run_benchmark(filepath, backend="cpu", rho=1.0, max_iterations=100):
    """
    Run one benchmark instance.

    Parameters
    ----------
    filepath : str
        Path to the MPS instance.
    backend : str
        Requested backend: "cpu", "cuda", or "auto".
    rho : float
        Solver rho parameter.
    max_iterations : int
        Maximum solver iterations.

    Returns
    -------
    dict
        Benchmark result with both requested and selected backend.
    """

    parser = MPSParser()

    # ---------------------------------------------------------
    # Parse
    # ---------------------------------------------------------
    parse_start = time.perf_counter()

    try:
        model = parser.parse(filepath)
    except Exception as exc:
        return {
            "file": filepath,
            "problem_type": None,
            "objective_sense": None,
            "variables": None,
            "constraints": None,
            "nonzeros": None,
            "integer_variables": None,
            "continuous_variables": None,
            "requested_backend": backend,
            "selected_backend": None,
            "backend": None,
            "status": "PARSE_ERROR",
            "objective": None,
            "iterations": None,
            "constraint_violation": None,
            "bound_violation": None,
            "parse_time": time.perf_counter() - parse_start,
            "solve_time": 0.0,
            "wall_time": time.perf_counter() - parse_start,
            "message": str(exc),
        }

    parse_time = time.perf_counter() - parse_start

    # ---------------------------------------------------------
    # Solve
    # ---------------------------------------------------------
    solve_start = time.perf_counter()

    try:
        result = solve(
            model,
            backend=backend,
            rho=rho,
            max_iterations=max_iterations,
        )

    except NotImplementedError as exc:
        solve_time = time.perf_counter() - solve_start

        return {
            "file": filepath,
            "problem_type": model.problem_type,
            "objective_sense": model.objective_sense,
            "variables": model.n_variables,
            "constraints": model.n_constraints,
            "nonzeros": model.nnz,
            "integer_variables": model.n_integer_variables,
            "continuous_variables": model.n_continuous_variables,
            "requested_backend": backend,
            "selected_backend": None,
            "backend": None,
            "status": "NOT_SUPPORTED",
            "objective": None,
            "iterations": None,
            "constraint_violation": None,
            "bound_violation": None,
            "parse_time": parse_time,
            "solve_time": solve_time,
            "wall_time": parse_time + solve_time,
            "message": str(exc),
        }

    except Exception as exc:
        solve_time = time.perf_counter() - solve_start

        return {
            "file": filepath,
            "problem_type": model.problem_type,
            "objective_sense": model.objective_sense,
            "variables": model.n_variables,
            "constraints": model.n_constraints,
            "nonzeros": model.nnz,
            "integer_variables": model.n_integer_variables,
            "continuous_variables": model.n_continuous_variables,
            "requested_backend": backend,
            "selected_backend": getattr(result, "backend", None)
            if "result" in locals()
            else None,
            "backend": getattr(result, "backend", None)
            if "result" in locals()
            else None,
            "status": "SOLVE_ERROR",
            "objective": None,
            "iterations": None,
            "constraint_violation": None,
            "bound_violation": None,
            "parse_time": parse_time,
            "solve_time": solve_time,
            "wall_time": parse_time + solve_time,
            "message": str(exc),
        }

    solve_time = time.perf_counter() - solve_start

    # ---------------------------------------------------------
    # Backend reporting
    # ---------------------------------------------------------
    selected_backend = getattr(result, "backend", None)

    # IMPORTANT:
    # requested_backend tells us what benchmark asked for.
    # selected_backend tells us what solver actually used.
    #
    # Example:
    # requested_backend = "auto"
    # selected_backend  = "cpu"
    #
    # This prevents AUTO results from being mislabeled as CPU.
    # ---------------------------------------------------------

    return {
        "file": filepath,
        "problem_type": model.problem_type,
        "objective_sense": model.objective_sense,
        "variables": model.n_variables,
        "constraints": model.n_constraints,
        "nonzeros": model.nnz,
        "integer_variables": model.n_integer_variables,
        "continuous_variables": model.n_continuous_variables,

        # New explicit backend fields
        "requested_backend": backend,
        "selected_backend": selected_backend,

        # Kept for backward compatibility with existing scripts
        "backend": selected_backend,

        "status": result.status,
        "objective": result.objective,
        "iterations": result.iterations,
        "constraint_violation": result.constraint_violation,
        "bound_violation": result.bound_violation,

        "parse_time": parse_time,
        "solve_time": solve_time,
        "wall_time": parse_time + solve_time,

        "message": getattr(result, "message", ""),
    }


def run_benchmarks(
    filepaths,
    backend="cpu",
    rho=1.0,
    max_iterations=100,
):
    """
    Run benchmarks for multiple instances.
    """

    return [
        run_benchmark(
            filepath,
            backend=backend,
            rho=rho,
            max_iterations=max_iterations,
        )
        for filepath in filepaths
    ]