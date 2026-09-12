import numpy as np
from backend.core.optimization_model import OptimizationResult
from backend.core.presolver import GenericPresolver
from backend.core.solver import ADMMSolver
from backend.core.milp_solver import MILPSolver
from backend.backends.backend_selector import BackendSelector
from backend.core.solver import (
    ADMMSolver,
    verify_solution
)

def solve(model, backend="auto", rho=1.0, max_iterations=100):
    
    original_model = model
    presolver = GenericPresolver()
    presolve_result = presolver.presolve(model)

    model = presolve_result.model
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            "Expected 'auto', 'cpu', or 'cuda'."
        )
    
    if backend == "auto":
        selector = BackendSelector()
        backend = selector.select(model)

    if presolve_result.status == "UNBOUNDED":
        return OptimizationResult(
            status="UNBOUNDED",
            objective=(
                np.inf
                if model.objective_sense == "min"
                else -np.inf
            ),
            solution=np.full(
                presolve_result.original_variables,
                np.nan
            ),
            solve_time=0.0,
            iterations=0,
            constraint_violation=np.inf,
            bound_violation=np.inf,
            backend=backend,
            problem_type=model.problem_type
        )

    if presolve_result.status == "SOLVED":
        original_solution = presolve_result.postsolve(
            np.array([], dtype=float)
        )

        objective = float(
            original_model.objective @ original_solution
        )

        verification = verify_solution(
            original_model,
            original_solution
        )

        return OptimizationResult(
            status=(
                "OPTIMAL"
                if verification["feasible"]
                else "VERIFICATION_FAILED"
            ),
            objective=objective,
            solution=original_solution,
            solve_time=0.0,
            iterations=0,
            constraint_violation=verification[
                "constraint_violation"
            ],
            bound_violation=verification[
                "bound_violation"
            ],
            backend=backend,
            problem_type=original_model.problem_type
        )

    if presolve_result.status == "INFEASIBLE":
        return OptimizationResult(
            status="INFEASIBLE",
            objective=np.inf if model.objective_sense == "min" else -np.inf,
            solution=np.zeros(
                presolve_result.original_variables,
                dtype=float
            ),
            solve_time=0.0,
            iterations=0,
            constraint_violation=np.inf,
            bound_violation=np.inf,
            backend=backend,
            problem_type=model.problem_type
        )

    if model.problem_type == "LP":

        solver = ADMMSolver(
            rho=rho,
            max_iterations=max_iterations,
            backend=backend
        )

    elif model.problem_type == "MILP":

        solver = MILPSolver(
            rho=rho,
            max_lp_iterations=max_iterations,
            backend=backend
        )

    else:
        raise NotImplementedError(
            f"Problem type '{model.problem_type}' is not currently supported."
        )

    result = solver.solve(model)

    if result.solution is not None:
        original_solution = presolve_result.postsolve(
            result.solution
        )

        result.solution = original_solution

        if presolve_result.objective_offset != 0.0:
            result.objective += presolve_result.objective_offset

        verification = verify_solution(
            original_model,
            result.solution
        )

        result.constraint_violation = verification[
            "constraint_violation"
        ]

        result.bound_violation = verification[
            "bound_violation"
        ]

        if not verification["feasible"]:
            result.status = "VERIFICATION_FAILED"

    return result


def main():
    import argparse
    from backend.formats.mps_parser import MPSParser

    parser = argparse.ArgumentParser(
        description="IndiOptima Optimization Solver"
    )

    parser.add_argument(
        "input_file",
        help="Path to an MPS optimization model"
    )

    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Numerical backend"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Maximum solver iterations"
    )

    args = parser.parse_args()

    model_parser = MPSParser()
    model = model_parser.parse(args.input_file)

    print("IndiOptima")
    print("=" * 40)

    summary = model.summary()

    print(f"Problem type: {summary['problem_type']}")
    print(f"Objective sense: {summary['objective_sense']}")
    print(f"Variables: {summary['variables']}")
    print(f"Constraints: {summary['constraints']}")
    print(f"Nonzeros: {summary['nonzeros']}")
    print(f"Integer variables: {summary['integer_variables']}")
    print(f"Continuous variables: {summary['continuous_variables']}")
    print()

    try:
        result = solve(
            model,
            backend=args.backend,
            max_iterations=args.iterations
        )

        print(f"Status: {result.status}")
        print(f"Objective: {result.objective}")
        print(f"Iterations: {result.iterations}")
        print(f"Constraint violation: {result.constraint_violation}")
        print(f"Bound violation: {result.bound_violation}")
        print(f"Backend: {result.backend}")

    except NotImplementedError as exc:
        print(f"Status: NOT_SUPPORTED")
        print(f"Message: {exc}")


if __name__ == "__main__":
    main()