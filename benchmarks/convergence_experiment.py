import csv
import time
from pathlib import Path

from backend.formats.mps_parser import MPSParser
from backend.core.indioptima import solve


def run_experiment(filepath, max_iterations):
    """
    Run IndiOptima on one benchmark instance
    using a specified iteration limit.
    """

    parser = MPSParser()

    parse_start = time.perf_counter()
    model = parser.parse(filepath)
    parse_time = time.perf_counter() - parse_start

    solve_start = time.perf_counter()

    try:
        result = solve(
            model,
            backend="cpu",
            rho=1.0,
            max_iterations=max_iterations
        )

        solve_time = time.perf_counter() - solve_start

        return {
            "instance": Path(filepath).stem,
            "problem_type": model.problem_type,
            "variables": model.n_variables,
            "constraints": model.n_constraints,
            "nonzeros": model.nnz,
            "max_iterations": max_iterations,
            "actual_iterations": result.iterations,
            "status": result.status,
            "objective": result.objective,
            "constraint_violation":
                result.constraint_violation,
            "bound_violation":
                result.bound_violation,
            "parse_time": parse_time,
            "solve_time": solve_time,
        }

    except Exception as exc:

        solve_time = time.perf_counter() - solve_start

        return {
            "instance": Path(filepath).stem,
            "problem_type": model.problem_type,
            "variables": model.n_variables,
            "constraints": model.n_constraints,
            "nonzeros": model.nnz,
            "max_iterations": max_iterations,
            "actual_iterations": None,
            "status": "ERROR",
            "objective": None,
            "constraint_violation": None,
            "bound_violation": None,
            "parse_time": parse_time,
            "solve_time": solve_time,
            "message": str(exc),
        }


def main():

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    manifest_path = (
        project_root
        / "benchmarks"
        / "instances.csv"
    )

    benchmark_dir = (
        project_root
        / "benchmarks"
        / "files"
    )

    results_dir = (
        project_root
        / "benchmarks"
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Iteration limits to test
    # ---------------------------------------------------------

    iteration_limits = [
        100,
        500,
        1000,
        5000
    ]

    # ---------------------------------------------------------
    # Load benchmark instances
    # ---------------------------------------------------------

    instances = []

    with open(
        manifest_path,
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            instance_name = row["Instance"]

            filepath = (
                benchmark_dir
                / f"{instance_name}.mps"
            )

            if filepath.exists():
                instances.append(filepath)

    print("=" * 80)
    print("IndiOptima Convergence Experiment")
    print("=" * 80)

    print(
        f"Instances: {len(instances)}"
    )

    print(
        f"Iteration limits: "
        f"{iteration_limits}"
    )

    results = []

    # ---------------------------------------------------------
    # Run experiments
    # ---------------------------------------------------------

    for filepath in instances:

        print(
            f"\nInstance: "
            f"{filepath.stem}"
        )

        for max_iterations in iteration_limits:

            print(
                f"  Testing "
                f"{max_iterations} iterations..."
            )

            result = run_experiment(
                filepath,
                max_iterations
            )

            results.append(result)

            print(
                f"    Status: "
                f"{result['status']}"
            )

            print(
                f"    Actual iterations: "
                f"{result['actual_iterations']}"
            )

            print(
                f"    Constraint violation: "
                f"{result['constraint_violation']}"
            )

            print(
                f"    Solve time: "
                f"{result['solve_time']:.6f} s"
            )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output_path = (
        results_dir
        / "convergence_results.csv"
    )

    fieldnames = [
        "instance",
        "problem_type",
        "variables",
        "constraints",
        "nonzeros",
        "max_iterations",
        "actual_iterations",
        "status",
        "objective",
        "constraint_violation",
        "bound_violation",
        "parse_time",
        "solve_time",
        "message"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print("\n")
    print("=" * 80)
    print("CONVERGENCE EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        f"Results saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()