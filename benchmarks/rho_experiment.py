import csv
import time
from pathlib import Path

from backend.formats.mps_parser import MPSParser
from backend.core.indioptima import solve


def run_experiment(filepath, rho):
    parser = MPSParser()

    parse_start = time.perf_counter()
    model = parser.parse(filepath)
    parse_time = time.perf_counter() - parse_start

    solve_start = time.perf_counter()

    try:
        result = solve(
            model,
            backend="cpu",
            rho=rho,
            max_iterations=5000
        )

        solve_time = time.perf_counter() - solve_start

        return {
            "instance": Path(filepath).stem,
            "problem_type": model.problem_type,
            "variables": model.n_variables,
            "constraints": model.n_constraints,
            "nonzeros": model.nnz,
            "rho": rho,
            "max_iterations": 5000,
            "actual_iterations": result.iterations,
            "status": result.status,
            "objective": result.objective,
            "constraint_violation": result.constraint_violation,
            "bound_violation": result.bound_violation,
            "parse_time": parse_time,
            "solve_time": solve_time,
            "message": "",
        }

    except Exception as exc:
        solve_time = time.perf_counter() - solve_start

        return {
            "instance": Path(filepath).stem,
            "problem_type": model.problem_type,
            "variables": model.n_variables,
            "constraints": model.n_constraints,
            "nonzeros": model.nnz,
            "rho": rho,
            "max_iterations": 5000,
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
    project_root = Path(__file__).resolve().parent.parent

    manifest_path = project_root / "benchmarks" / "instances.csv"
    benchmark_dir = project_root / "benchmarks" / "files"
    results_dir = project_root / "benchmarks" / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    rho_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    instances = []

    with open(manifest_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            instance_name = row["Instance"]
            filepath = benchmark_dir / f"{instance_name}.mps"

            if filepath.exists():
                instances.append(filepath)

    print("=" * 80)
    print("IndiOptima Rho Sensitivity Experiment")
    print("=" * 80)
    print(f"Instances: {len(instances)}")
    print(f"Rho values: {rho_values}")
    print("Maximum iterations per run: 5000")

    results = []

    for filepath in instances:
        print(f"\nInstance: {filepath.stem}")

        for rho in rho_values:
            print(f"  Testing rho={rho}...")

            result = run_experiment(filepath, rho)
            results.append(result)

            print(f"    Status: {result['status']}")
            print(f"    Actual iterations: {result['actual_iterations']}")
            print(
                f"    Constraint violation: "
                f"{result['constraint_violation']}"
            )
            print(
                f"    Solve time: "
                f"{result['solve_time']:.6f} s"
            )

    output_path = results_dir / "rho_results.csv"

    fieldnames = [
        "instance",
        "problem_type",
        "variables",
        "constraints",
        "nonzeros",
        "rho",
        "max_iterations",
        "actual_iterations",
        "status",
        "objective",
        "constraint_violation",
        "bound_violation",
        "parse_time",
        "solve_time",
        "message",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print("\n")
    print("=" * 80)
    print("RHO SENSITIVITY EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to:\n{output_path}")


if __name__ == "__main__":
    main()