import csv
import json
import time
from pathlib import Path

from backend.formats.mps_parser import MPSParser
from backend.core.indioptima import solve
from benchmarks.compare_results import compare_result


def run_benchmark(
    filepath,
    backend="cpu",
    rho=1.0,
    max_iterations=5000
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
    max_iterations=5000
):
    """
    Run the benchmark runner on multiple optimization models.

    Returns a list of benchmark result dictionaries.
    """

    results = []

    for filepath in filepaths:

        print(f"\nRunning benchmark: {filepath}")

        try:
            result = run_benchmark(
                filepath,
                backend=backend,
                rho=rho,
                max_iterations=max_iterations
            )

        except Exception as exc:
            result = {
                "file": filepath,
                "status": "ERROR",
                "message": str(exc),
            }

        results.append(result)

    return results


def save_results_csv(results, output_path):
    """
    Save benchmark results to a CSV file.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "instance",
        "problem_type",
        "objective_sense",
        "variables",
        "constraints",
        "nonzeros",
        "integer_variables",
        "continuous_variables",
        "status",
        "objective",
        "iterations",
        "constraint_violation",
        "bound_violation",
        "parse_time",
        "solve_time",
        "backend"
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

            writer.writerow({
                "instance": Path(
                    result["file"]
                ).stem,

                "problem_type":
                    result.get("problem_type"),

                "objective_sense":
                    result.get("objective_sense"),

                "variables":
                    result.get("variables"),

                "constraints":
                    result.get("constraints"),

                "nonzeros":
                    result.get("nonzeros"),

                "integer_variables":
                    result.get("integer_variables"),

                "continuous_variables":
                    result.get("continuous_variables"),

                "status":
                    result.get("status"),

                "objective":
                    result.get("objective"),

                "iterations":
                    result.get("iterations"),

                "constraint_violation":
                    result.get("constraint_violation"),

                "bound_violation":
                    result.get("bound_violation"),

                "parse_time":
                    result.get("parse_time"),

                "solve_time":
                    result.get("solve_time"),

                "backend":
                    result.get("backend")
            })

    print(
        f"\nBenchmark results saved to: "
        f"{output_path}"
    )


def main():
    """
    Load benchmark instances from instances.csv,
    run IndiOptima, and compare results with HiGHS.
    """

    project_root = Path(__file__).resolve().parent.parent

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

    reference_path = (
        project_root
        / "benchmarks"
        / "highs_references.json"
    )

    # ---------------------------------------------------------
    # Load benchmark instances
    # ---------------------------------------------------------

    filepaths = []

    with open(
        manifest_path,
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            instance_name = row["Instance"]

            mps_path = (
                benchmark_dir
                / f"{instance_name}.mps"
            )

            if not mps_path.exists():

                print(
                    f"WARNING: Missing benchmark file: "
                    f"{mps_path}"
                )

                continue

            filepaths.append(
                str(mps_path)
            )

    print("=" * 70)
    print("IndiOptima Benchmark")
    print("=" * 70)

    print(
        f"Instances found: "
        f"{len(filepaths)}"
    )

    print(
        "Maximum LP iterations: 5000"
    )

    # ---------------------------------------------------------
    # Run IndiOptima benchmarks
    # ---------------------------------------------------------

    results = run_benchmarks(
        filepaths,
        max_iterations=5000
    )

    # ---------------------------------------------------------
    # Save benchmark results
    # ---------------------------------------------------------

    results_path = (
        project_root
        / "benchmarks"
        / "results"
        / "benchmark_results.csv"
    )

    save_results_csv(
        results,
        results_path
    )

    # ---------------------------------------------------------
    # Print IndiOptima results
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    for result in results:

        print("\n" + "-" * 70)

        print(
            f"File:         "
            f"{Path(result['file']).name}"
        )

        print(
            f"Status:       "
            f"{result.get('status')}"
        )

        if "problem_type" in result:

            print(
                f"Problem type: "
                f"{result['problem_type']}"
            )

            print(
                f"Variables:    "
                f"{result['variables']}"
            )

            print(
                f"Constraints:  "
                f"{result['constraints']}"
            )

            print(
                f"Nonzeros:     "
                f"{result['nonzeros']}"
            )

            print(
                f"Integer vars: "
                f"{result['integer_variables']}"
            )

            print(
                f"Parse time:   "
                f"{result['parse_time']:.6f} s"
            )

        if result.get("solve_time") is not None:

            print(
                f"Solve time:   "
                f"{result['solve_time']:.6f} s"
            )

        print(
            f"Objective:     "
            f"{result.get('objective')}"
        )

        print(
            f"Iterations:   "
            f"{result.get('iterations')}"
        )

        print(
            f"Constraint violation: "
            f"{result.get('constraint_violation')}"
        )

        print(
            f"Bound violation:      "
            f"{result.get('bound_violation')}"
        )

        if result.get("message"):

            print(
                f"Message:      "
                f"{result['message']}"
            )

    # ---------------------------------------------------------
    # Load HiGHS reference results
    # ---------------------------------------------------------

    with open(
        reference_path,
        "r",
        encoding="utf-8"
    ) as file:

        references = json.load(file)

    # ---------------------------------------------------------
    # Compare IndiOptima with HiGHS
    # ---------------------------------------------------------

    comparison_results = {}

    for result in results:

        instance_name = Path(
            result["file"]
        ).stem

        if instance_name not in references:
            continue

        comparison = compare_result(
            result,
            references[instance_name]
        )

        comparison_results[instance_name] = {
            "result": result,
            "comparison": comparison
        }

    # ---------------------------------------------------------
    # Print detailed comparison table
    # ---------------------------------------------------------

    print("\n")
    print("=" * 120)
    print("INDIOPTIMA vs HiGHS COMPARISON")
    print("=" * 120)

    print(
        f"{'Instance':<12}"
        f"{'Indi Status':<24}"
        f"{'Feasible':<10}"
        f"{'Obj Error':<15}"
        f"{'HiGHS Status':<24}"
        f"{'Result':<28}"
    )

    print("-" * 120)

    for instance, data in comparison_results.items():

        comparison = data["comparison"]

        objective_error = comparison.get(
            "objective_error"
        )

        if objective_error is None:

            error_text = "N/A"

        else:

            error_text = (
                f"{objective_error:.6e}"
            )

        feasible_text = (
            "YES"
            if comparison.get("feasible")
            else "NO"
        )

        print(
            f"{instance:<12}"
            f"{comparison.get('indioptima_status', 'N/A'):<24}"
            f"{feasible_text:<10}"
            f"{error_text:<15}"
            f"{comparison.get('highs_status', 'N/A'):<24}"
            f"{comparison.get('overall_result', 'N/A'):<28}"
        )

    print("=" * 120)


if __name__ == "__main__":
    main()