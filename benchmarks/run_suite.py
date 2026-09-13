import csv
import json
from pathlib import Path

from benchmarks.benchmark import run_benchmark


BASE_DIR = Path(__file__).resolve().parent
INSTANCES_FILE = BASE_DIR / "instances.csv"
RESULTS_DIR = BASE_DIR / "results"

CSV_OUTPUT = RESULTS_DIR / "benchmark_results.csv"
JSON_OUTPUT = RESULTS_DIR / "benchmark_results.json"

BACKENDS = [
    "cpu",
    "cuda",
    "auto",
]

MAX_ITERATIONS = 5000
RHO = 1.0


def load_instances():
    instances = []

    with INSTANCES_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            instances.append(row)

    return instances


def main():

    print("=" * 60)
    print("ApexCUDA Benchmark Suite")
    print("=" * 60)

    instances = load_instances()

    results = []

    total = len(instances) * len(BACKENDS)
    counter = 0

    for instance in instances:

        name = instance["Instance"]

        # instances.csv stores the filename indirectly
        filepath = BASE_DIR / "files" / f"{name}.mps"

        print()
        print(
            f"--- {name} "
            f"({instance['Type']}) ---"
        )

        for backend in BACKENDS:

            counter += 1

            print(
                f"[{counter}/{total}] "
                f"{name} / {backend} ... ",
                end="",
                flush=True,
            )

            result = run_benchmark(
                str(filepath),
                backend=backend,
                rho=RHO,
                max_iterations=MAX_ITERATIONS,
            )

            results.append(result)

            print(
                f"{result['status']} "
                f"({result['solve_time']:.6f}s)"
            )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # CSV
    # ------------------------------------------------------

    fieldnames = [
        "instance",
        "file",
        "problem_type",
        "objective_sense",
        "variables",
        "constraints",
        "nonzeros",
        "integer_variables",
        "continuous_variables",
        "requested_backend",
        "selected_backend",
        "backend",
        "status",
        "objective",
        "iterations",
        "constraint_violation",
        "bound_violation",
        "parse_time",
        "solve_time",
        "wall_time",
        "message",
    ]

    with CSV_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(results)

    # ------------------------------------------------------
    # JSON
    # ------------------------------------------------------

    with JSON_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("Benchmark complete")
    print("=" * 60)

    print(
        f"CSV : {CSV_OUTPUT.resolve()}"
    )

    print(
        f"JSON: {JSON_OUTPUT.resolve()}"
    )

    print(
        f"Rows: {len(results)}"
    )

    print(
        f"Maximum iterations: "
        f"{MAX_ITERATIONS}"
    )


if __name__ == "__main__":
    main()