import json
import time
from pathlib import Path

import highspy


def solve_with_highs(filepath):
    """
    Solve one MPS instance using HiGHS
    and return reference benchmark information.
    """

    highs = highspy.Highs()

    # Disable HiGHS console output
    highs.setOptionValue(
        "log_to_console",
        False
    )

    # Read MPS model
    highs.readModel(
        str(filepath)
    )

    # Measure HiGHS solve time
    start = time.perf_counter()

    highs.run()

    solve_time = (
        time.perf_counter() - start
    )

    model_status = (
        highs.getModelStatus()
    )

    info = highs.getInfo()
    model = highs.getLp()

    status = str(model_status)

    if "kOptimal" in status:
        status = "HighsModelStatus.kOptimal"

    elif "kInfeasible" in status:
        status = "HighsModelStatus.kInfeasible"

    elif "kUnbounded" in status:
        status = "HighsModelStatus.kUnbounded"

    objective_value = (
        info.objective_function_value
    )

    return {
        "status": status,
        "objective_value": objective_value,
        "runtime_seconds": solve_time,
        "variables": model.num_col_,
        "constraints": model.num_row_,
        "type": "LP"
    }


def main():

    project_root = (
        Path(__file__).resolve().parent.parent
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

    instances = [
        "bandm",
        "sc105",
        "25fv47"
    ]

    # Load existing references
    with open(
        reference_path,
        "r",
        encoding="utf-8"
    ) as file:

        references = json.load(file)

    print("=" * 70)
    print("Generating HiGHS References")
    print("=" * 70)

    for instance in instances:

        filepath = (
            benchmark_dir
            / f"{instance}.mps"
        )

        if not filepath.exists():

            print(
                f"ERROR: Missing file: "
                f"{filepath}"
            )

            continue

        print(
            f"\nRunning HiGHS: "
            f"{instance}.mps"
        )

        result = solve_with_highs(
            filepath
        )

        references[instance] = result

        print(
            f"Status:      "
            f"{result['status']}"
        )

        print(
            f"Objective:   "
            f"{result['objective_value']}"
        )

        print(
            f"Runtime:     "
            f"{result['runtime_seconds']:.6f} s"
        )

        print(
            f"Variables:   "
            f"{result['variables']}"
        )

        print(
            f"Constraints: "
            f"{result['constraints']}"
        )

    # Save updated references
    with open(
        reference_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            references,
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print(
        "HiGHS references updated successfully."
    )
    print("=" * 70)

    print(
        f"\nSaved to:\n"
        f"{reference_path}"
    )


if __name__ == "__main__":
    main()