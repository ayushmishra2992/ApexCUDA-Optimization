import json
import math
from pathlib import Path


def load_references():
    reference_path = Path(__file__).parent / "highs_references.json"

    with open(reference_path, "r", encoding="utf-8") as file:
        return json.load(file)


def compare_objectives(indi_objective, highs_objective, tolerance=1e-6):
    if indi_objective is None or highs_objective is None:
        return None

    if not (
        math.isfinite(indi_objective)
        and math.isfinite(highs_objective)
    ):
        return None

    difference = abs(indi_objective - highs_objective)

    if highs_objective == 0:
        return difference

    return difference / abs(highs_objective)


def check_feasibility(result, tolerance=1e-6):
    constraint_violation = result.get("constraint_violation")
    bound_violation = result.get("bound_violation")

    if constraint_violation is None or bound_violation is None:
        return False

    if not (
        math.isfinite(constraint_violation)
        and math.isfinite(bound_violation)
    ):
        return False

    return (
        constraint_violation <= tolerance
        and bound_violation <= tolerance
    )


def compare_result(result, reference):
    indi_status = result.get("status")
    indi_objective = result.get("objective")

    highs_status = reference.get("status")
    highs_objective = reference.get("objective_value")

    feasible = check_feasibility(result)

    objective_error = compare_objectives(
        indi_objective,
        highs_objective
    )

    objective_match = (
        objective_error is not None
        and objective_error <= 1e-6
    )

    highs_optimal = highs_status == "HighsModelStatus.kOptimal"
    indi_optimal = indi_status == "OPTIMAL"

    optimality_match = indi_optimal and highs_optimal

    # ---------------------------------------------------------
    # Final benchmark classification
    # ---------------------------------------------------------

    if indi_status == "OPTIMAL" and feasible and objective_match:
        overall_result = "OPTIMAL_MATCH"

    elif indi_status == "FEASIBLE" and feasible and objective_match:
        overall_result = "FEASIBLE_OBJECTIVE_MATCH"

    elif indi_status == "FEASIBLE" and feasible:
        overall_result = "FEASIBLE"

    elif indi_status == "MAX_ITERATIONS_REACHED":
        overall_result = "NOT_CONVERGED"

    elif indi_status == "NO_INTEGER_SOLUTION":
        overall_result = "MILP_SOLVER_LIMITATION"

    elif indi_status == "INFEASIBLE":
        overall_result = "INFEASIBLE"

    else:
        overall_result = "SOLVER_FAILURE"

    indi_runtime = result.get("solve_time")
    highs_runtime = reference.get("runtime_seconds")

    speedup = None

    if (
        indi_runtime is not None
        and highs_runtime is not None
        and indi_runtime > 0
        and math.isfinite(indi_runtime)
        and math.isfinite(highs_runtime)
    ):
        speedup = highs_runtime / indi_runtime

    return {
        "feasible": feasible,
        "objective_match": objective_match,
        "optimality_match": optimality_match,
        "objective_error": objective_error,
        "overall_result": overall_result,
        "indioptima_status": indi_status,
        "highs_status": highs_status,
        "indioptima_objective": indi_objective,
        "highs_objective": highs_objective,
        "indioptima_runtime": indi_runtime,
        "highs_runtime": highs_runtime,
        "speedup": speedup,
    }


def print_comparison(results):
    print("\n")
    print("=" * 120)
    print("INDIOPTIMA vs HiGHS COMPARISON")
    print("=" * 120)

    header = (
        f"{'Instance':<12}"
        f"{'Indi Status':<25}"
        f"{'Feasible':<10}"
        f"{'Obj Error':<15}"
        f"{'HiGHS Status':<30}"
        f"{'Result':<30}"
    )

    print(header)
    print("-" * 120)

    for instance, comparison in results.items():

        objective_error = comparison["objective_error"]

        if objective_error is None:
            objective_error_text = "N/A"
        else:
            objective_error_text = f"{objective_error:.6e}"

        print(
            f"{instance:<12}"
            f"{comparison['indioptima_status']:<25}"
            f"{'YES' if comparison['feasible'] else 'NO':<10}"
            f"{objective_error_text:<15}"
            f"{comparison['highs_status']:<30}"
            f"{comparison['overall_result']:<30}"
        )

    print("=" * 120)