from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

BENCHMARK_FILE = RESULTS / "benchmark_results.csv"
HIGHS_FILE = ROOT / "highs_references.json"

OUTPUT_CSV = RESULTS / "final_benchmark_summary.csv"
OUTPUT_JSON = RESULTS / "final_benchmark_summary.json"


# ============================================================
# LOAD DATA
# ============================================================

benchmark = pd.read_csv(BENCHMARK_FILE)

# Recover instance name from MPS path because the current
# benchmark runner leaves the instance column blank.
benchmark["instance"] = (
    benchmark["instance"]
    .fillna("")
    .astype(str)
    .str.strip()
)

missing = benchmark["instance"] == ""

benchmark.loc[missing, "instance"] = (
    benchmark.loc[missing, "file"]
    .astype(str)
    .apply(lambda x: Path(x).stem)
)


# ============================================================
# LOAD HIGHS REFERENCES
# ============================================================

import json

with open(HIGHS_FILE, "r", encoding="utf-8") as f:
    highs = json.load(f)


# ============================================================
# BUILD HIGHS LOOKUP
# ============================================================

highs_lookup = {}

if isinstance(highs, dict):

    for key, value in highs.items():

        if isinstance(value, dict):

            instance = (
                value.get("instance")
                or value.get("name")
                or key
            )

            highs_lookup[str(instance).lower()] = value

        else:

            highs_lookup[str(key).lower()] = {
                "objective": value
            }

elif isinstance(highs, list):

    for value in highs:

        if not isinstance(value, dict):
            continue

        instance = (
            value.get("instance")
            or value.get("name")
        )

        if instance is not None:
            highs_lookup[str(instance).lower()] = value


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "variables",
    "constraints",
    "nonzeros",
    "integer_variables",
    "continuous_variables",
    "objective",
    "iterations",
    "constraint_violation",
    "bound_violation",
    "parse_time",
    "solve_time",
    "wall_time",
]

for column in numeric_columns:

    if column in benchmark.columns:

        benchmark[column] = pd.to_numeric(
            benchmark[column],
            errors="coerce"
        )


# ============================================================
# OBJECTIVE ERROR
# ============================================================

def get_highs_objective(instance):

    ref = highs_lookup.get(str(instance).lower())

    if ref is None:
        return None

    if isinstance(ref, dict):

        for key in [
            "objective",
            "optimal_objective",
            "objective_value",
            "value",
        ]:

            if key in ref:
                try:
                    return float(ref[key])
                except (TypeError, ValueError):
                    return None

    return None


benchmark["highs_objective"] = benchmark["instance"].apply(
    get_highs_objective
)


def calculate_objective_error(row):

    obj = row["objective"]
    ref = row["highs_objective"]

    if pd.isna(obj) or pd.isna(ref):
        return None

    return abs(float(obj) - float(ref))


benchmark["objective_error"] = benchmark.apply(
    calculate_objective_error,
    axis=1
)


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================

columns = [
    "instance",
    "problem_type",
    "variables",
    "constraints",
    "nonzeros",
    "integer_variables",
    "requested_backend",
    "selected_backend",
    "status",
    "objective",
    "highs_objective",
    "objective_error",
    "iterations",
    "constraint_violation",
    "bound_violation",
    "solve_time",
    "wall_time",
]

columns = [
    c for c in columns
    if c in benchmark.columns
]

summary = benchmark[columns].copy()


# ============================================================
# SORT
# ============================================================

backend_order = {
    "cpu": 0,
    "cuda": 1,
    "auto": 2,
}

summary["_backend_order"] = (
    summary["requested_backend"]
    .map(backend_order)
    .fillna(99)
)

summary = summary.sort_values(
    ["instance", "_backend_order"]
)

summary = summary.drop(
    columns=["_backend_order"]
)


# ============================================================
# SAVE CSV
# ============================================================

summary.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SAVE JSON
# ============================================================

summary.to_json(
    OUTPUT_JSON,
    orient="records",
    indent=2
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n==========================================")
print("FINAL BENCHMARK SUMMARY")
print("==========================================")

print(f"\nCSV:")
print(OUTPUT_CSV)

print("\nJSON:")
print(OUTPUT_JSON)

print("\nRows:", len(summary))

print("\nStatus counts:")

print(
    summary["status"]
    .value_counts()
    .to_string()
)

print("\nFinal table:\n")

print(
    summary[
        [
            "instance",
            "problem_type",
            "nonzeros",
            "requested_backend",
            "selected_backend",
            "status",
            "objective",
            "highs_objective",
            "objective_error",
            "iterations",
            "solve_time",
        ]
    ].to_string(index=False)
)

print("\n==========================================")
print("SUMMARY CREATION COMPLETE")
print("==========================================")