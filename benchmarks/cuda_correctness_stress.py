import csv
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from backend.backends.cuda_backend import CUDABackend


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_FILE = OUTPUT_DIR / "cuda_correctness_stress.csv"

CASES = [
    # rows, cols, nnz
    (10, 10, 10),
    (100, 50, 100),
    (1000, 500, 1000),
    (5000, 1000, 5000),
    (10000, 10000, 10000),
    (10000, 5000, 50000),
    (20000, 10000, 100000),
]

SEED = 2026


def make_matrix(rows, cols, nnz, seed):

    rng = np.random.default_rng(seed)

    row = rng.integers(
        0,
        rows,
        size=nnz,
        dtype=np.int32,
    )

    col = rng.integers(
        0,
        cols,
        size=nnz,
        dtype=np.int32,
    )

    data = rng.standard_normal(nnz).astype(np.float64)

    A = csr_matrix(
        (data, (row, col)),
        shape=(rows, cols),
        dtype=np.float64,
    )

    A.sum_duplicates()
    A.sort_indices()

    return A


def check_case(case_no, rows, cols, nnz):

    print()
    print("=" * 65)
    print(
        f"Case {case_no}: "
        f"{rows} x {cols}, target NNZ={nnz}"
    )
    print("=" * 65)

    rng = np.random.default_rng(
        SEED + case_no
    )

    A = make_matrix(
        rows,
        cols,
        nnz,
        SEED + case_no,
    )

    x = rng.standard_normal(cols).astype(
        np.float64
    )

    y = rng.standard_normal(rows).astype(
        np.float64
    )

    # --------------------------------------------------------
    # CPU forward
    # --------------------------------------------------------

    cpu_forward = A @ x

    # --------------------------------------------------------
    # GPU backend
    # --------------------------------------------------------

    backend = CUDABackend(A)

    gpu_forward = backend.to_cpu(
        backend.matvec(x)
    )

    # --------------------------------------------------------
    # Forward error
    # --------------------------------------------------------

    forward_abs_error = float(
        np.max(
            np.abs(
                cpu_forward - gpu_forward
            )
        )
    )

    forward_rel_error = float(
        np.linalg.norm(
            cpu_forward - gpu_forward
        )
        /
        max(
            np.linalg.norm(cpu_forward),
            1e-30,
        )
    )

    forward_pass = (
        forward_abs_error <= 1e-10
        or forward_rel_error <= 1e-10
    )

    # --------------------------------------------------------
    # CPU transpose
    # --------------------------------------------------------

    cpu_transpose = A.T @ y

    # --------------------------------------------------------
    # GPU transpose
    # --------------------------------------------------------

    gpu_transpose = backend.to_cpu(
        backend.rmatvec(y)
    )

    # --------------------------------------------------------
    # Transpose error
    # --------------------------------------------------------

    transpose_abs_error = float(
        np.max(
            np.abs(
                cpu_transpose - gpu_transpose
            )
        )
    )

    transpose_rel_error = float(
        np.linalg.norm(
            cpu_transpose - gpu_transpose
        )
        /
        max(
            np.linalg.norm(cpu_transpose),
            1e-30,
        )
    )

    transpose_pass = (
        transpose_abs_error <= 1e-10
        or transpose_rel_error <= 1e-10
    )

    overall_pass = (
        forward_pass
        and transpose_pass
    )

    print(
        f"Actual NNZ          : {A.nnz:,}"
    )

    print(
        f"Forward max error   : "
        f"{forward_abs_error:.3e}"
    )

    print(
        f"Forward rel error   : "
        f"{forward_rel_error:.3e}"
    )

    print(
        f"Transpose max error : "
        f"{transpose_abs_error:.3e}"
    )

    print(
        f"Transpose rel error : "
        f"{transpose_rel_error:.3e}"
    )

    print(
        f"Forward             : "
        f"{'PASS' if forward_pass else 'FAIL'}"
    )

    print(
        f"Transpose           : "
        f"{'PASS' if transpose_pass else 'FAIL'}"
    )

    print(
        f"Overall             : "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    return {
        "case": case_no,
        "rows": rows,
        "columns": cols,
        "target_nnz": nnz,
        "actual_nnz": A.nnz,

        "forward_max_error": forward_abs_error,
        "forward_relative_error": forward_rel_error,
        "forward_status": (
            "PASS"
            if forward_pass
            else "FAIL"
        ),

        "transpose_max_error": transpose_abs_error,
        "transpose_relative_error": transpose_rel_error,
        "transpose_status": (
            "PASS"
            if transpose_pass
            else "FAIL"
        ),

        "overall_status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
    }


def main():

    print("=" * 65)
    print("ApexCUDA CUDA Correctness Stress Test")
    print("=" * 65)

    results = []

    for case_no, (rows, cols, nnz) in enumerate(
        CASES,
        start=1,
    ):

        try:

            result = check_case(
                case_no,
                rows,
                cols,
                nnz,
            )

            results.append(result)

        except Exception as exc:

            print(
                f"Case {case_no} ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            results.append({
                "case": case_no,
                "rows": rows,
                "columns": cols,
                "target_nnz": nnz,
                "actual_nnz": 0,
                "overall_status": "ERROR",
                "error": str(exc),
            })

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = sorted({
        key
        for result in results
        for key in result.keys()
    })

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    passed = sum(
        1
        for r in results
        if r.get("overall_status") == "PASS"
    )

    print()
    print("=" * 65)
    print("CUDA Correctness Test Complete")
    print("=" * 65)

    print(
        f"Passed: {passed}/{len(results)}"
    )

    print(
        f"CSV: {OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()