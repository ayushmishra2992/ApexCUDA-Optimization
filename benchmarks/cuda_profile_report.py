import csv
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from backend.backends.cuda_backend import CUDABackend


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_FILE = OUTPUT_DIR / "cuda_profile_report.csv"

CASES = [
    (10_000, 10_000, 100_000),
    (25_000, 25_000, 500_000),
    (100_000, 100_000, 1_000_000),
    (100_000, 100_000, 5_000_000),
    (200_000, 200_000, 10_000_000),
]

REPEATS = 100
SEED = 999


def make_sparse_matrix(rows, cols, nnz, seed):

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

    data = rng.standard_normal(
        nnz
    ).astype(np.float64)

    A = csr_matrix(
        (data, (row, col)),
        shape=(rows, cols),
        dtype=np.float64,
    )

    A.sum_duplicates()
    A.sort_indices()

    return A


def run_case(case_no, rows, cols, target_nnz):

    print()
    print("=" * 70)
    print(
        f"Case {case_no}: "
        f"{rows:,} x {cols:,} | "
        f"target NNZ={target_nnz:,}"
    )
    print("=" * 70)

    A = make_sparse_matrix(
        rows,
        cols,
        target_nnz,
        SEED + case_no,
    )

    rng = np.random.default_rng(
        SEED + case_no
    )

    x = rng.standard_normal(
        cols
    ).astype(np.float64)

    y = rng.standard_normal(
        rows
    ).astype(np.float64)

    # --------------------------------------------------------
    # Create profiling-enabled CUDA backend
    # --------------------------------------------------------

    print("Creating profiling CUDA backend...")

    backend = CUDABackend(
        A,
        profiling=True,
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(10):

        backend.matvec(x)
        backend.rmatvec(y)

    backend.synchronize()
    backend.reset_profile()

    # --------------------------------------------------------
    # Profile repeated forward SpMV
    # --------------------------------------------------------

    print("Profiling forward SpMV...")

    for _ in range(REPEATS):

        result_forward = backend.matvec(x)

    # --------------------------------------------------------
    # Profile repeated transpose SpMV
    # --------------------------------------------------------

    print("Profiling transpose SpMV...")

    for _ in range(REPEATS):

        result_transpose = backend.rmatvec(y)

    # Force completion
    backend.synchronize()

    # --------------------------------------------------------
    # Transfer one result back
    # --------------------------------------------------------

    result_cpu = backend.to_cpu(
        result_forward
    )

    # --------------------------------------------------------
    # Profile summary
    # --------------------------------------------------------

    profile = backend.profile_summary()

    total_spmv = (
        profile["total_spmv_time_seconds"]
    )

    avg_matvec = (
        profile["matvec_time_seconds"]
        / max(profile["matvec_calls"], 1)
    )

    avg_rmatvec = (
        profile["rmatvec_time_seconds"]
        / max(profile["rmatvec_calls"], 1)
    )

    total_calls = (
        profile["matvec_calls"]
        + profile["rmatvec_calls"]
    )

    avg_spmv = (
        total_spmv / max(total_calls, 1)
    )

    setup_ms = (
        profile["setup_time_seconds"]
        * 1000
    )

    matvec_ms = (
        profile["matvec_time_seconds"]
        * 1000
    )

    rmatvec_ms = (
        profile["rmatvec_time_seconds"]
        * 1000
    )

    total_spmv_ms = (
        total_spmv * 1000
    )

    avg_spmv_ms = (
        avg_spmv * 1000
    )

    sync_ms = (
        profile["sync_time_seconds"]
        * 1000
    )

    peak_memory_mb = (
        profile["peak_memory_bytes"]
        / (1024 ** 2)
    )

    setup_memory_mb = (
        profile["setup_memory_bytes"]
        / (1024 ** 2)
    )

    # --------------------------------------------------------
    # CPU reference for correctness
    # --------------------------------------------------------

    cpu_forward = A @ x

    max_error = float(
        np.max(
            np.abs(
                cpu_forward - result_cpu
            )
        )
    )

    correctness = (
        max_error <= 1e-10
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Actual NNZ          : "
        f"{A.nnz:,}"
    )

    print(
        f"Matvec calls        : "
        f"{profile['matvec_calls']}"
    )

    print(
        f"Rmatvec calls       : "
        f"{profile['rmatvec_calls']}"
    )

    print(
        f"Average matvec      : "
        f"{avg_matvec * 1000:.4f} ms"
    )

    print(
        f"Average rmatvec     : "
        f"{avg_rmatvec * 1000:.4f} ms"
    )

    print(
        f"Total SpMV time     : "
        f"{total_spmv_ms:.4f} ms"
    )

    print(
        f"Average SpMV        : "
        f"{avg_spmv_ms:.4f} ms"
    )

    print(
        f"Setup time          : "
        f"{setup_ms:.4f} ms"
    )

    print(
        f"Sync time           : "
        f"{sync_ms:.4f} ms"
    )

    print(
        f"Setup memory        : "
        f"{setup_memory_mb:.2f} MB"
    )

    print(
        f"Peak GPU memory     : "
        f"{peak_memory_mb:.2f} MB"
    )

    print(
        f"Max error           : "
        f"{max_error:.3e}"
    )

    print(
        f"Correctness         : "
        f"{'PASS' if correctness else 'FAIL'}"
    )

    return {
        "case": case_no,
        "rows": rows,
        "columns": cols,
        "target_nnz": target_nnz,
        "actual_nnz": A.nnz,

        "matvec_calls": profile[
            "matvec_calls"
        ],

        "rmatvec_calls": profile[
            "rmatvec_calls"
        ],

        "matvec_time_seconds": profile[
            "matvec_time_seconds"
        ],

        "rmatvec_time_seconds": profile[
            "rmatvec_time_seconds"
        ],

        "total_spmv_time_seconds": total_spmv,

        "average_matvec_seconds": avg_matvec,

        "average_rmatvec_seconds": avg_rmatvec,

        "average_spmv_seconds": avg_spmv,

        "setup_time_seconds": profile[
            "setup_time_seconds"
        ],

        "sync_time_seconds": profile[
            "sync_time_seconds"
        ],

        "setup_memory_bytes": profile[
            "setup_memory_bytes"
        ],

        "peak_memory_bytes": profile[
            "peak_memory_bytes"
        ],

        "setup_memory_mb": setup_memory_mb,

        "peak_memory_mb": peak_memory_mb,

        "max_absolute_error": max_error,

        "correctness": (
            "PASS"
            if correctness
            else "FAIL"
        ),

        "gpu_name": profile[
            "gpu_name"
        ],
    }


def main():

    print("=" * 70)
    print("ApexCUDA CUDA Profiling Report")
    print("=" * 70)

    results = []

    for case_no, (rows, cols, nnz) in enumerate(
        CASES,
        start=1,
    ):

        try:

            result = run_case(
                case_no,
                rows,
                cols,
                nnz,
            )

            results.append(result)

        except Exception as exc:

            print(
                f"CASE {case_no} FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    if results:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = list(
            results[0].keys()
        )

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

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CUDA Profiling Complete")
    print("=" * 70)

    print(
        f"Successful cases: "
        f"{len(results)}/{len(CASES)}"
    )

    print(
        f"CSV: {OUTPUT_FILE.resolve()}"
    )

    print()

    print(
        f"{'NNZ':>12} "
        f"{'Matvec':>12} "
        f"{'Rmatvec':>12} "
        f"{'Setup':>12} "
        f"{'Peak MB':>12}"
    )

    print("-" * 65)

    for r in results:

        print(
            f"{r['actual_nnz']:>12,} "
            f"{r['average_matvec_seconds'] * 1000:>12.4f} "
            f"{r['average_rmatvec_seconds'] * 1000:>12.4f} "
            f"{r['setup_time_seconds'] * 1000:>12.3f} "
            f"{r['peak_memory_mb']:>12.2f}"
        )


if __name__ == "__main__":
    main()