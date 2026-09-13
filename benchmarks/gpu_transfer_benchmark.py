import csv
import time
from pathlib import Path

import numpy as np
import cupy as cp
from scipy.sparse import csr_matrix

from backend.backends.cuda_backend import CUDABackend


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_FILE = OUTPUT_DIR / "gpu_transfer_benchmark.csv"

CASES = [
    (10_000, 10_000, 100_000),
    (25_000, 25_000, 500_000),
    (100_000, 100_000, 1_000_000),
    (100_000, 100_000, 5_000_000),
    (200_000, 200_000, 10_000_000),
]

REPEATS = 100
SEED = 777


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


def benchmark_case(case_no, rows, cols, target_nnz):

    print()
    print("=" * 70)
    print(
        f"Case {case_no}: "
        f"{rows:,} x {cols:,} | "
        f"target NNZ={target_nnz:,}"
    )
    print("=" * 70)

    rng = np.random.default_rng(
        SEED + case_no
    )

    A = make_sparse_matrix(
        rows,
        cols,
        target_nnz,
        SEED + case_no,
    )

    x = rng.standard_normal(
        cols
    ).astype(np.float64)

    actual_nnz = A.nnz

    # ========================================================
    # 1. CPU baseline
    # ========================================================

    for _ in range(10):
        A @ x

    start = time.perf_counter()

    for _ in range(REPEATS):
        cpu_result = A @ x

    cpu_total = time.perf_counter() - start
    cpu_per_run = cpu_total / REPEATS

    # ========================================================
    # 2. GPU CSR setup
    # ========================================================

    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()

    setup_start = time.perf_counter()

    backend = CUDABackend(A)

    cp.cuda.Stream.null.synchronize()

    setup_time = time.perf_counter() - setup_start

    # ========================================================
    # 3. H2D vector transfer
    # ========================================================

    h2d_times = []

    for _ in range(REPEATS):

        cp.cuda.Stream.null.synchronize()

        start = time.perf_counter()

        x_gpu = cp.asarray(x)

        cp.cuda.Stream.null.synchronize()

        elapsed = time.perf_counter() - start

        h2d_times.append(elapsed)

        del x_gpu

    h2d_per_run = float(
        np.mean(h2d_times)
    )

    # Keep GPU vector for kernel tests
    x_gpu = cp.asarray(x)

    cp.cuda.Stream.null.synchronize()

    # ========================================================
    # 4. GPU kernel
    # ========================================================

    for _ in range(10):
        backend.A @ x_gpu

    cp.cuda.Stream.null.synchronize()

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()

    for _ in range(REPEATS):
        y_gpu = backend.A @ x_gpu

    end_event.record()
    end_event.synchronize()

    kernel_total = (
        cp.cuda.get_elapsed_time(
            start_event,
            end_event,
        )
        / 1000.0
    )

    kernel_per_run = (
        kernel_total / REPEATS
    )

    # ========================================================
    # 5. D2H transfer
    # ========================================================

    cp.cuda.Stream.null.synchronize()

    d2h_times = []

    for _ in range(REPEATS):

        start = time.perf_counter()

        y_cpu = cp.asnumpy(y_gpu)

        cp.cuda.Stream.null.synchronize()

        elapsed = time.perf_counter() - start

        d2h_times.append(elapsed)

    d2h_per_run = float(
        np.mean(d2h_times)
    )

    # ========================================================
    # 6. Complete repeated GPU pipeline
    # ========================================================

    cp.cuda.Stream.null.synchronize()

    pipeline_start = time.perf_counter()

    for _ in range(REPEATS):

        x_temp = cp.asarray(x)

        y_temp = backend.A @ x_temp

        result_temp = cp.asnumpy(
            y_temp
        )

    cp.cuda.Stream.null.synchronize()

    pipeline_total = (
        time.perf_counter()
        - pipeline_start
    )

    pipeline_per_run = (
        pipeline_total / REPEATS
    )

    # ========================================================
    # 7. Calculations
    # ========================================================

    components_sum = (
        h2d_per_run
        + kernel_per_run
        + d2h_per_run
    )

    overhead_per_run = (
        pipeline_per_run
        - kernel_per_run
    )

    kernel_fraction = (
        kernel_per_run
        / pipeline_per_run
        if pipeline_per_run > 0
        else 0
    )

    transfer_fraction = (
        (
            h2d_per_run
            + d2h_per_run
        )
        / pipeline_per_run
        if pipeline_per_run > 0
        else 0
    )

    setup_amortized = (
        setup_time / REPEATS
    )

    total_amortized = (
        setup_amortized
        + pipeline_per_run
    )

    amortized_speedup = (
        cpu_per_run / total_amortized
        if total_amortized > 0
        else 0
    )

    max_error = float(
        np.max(
            np.abs(
                cpu_result - y_cpu
            )
        )
    )

    # ========================================================
    # 8. Print
    # ========================================================

    print(
        f"Actual NNZ       : {actual_nnz:,}"
    )

    print(
        f"CPU SpMV         : "
        f"{cpu_per_run * 1000:.4f} ms"
    )

    print(
        f"GPU setup        : "
        f"{setup_time * 1000:.4f} ms"
    )

    print(
        f"H2D              : "
        f"{h2d_per_run * 1000:.4f} ms"
    )

    print(
        f"GPU kernel       : "
        f"{kernel_per_run * 1000:.4f} ms"
    )

    print(
        f"D2H              : "
        f"{d2h_per_run * 1000:.4f} ms"
    )

    print(
        f"Complete pipeline: "
        f"{pipeline_per_run * 1000:.4f} ms"
    )

    print(
        f"Kernel fraction  : "
        f"{kernel_fraction * 100:.2f}%"
    )

    print(
        f"Transfer fraction: "
        f"{transfer_fraction * 100:.2f}%"
    )

    print(
        f"Amortized speedup: "
        f"{amortized_speedup:.2f}x"
    )

    print(
        f"Max error        : "
        f"{max_error:.3e}"
    )

    print(
        f"Correctness      : "
        f"{'PASS' if max_error <= 1e-10 else 'FAIL'}"
    )

    return {
        "case": case_no,
        "rows": rows,
        "columns": cols,
        "target_nnz": target_nnz,
        "actual_nnz": actual_nnz,
        "repeats": REPEATS,

        "cpu_spmv_seconds": cpu_per_run,

        "gpu_setup_seconds": setup_time,
        "gpu_setup_amortized_seconds": setup_amortized,

        "h2d_seconds": h2d_per_run,
        "gpu_kernel_seconds": kernel_per_run,
        "d2h_seconds": d2h_per_run,

        "component_sum_seconds": components_sum,
        "gpu_pipeline_seconds": pipeline_per_run,

        "transfer_overhead_seconds": (
            h2d_per_run + d2h_per_run
        ),

        "non_kernel_overhead_seconds": (
            overhead_per_run
        ),

        "kernel_fraction": kernel_fraction,
        "transfer_fraction": transfer_fraction,

        "amortized_total_seconds": total_amortized,
        "amortized_speedup": amortized_speedup,

        "max_absolute_error": max_error,

        "correctness": (
            "PASS"
            if max_error <= 1e-10
            else "FAIL"
        ),
    }


def main():

    print("=" * 70)
    print("ApexCUDA GPU Transfer Analysis")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for case_no, (rows, cols, nnz) in enumerate(
        CASES,
        start=1,
    ):

        try:

            result = benchmark_case(
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

    # ========================================================
    # CSV
    # ========================================================

    if results:

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

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("Transfer Analysis Complete")
    print("=" * 70)

    print(
        f"CSV: {OUTPUT_FILE.resolve()}"
    )

    print()

    print(
        f"{'NNZ':>12} "
        f"{'CPU':>10} "
        f"{'H2D':>10} "
        f"{'Kernel':>10} "
        f"{'D2H':>10} "
        f"{'Pipeline':>10} "
        f"{'Amortized':>10}"
    )

    print("-" * 82)

    for r in results:

        print(
            f"{r['actual_nnz']:>12,} "
            f"{r['cpu_spmv_seconds'] * 1000:>10.3f} "
            f"{r['h2d_seconds'] * 1000:>10.3f} "
            f"{r['gpu_kernel_seconds'] * 1000:>10.3f} "
            f"{r['d2h_seconds'] * 1000:>10.3f} "
            f"{r['gpu_pipeline_seconds'] * 1000:>10.3f} "
            f"{r['amortized_speedup']:>9.2f}x"
        )

    print()
    print("All measurements saved to CSV.")


if __name__ == "__main__":
    main()