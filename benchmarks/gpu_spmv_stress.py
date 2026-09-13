# benchmarks/gpu_spmv_stress.py

import csv
import gc
import time
from pathlib import Path

import numpy as np
import cupy as cp
from scipy.sparse import csr_matrix

from backend.backends.cuda_backend import CUDABackend


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_FILE = OUTPUT_DIR / "gpu_spmv_stress.csv"

# Rows, columns, approximate NNZ
CASES = [
    (10_000, 10_000, 100_000),
    (25_000, 25_000, 500_000),
    (100_000, 100_000, 1_000_000),
    (100_000, 100_000, 5_000_000),
    (200_000, 200_000, 10_000_000),
]

REPEATS = 100
WARMUP = 10
SEED = 42


# ============================================================
# Utilities
# ============================================================

def make_sparse_matrix(rows, cols, nnz, seed=42):
    """
    Generate a sparse CSR matrix without using scipy.sparse.random,
    avoiding large temporary allocations.
    """

    rng = np.random.default_rng(seed)

    row_indices = rng.integers(
        0,
        rows,
        size=nnz,
        dtype=np.int32,
    )

    col_indices = rng.integers(
        0,
        cols,
        size=nnz,
        dtype=np.int32,
    )

    data = rng.standard_normal(nnz).astype(np.float64)

    A = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(rows, cols),
        dtype=np.float64,
    )

    A.sum_duplicates()
    A.sort_indices()

    return A


def get_gpu_memory():
    """
    Return currently used GPU memory from CuPy's memory pool.
    """

    try:
        pool = cp.get_default_memory_pool()
        return int(pool.used_bytes())
    except Exception:
        return 0


def get_gpu_name():
    try:
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props["name"]

        if isinstance(name, bytes):
            name = name.decode("utf-8")

        return name

    except Exception:
        return "Unknown GPU"


def cpu_spmv(A, x, repeats):
    """
    Benchmark CPU sparse matrix-vector multiplication.
    """

    # Warmup
    for _ in range(WARMUP):
        A @ x

    start = time.perf_counter()

    y = None

    for _ in range(repeats):
        y = A @ x

    elapsed = time.perf_counter() - start

    return elapsed / repeats, y


def gpu_spmv(A, x, repeats):
    """
    Benchmark GPU sparse matrix-vector multiplication.

    Measures:
        1. CSR setup
        2. H2D transfer
        3. GPU kernel time
        4. End-to-end GPU execution
    """

    # Clear unused CuPy allocations
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()

    memory_before = get_gpu_memory()

    # --------------------------------------------------------
    # GPU setup / CSR conversion
    # --------------------------------------------------------

    setup_start = time.perf_counter()

    backend = CUDABackend(A)

    cp.cuda.Stream.null.synchronize()

    setup_time = time.perf_counter() - setup_start

    memory_after_setup = get_gpu_memory()

    # --------------------------------------------------------
    # H2D vector transfer
    # --------------------------------------------------------

    transfer_start = time.perf_counter()

    x_gpu = cp.asarray(x)

    cp.cuda.Stream.null.synchronize()

    h2d_time = time.perf_counter() - transfer_start

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(WARMUP):
        backend.A @ x_gpu

    cp.cuda.Stream.null.synchronize()

    # --------------------------------------------------------
    # GPU kernel timing
    # --------------------------------------------------------

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()

    for _ in range(repeats):
        y_gpu = backend.A @ x_gpu

    end_event.record()

    end_event.synchronize()

    kernel_time = cp.cuda.get_elapsed_time(
        start_event,
        end_event,
    ) / 1000.0

    kernel_time_per_run = kernel_time / repeats

    # --------------------------------------------------------
    # GPU → CPU transfer
    # --------------------------------------------------------

    transfer_back_start = time.perf_counter()

    y_cpu = cp.asnumpy(y_gpu)

    cp.cuda.Stream.null.synchronize()

    d2h_time = time.perf_counter() - transfer_back_start

    # --------------------------------------------------------
    # End-to-end
    # --------------------------------------------------------

    end_to_end_start = time.perf_counter()

    x_gpu_e2e = cp.asarray(x)

    for _ in range(repeats):
        y_gpu_e2e = backend.A @ x_gpu_e2e

    cp.cuda.Stream.null.synchronize()

    y_cpu_e2e = cp.asnumpy(y_gpu_e2e)

    cp.cuda.Stream.null.synchronize()

    end_to_end_time = time.perf_counter() - end_to_end_start

    end_to_end_per_run = end_to_end_time / repeats

    peak_memory = max(
        memory_before,
        memory_after_setup,
        get_gpu_memory(),
    )

    return {
        "backend": backend,
        "kernel_time": kernel_time_per_run,
        "setup_time": setup_time,
        "h2d_time": h2d_time,
        "d2h_time": d2h_time,
        "end_to_end_time": end_to_end_per_run,
        "gpu_memory_bytes": peak_memory,
        "gpu_result": y_cpu,
    }


# ============================================================
# Main benchmark
# ============================================================

def run_case(rows, cols, target_nnz, case_number):

    print()
    print("=" * 65)
    print(
        f"Case {case_number}: "
        f"{rows:,} x {cols:,} | "
        f"target NNZ = {target_nnz:,}"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # Generate matrix
    # --------------------------------------------------------

    print("Generating sparse matrix...")

    A = make_sparse_matrix(
        rows,
        cols,
        target_nnz,
        seed=SEED + case_number,
    )

    actual_nnz = A.nnz

    print(f"Actual NNZ : {actual_nnz:,}")

    # --------------------------------------------------------
    # Vector
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED + case_number)

    x = rng.standard_normal(cols).astype(np.float64)

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    print("Running CPU SpMV...")

    cpu_time, cpu_result = cpu_spmv(
        A,
        x,
        REPEATS,
    )

    print(
        f"CPU SpMV  : "
        f"{cpu_time * 1000:.4f} ms"
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    print("Running CUDA SpMV...")

    gpu = gpu_spmv(
        A,
        x,
        REPEATS,
    )

    print(
        f"GPU kernel: "
        f"{gpu['kernel_time'] * 1000:.4f} ms"
    )

    print(
        f"GPU setup : "
        f"{gpu['setup_time']:.4f} s"
    )

    print(
        f"H2D       : "
        f"{gpu['h2d_time'] * 1000:.4f} ms"
    )

    print(
        f"D2H       : "
        f"{gpu['d2h_time'] * 1000:.4f} ms"
    )

    print(
        f"GPU E2E   : "
        f"{gpu['end_to_end_time'] * 1000:.4f} ms"
    )

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    max_error = float(
        np.max(
            np.abs(
                cpu_result - gpu["gpu_result"]
            )
        )
    )

    relative_error = float(
        np.linalg.norm(
            cpu_result - gpu["gpu_result"]
        )
        /
        max(
            np.linalg.norm(cpu_result),
            1e-30,
        )
    )

    correctness = (
        max_error <= 1e-10
        or relative_error <= 1e-10
    )

    # --------------------------------------------------------
    # Speedups
    # --------------------------------------------------------

    kernel_speedup = (
        cpu_time / gpu["kernel_time"]
        if gpu["kernel_time"] > 0
        else 0.0
    )

    e2e_speedup = (
        cpu_time / gpu["end_to_end_time"]
        if gpu["end_to_end_time"] > 0
        else 0.0
    )

    # Amortized GPU time after setup.
    amortized_gpu_time = (
        gpu["setup_time"] / REPEATS
        + gpu["end_to_end_time"]
    )

    amortized_speedup = (
        cpu_time / amortized_gpu_time
        if amortized_gpu_time > 0
        else 0.0
    )

    gpu_memory_mb = (
        gpu["gpu_memory_bytes"]
        / (1024 ** 2)
    )

    print()
    print(f"Kernel speedup    : {kernel_speedup:.2f}x")
    print(f"End-to-end speedup: {e2e_speedup:.2f}x")
    print(f"Amortized speedup : {amortized_speedup:.2f}x")
    print(f"Max error         : {max_error:.3e}")
    print(f"Relative error    : {relative_error:.3e}")
    print(
        f"Correctness       : "
        f"{'PASS' if correctness else 'FAIL'}"
    )
    print(f"GPU memory        : {gpu_memory_mb:.2f} MB")

    return {
        "case": case_number,
        "rows": rows,
        "columns": cols,
        "target_nnz": target_nnz,
        "actual_nnz": actual_nnz,
        "repeats": REPEATS,
        "cpu_spmv_seconds": cpu_time,
        "gpu_kernel_seconds": gpu["kernel_time"],
        "gpu_setup_seconds": gpu["setup_time"],
        "h2d_seconds": gpu["h2d_time"],
        "d2h_seconds": gpu["d2h_time"],
        "gpu_end_to_end_seconds": gpu["end_to_end_time"],
        "amortized_gpu_seconds": amortized_gpu_time,
        "kernel_speedup": kernel_speedup,
        "end_to_end_speedup": e2e_speedup,
        "amortized_speedup": amortized_speedup,
        "max_absolute_error": max_error,
        "relative_error": relative_error,
        "correctness": "PASS" if correctness else "FAIL",
        "gpu_memory_bytes": gpu["gpu_memory_bytes"],
        "gpu_memory_mb": gpu_memory_bytes_to_mb(
            gpu["gpu_memory_bytes"]
        ),
        "gpu_name": get_gpu_name(),
    }


def gpu_memory_bytes_to_mb(value):
    return value / (1024 ** 2)


# ============================================================
# Entry point
# ============================================================

def main():

    print("=" * 65)
    print("ApexCUDA GPU SpMV Stress Benchmark")
    print("=" * 65)

    print(f"GPU      : {get_gpu_name()}")
    print(f"Repeats  : {REPEATS}")
    print(f"Warmup   : {WARMUP}")
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for case_number, (rows, cols, nnz) in enumerate(
        CASES,
        start=1,
    ):

        try:

            result = run_case(
                rows,
                cols,
                nnz,
                case_number,
            )

            results.append(result)

        except Exception as exc:

            print()
            print(
                f"CASE {case_number} FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

        finally:

            gc.collect()

            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    if results:

        fieldnames = list(results[0].keys())

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
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("GPU SpMV Benchmark Complete")
    print("=" * 65)

    print(
        f"Successful cases: "
        f"{len(results)}/{len(CASES)}"
    )

    print(
        f"CSV: {OUTPUT_FILE.resolve()}"
    )

    if results:

        print()
        print(
            f"{'NNZ':>12} "
            f"{'CPU(ms)':>12} "
            f"{'GPU(ms)':>12} "
            f"{'Kernel':>10} "
            f"{'E2E':>10} "
            f"{'Error':>12}"
        )

        print("-" * 72)

        for r in results:

            print(
                f"{r['actual_nnz']:>12,} "
                f"{r['cpu_spmv_seconds'] * 1000:>12.4f} "
                f"{r['gpu_kernel_seconds'] * 1000:>12.4f} "
                f"{r['kernel_speedup']:>9.2f}x "
                f"{r['end_to_end_speedup']:>9.2f}x "
                f"{r['max_absolute_error']:>12.3e}"
            )


if __name__ == "__main__":
    main()