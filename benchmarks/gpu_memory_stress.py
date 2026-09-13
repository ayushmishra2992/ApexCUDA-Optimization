import gc
import csv
import time
from pathlib import Path

import numpy as np
import cupy as cp
from scipy.sparse import csr_matrix

from backend.backends.cuda_backend import CUDABackend


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_FILE = OUTPUT_DIR / "gpu_memory_stress.csv"

# Keep this conservative for your RTX 3050 Laptop GPU (~4 GB VRAM).
CASES = [
    (50_000, 50_000, 500_000),
    (100_000, 100_000, 1_000_000),
    (100_000, 100_000, 5_000_000),
    (200_000, 200_000, 10_000_000),
    (300_000, 300_000, 20_000_000),
    (500_000, 500_000, 30_000_000),
]

SEED = 1234


def make_sparse_matrix(rows, cols, nnz, seed):
    rng = np.random.default_rng(seed)

    row = rng.integers(
        0, rows, size=nnz, dtype=np.int32
    )

    col = rng.integers(
        0, cols, size=nnz, dtype=np.int32
    )

    data = rng.standard_normal(nnz).astype(np.float64)

    A = csr_matrix(
        (data, (row, col)),
        shape=(rows, cols),
    )

    A.sum_duplicates()
    A.sort_indices()

    return A


def memory_bytes():
    try:
        return int(cp.get_default_memory_pool().used_bytes())
    except Exception:
        return 0


def mb(value):
    return value / (1024 ** 2)


def gpu_name():
    props = cp.cuda.runtime.getDeviceProperties(
        cp.cuda.Device().id
    )

    name = props["name"]

    if isinstance(name, bytes):
        name = name.decode()

    return name


def run_case(case_no, rows, cols, target_nnz):

    print()
    print("=" * 70)
    print(
        f"Case {case_no}: "
        f"{rows:,} x {cols:,} | "
        f"target NNZ = {target_nnz:,}"
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # Clean GPU memory
    # ---------------------------------------------------------

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()

    baseline = memory_bytes()

    # ---------------------------------------------------------
    # Create CPU matrix
    # ---------------------------------------------------------

    print("Creating sparse matrix...")

    start = time.perf_counter()

    A = make_sparse_matrix(
        rows,
        cols,
        target_nnz,
        SEED + case_no,
    )

    generation_time = time.perf_counter() - start

    actual_nnz = A.nnz

    print(f"Actual NNZ      : {actual_nnz:,}")
    print(
        f"CPU CSR memory  : "
        f"{A.data.nbytes + A.indices.nbytes + A.indptr.nbytes:,} bytes"
    )

    cpu_memory = (
        A.data.nbytes
        + A.indices.nbytes
        + A.indptr.nbytes
    )

    # ---------------------------------------------------------
    # Upload / CUDA CSR construction
    # ---------------------------------------------------------

    print("Uploading CSR to GPU...")

    setup_start = time.perf_counter()

    try:

        backend = CUDABackend(A)

        cp.cuda.Stream.null.synchronize()

        setup_time = time.perf_counter() - setup_start

    except Exception as exc:

        print(
            f"GPU allocation FAILED: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "case": case_no,
            "rows": rows,
            "columns": cols,
            "target_nnz": target_nnz,
            "actual_nnz": actual_nnz,
            "cpu_csr_bytes": cpu_memory,
            "cpu_csr_mb": mb(cpu_memory),
            "gpu_memory_before_bytes": baseline,
            "gpu_memory_after_bytes": 0,
            "gpu_memory_after_mb": 0,
            "gpu_memory_increase_bytes": 0,
            "gpu_memory_increase_mb": 0,
            "gpu_memory_per_nnz_bytes": 0,
            "setup_seconds": 0,
            "generation_seconds": generation_time,
            "status": "GPU_MEMORY_ERROR",
            "error": str(exc),
        }

    after = memory_bytes()

    gpu_increase = max(
        0,
        after - baseline,
    )

    bytes_per_nnz = (
        gpu_increase / actual_nnz
        if actual_nnz > 0
        else 0
    )

    print(
        f"GPU CSR memory  : "
        f"{mb(gpu_increase):.2f} MB"
    )

    print(
        f"Memory / NNZ    : "
        f"{bytes_per_nnz:.2f} bytes"
    )

    print(
        f"Setup time      : "
        f"{setup_time:.4f} s"
    )

    # ---------------------------------------------------------
    # Allocate a vector too
    # ---------------------------------------------------------

    print("Allocating GPU vector...")

    x = cp.zeros(
        cols,
        dtype=cp.float64,
    )

    cp.cuda.Stream.null.synchronize()

    after_vector = memory_bytes()

    vector_increase = max(
        0,
        after_vector - after,
    )

    print(
        f"Vector memory   : "
        f"{mb(vector_increase):.2f} MB"
    )

    # ---------------------------------------------------------
    # One SpMV to ensure actual GPU usage
    # ---------------------------------------------------------

    print("Running one GPU SpMV...")

    try:

        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()

        y = backend.A @ x

        end.record()
        end.synchronize()

        spmv_ms = cp.cuda.get_elapsed_time(
            start,
            end,
        )

        cp.cuda.Stream.null.synchronize()

        status = "PASS"
        error = ""

    except Exception as exc:

        spmv_ms = 0.0
        status = "GPU_SPMV_ERROR"
        error = str(exc)

        print(
            f"SpMV failed: "
            f"{type(exc).__name__}: {exc}"
        )

    final_memory = memory_bytes()

    print(
        f"GPU memory total: "
        f"{mb(final_memory):.2f} MB"
    )

    print(
        f"SpMV time       : "
        f"{spmv_ms:.4f} ms"
    )

    print(f"Status          : {status}")

    result = {
        "case": case_no,
        "rows": rows,
        "columns": cols,
        "target_nnz": target_nnz,
        "actual_nnz": actual_nnz,

        "cpu_csr_bytes": cpu_memory,
        "cpu_csr_mb": mb(cpu_memory),

        "gpu_memory_before_bytes": baseline,
        "gpu_memory_after_bytes": after,
        "gpu_memory_after_mb": mb(after),

        "gpu_memory_increase_bytes": gpu_increase,
        "gpu_memory_increase_mb": mb(gpu_increase),

        "gpu_memory_per_nnz_bytes": bytes_per_nnz,

        "vector_memory_bytes": vector_increase,
        "vector_memory_mb": mb(vector_increase),

        "final_gpu_memory_bytes": final_memory,
        "final_gpu_memory_mb": mb(final_memory),

        "setup_seconds": setup_time,
        "generation_seconds": generation_time,

        "spmv_ms": spmv_ms,

        "status": status,
        "error": error,
    }

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    del x
    del backend
    del A

    try:
        del y
    except UnboundLocalError:
        pass

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()

    gc.collect()

    return result


def main():

    print("=" * 70)
    print("ApexCUDA GPU Memory Scaling Benchmark")
    print("=" * 70)

    print(f"GPU: {gpu_name()}")
    print("")

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

            result = run_case(
                case_no,
                rows,
                cols,
                nnz,
            )

            results.append(result)

        except MemoryError as exc:

            print(
                f"CASE {case_no}: "
                f"CPU memory allocation failed"
            )

            results.append({
                "case": case_no,
                "rows": rows,
                "columns": cols,
                "target_nnz": nnz,
                "status": "CPU_MEMORY_ERROR",
                "error": str(exc),
            })

        except Exception as exc:

            print(
                f"CASE {case_no}: "
                f"{type(exc).__name__}: {exc}"
            )

            results.append({
                "case": case_no,
                "rows": rows,
                "columns": cols,
                "target_nnz": nnz,
                "status": "ERROR",
                "error": str(exc),
            })

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    if results:

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

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("GPU Memory Benchmark Complete")
    print("=" * 70)

    print(
        f"CSV: {OUTPUT_FILE.resolve()}"
    )

    print()

    print(
        f"{'NNZ':>12} "
        f"{'GPU MB':>12} "
        f"{'Bytes/NNZ':>12} "
        f"{'SpMV ms':>12} "
        f"{'Status':>20}"
    )

    print("-" * 72)

    for r in results:

        print(
            f"{r.get('actual_nnz', 0):>12,} "
            f"{r.get('gpu_memory_increase_mb', 0):>12.2f} "
            f"{r.get('gpu_memory_per_nnz_bytes', 0):>12.2f} "
            f"{r.get('spmv_ms', 0):>12.4f} "
            f"{r.get('status', 'UNKNOWN'):>20}"
        )


if __name__ == "__main__":
    main()