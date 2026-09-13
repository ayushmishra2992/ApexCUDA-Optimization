from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def load_csv(filename):
    path = RESULTS / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def save_plot(filename):
    path = PLOTS / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Created: {filename}")


# ============================================================
# 1. SOLVER BENCHMARK
# ============================================================

print("\n[1/6] Solver benchmark")

df = load_csv("benchmark_results.csv")

# Recover instance name from MPS filename because instance column
# is currently blank in benchmark_results.csv.
df["instance"] = df["instance"].fillna("").astype(str).str.strip()

missing = df["instance"] == ""

df.loc[missing, "instance"] = (
    df.loc[missing, "file"]
    .astype(str)
    .apply(lambda x: Path(x).stem)
)

df["solve_time"] = pd.to_numeric(
    df["solve_time"],
    errors="coerce"
)

df["iterations"] = pd.to_numeric(
    df["iterations"],
    errors="coerce"
)

# Only use successful solves for timing comparison.
feasible = df[
    df["status"].astype(str).str.upper() == "FEASIBLE"
].copy()

if not feasible.empty:

    table = feasible.pivot_table(
        index="instance",
        columns="requested_backend",
        values="solve_time",
        aggfunc="first"
    )

    backend_order = [
        b for b in ["cpu", "cuda", "auto"]
        if b in table.columns
    ]

    table = table[backend_order]

    plt.figure(figsize=(10, 6))

    table.plot(
        kind="bar",
        ax=plt.gca(),
        logy=True
    )

    plt.title("ApexCUDA Solver Benchmark")
    plt.xlabel("Problem Instance")
    plt.ylabel("Solve Time (seconds, log scale)")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)

    save_plot("01_solver_time_cpu_cuda_auto.png")


# ============================================================
# 2. SPMV CPU VS GPU
# ============================================================

print("\n[2/6] Sparse SpMV CPU vs GPU")

df = load_csv("gpu_spmv_stress.csv")

df["actual_nnz"] = pd.to_numeric(
    df["actual_nnz"],
    errors="coerce"
)

# Convert seconds → milliseconds.
df["cpu_ms"] = (
    pd.to_numeric(df["cpu_spmv_seconds"], errors="coerce")
    * 1000
)

df["gpu_e2e_ms"] = (
    pd.to_numeric(
        df["gpu_end_to_end_seconds"],
        errors="coerce"
    )
    * 1000
)

df = df.dropna(
    subset=[
        "actual_nnz",
        "cpu_ms",
        "gpu_e2e_ms"
    ]
).sort_values("actual_nnz")


plt.figure(figsize=(10, 6))

plt.plot(
    df["actual_nnz"],
    df["cpu_ms"],
    marker="o",
    label="CPU SpMV"
)

plt.plot(
    df["actual_nnz"],
    df["gpu_e2e_ms"],
    marker="o",
    label="GPU End-to-End"
)

plt.title("Sparse SpMV: CPU vs GPU End-to-End")
plt.xlabel("Nonzeros (NNZ)")
plt.ylabel("Time (ms)")
plt.grid(True, alpha=0.3)
plt.legend()

save_plot("02_spmv_cpu_vs_gpu.png")


# ============================================================
# 3. GPU SPEEDUP
# ============================================================

print("\n[3/6] GPU speedup")

df["speedup"] = (
    pd.to_numeric(
        df["cpu_spmv_seconds"],
        errors="coerce"
    )
    /
    pd.to_numeric(
        df["gpu_end_to_end_seconds"],
        errors="coerce"
    )
)

plt.figure(figsize=(10, 6))

plt.plot(
    df["actual_nnz"],
    df["speedup"],
    marker="o"
)

plt.axhline(
    1.0,
    linestyle="--",
    label="1× baseline"
)

plt.title("GPU End-to-End SpMV Speedup")
plt.xlabel("Nonzeros (NNZ)")
plt.ylabel("Speedup (CPU / GPU)")
plt.grid(True, alpha=0.3)
plt.legend()

save_plot("03_spmv_gpu_speedup.png")


# ============================================================
# 4. GPU MEMORY
# ============================================================

print("\n[4/6] GPU memory scaling")

df = load_csv("gpu_memory_stress.csv")

df["actual_nnz"] = pd.to_numeric(
    df["actual_nnz"],
    errors="coerce"
)

df["gpu_memory_mb"] = pd.to_numeric(
    df["gpu_memory_after_mb"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "actual_nnz",
        "gpu_memory_mb"
    ]
).sort_values("actual_nnz")


plt.figure(figsize=(10, 6))

plt.plot(
    df["actual_nnz"],
    df["gpu_memory_mb"],
    marker="o"
)

plt.title("GPU CSR Memory Usage vs Problem Size")
plt.xlabel("Nonzeros (NNZ)")
plt.ylabel("GPU Memory (MB)")
plt.grid(True, alpha=0.3)

save_plot("04_gpu_memory_vs_nnz.png")


# ============================================================
# 5. CUDA TRANSFER / KERNEL BREAKDOWN
# ============================================================

print("\n[5/6] CUDA transfer and kernel breakdown")

df = load_csv("gpu_transfer_benchmark.csv")

df["actual_nnz"] = pd.to_numeric(
    df["actual_nnz"],
    errors="coerce"
)

# All source values are seconds → milliseconds.
df["h2d_ms"] = (
    pd.to_numeric(
        df["h2d_seconds"],
        errors="coerce"
    ) * 1000
)

df["kernel_ms"] = (
    pd.to_numeric(
        df["gpu_kernel_seconds"],
        errors="coerce"
    ) * 1000
)

df["d2h_ms"] = (
    pd.to_numeric(
        df["d2h_seconds"],
        errors="coerce"
    ) * 1000
)

df = df.dropna(
    subset=[
        "actual_nnz",
        "h2d_ms",
        "kernel_ms",
        "d2h_ms"
    ]
).sort_values("actual_nnz")


plt.figure(figsize=(10, 6))

plt.plot(
    df["actual_nnz"],
    df["h2d_ms"],
    marker="o",
    label="Host → Device"
)

plt.plot(
    df["actual_nnz"],
    df["kernel_ms"],
    marker="o",
    label="GPU Kernel"
)

plt.plot(
    df["actual_nnz"],
    df["d2h_ms"],
    marker="o",
    label="Device → Host"
)

plt.title("CUDA Execution Breakdown")
plt.xlabel("Nonzeros (NNZ)")
plt.ylabel("Time (ms)")
plt.grid(True, alpha=0.3)
plt.legend()

save_plot("05_cuda_transfer_kernel_breakdown.png")


# ============================================================
# 6. CUDA PROFILING
# ============================================================

print("\n[6/6] CUDA profiling")

df = load_csv("cuda_profile_report.csv")

df["actual_nnz"] = pd.to_numeric(
    df["actual_nnz"],
    errors="coerce"
)

# Average timings are already provided by the profiler.
df["matvec_ms"] = (
    pd.to_numeric(
        df["average_matvec_seconds"],
        errors="coerce"
    ) * 1000
)

df["rmatvec_ms"] = (
    pd.to_numeric(
        df["average_rmatvec_seconds"],
        errors="coerce"
    ) * 1000
)

df = df.dropna(
    subset=[
        "actual_nnz",
        "matvec_ms",
        "rmatvec_ms"
    ]
).sort_values("actual_nnz")


plt.figure(figsize=(10, 6))

plt.plot(
    df["actual_nnz"],
    df["matvec_ms"],
    marker="o",
    label="A @ x"
)

plt.plot(
    df["actual_nnz"],
    df["rmatvec_ms"],
    marker="o",
    label="Aᵀ @ y"
)

plt.title("CUDA Sparse Matrix-Vector Profiling")
plt.xlabel("Nonzeros (NNZ)")
plt.ylabel("Average Time (ms)")
plt.grid(True, alpha=0.3)
plt.legend()

save_plot("06_cuda_matvec_rmatvec_profile.png")


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("GRAPH GENERATION COMPLETE")
print("==========================================")

print(f"\nOutput directory:")
print(PLOTS.resolve())

print("\nGenerated files:")

for file in sorted(PLOTS.glob("*.png")):
    print(" -", file.name)