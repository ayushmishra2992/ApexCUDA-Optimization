# ==============================================================================
# APEXCUDA BACKEND API SERVER (SIH26119)
# ==============================================================================
# HONESTY NOTE:
# The full 472,798-variable / 3,443,145-constraint airline MILP is NOT solved
# live from this API (that run is minutes-scale, executed offline via the
# notebooks / backend.core pipeline). This server exposes:
#   - real, static, validated model/presolver/benchmark statistics
#   - genuine on-the-fly GPU/CPU detection
#   - a genuinely-executed, reproducible sparse SpMV micro-benchmark that runs
#     on whatever machine hosts this process (CPU always, GPU if a CUDA
#     device + CuPy are available)
#   - genuine small solves (via /api/solve-example) using the real
#     MPSParser -> Presolver -> IndiOptima -> MILP/PDHG pipeline
#   - a staged, clearly-labeled DEMONSTRATION of the disruption -> presolve ->
#     IndiOptima -> B&B/PDHG -> backend pipeline, which does NOT fabricate a
#     final optimized schedule.
# ==============================================================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import math
import numpy as np
import scipy.sparse as sp

app = FastAPI(
    title="IndiOptima / ApexCUDA Optimization Engine API",
    description="Indigenous GPU-Accelerated Optimization Solver API (SIH26119)",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# REAL, VALIDATED AIRLINE MODEL STATISTICS (SIH26119 dataset)
# ------------------------------------------------------------------------------
MODEL_STATS = {
    "flights": 789,
    "aircraft": 120,
    "crew_total": 400,
    "crew_available": 388,
    "crew_unavailable": 12,
    "bases": ["BLR", "BOM", "CCU", "DEL", "HYD"],
    "variables": 472798,
    "constraints": 3443145,
    "non_zeros": 7898666,
    "sparse_memory_mb": 103.53,
    "dense_memory_tb": 11.84,
}

PRESOLVER_STATS = {
    "aircraft_candidates_raw": 31247,
    "aircraft_candidates_feasible": 26488,
    "aircraft_candidates_removed": 4759,
    "flights_with_feasible_candidate": 789,
    "flights_total": 789,
    "crew_flight_candidates_raw": 106264,
    "crew_flight_candidates_feasible": 28679,
    "crew_conflict_constraints": 3364254,
    "aircraft_sequence_arcs": 338468,
}

VALIDATED_BENCHMARK = {
    "label": "GPU Sparse SpMV Benchmark (validated reference run)",
    "note": "Kernel-level sparse matrix-vector benchmark; NOT end-to-end MILP solve time.",
    "cpu_spmv_ms": 43.511,
    "gpu_spmv_ms": 1.216,
    "speedup": 35.79,
    "max_numerical_error": 4.26e-14,
    "initial_transfer_ms": 860.77,
    "breakeven_operations": 21,
}

OBJECTIVE = {
    "cancellation_cost_per_flight_inr": 500000,
    "delay_cost_per_minute_inr": 1000,
}

ARCHITECTURE_STAGES = [
    {"id": "application", "layer": "application", "label": "Airline Formulation",
     "detail": "Domain-specific model builder for the airline schedule-recovery demonstration workload."},
    {"id": "opt_model", "layer": "solver", "label": "Generic OptimizationModel",
     "detail": "Application-independent variables/constraints/objective container (sparse CSR)."},
    {"id": "presolver", "layer": "solver", "label": "Generic Presolver",
     "detail": "Removes infeasible/redundant candidates before the model reaches the solver."},
    {"id": "indioptima", "layer": "solver", "label": "IndiOptima",
     "detail": "The generic sovereign optimization solver — application-independent of the airline formulation."},
    {"id": "milp", "layer": "solver", "label": "MILP Solver / Branch-and-Bound",
     "detail": "Handles the integer/MILP decisions by exploring LP relaxations."},
    {"id": "pdhg", "layer": "solver", "label": "PDHG (LP relaxation core)",
     "detail": "Primal-Dual Hybrid Gradient method solves the continuous LP relaxations at each B&B node."},
    {"id": "backend", "layer": "hardware", "label": "CPU / CUDA Numerical Backend",
     "detail": "Sparse SpMV / linear-algebra kernels, dispatched to GPU when available and beneficial."},
    {"id": "solution", "layer": "solver", "label": "Solution + Verification",
     "detail": "Feasibility/optimality verification of the returned solution."},
]


def get_model_stats():
    return dict(MODEL_STATS)


@app.get("/api/health")
def health_check():
    return {
        "status": "ONLINE",
        "solver": "IndiOptima (MILP: Branch-and-Bound over PDHG LP relaxations)",
        "note": "Backend API is live. Full-scale MILP solve is not executed on-demand from this endpoint.",
    }


@app.get("/api/model-stats")
def model_stats():
    return {"model": get_model_stats(), "presolver": PRESOLVER_STATS, "objective": OBJECTIVE}


@app.get("/api/architecture")
def architecture():
    return {"stages": ARCHITECTURE_STAGES}


# ------------------------------------------------------------------------------
# GENUINE GPU DETECTION (no hardcoded GPU model/specs)
# ------------------------------------------------------------------------------
def detect_gpu():
    try:
        import cupy as cp  # type: ignore
        count = cp.cuda.runtime.getDeviceCount()
        if count <= 0:
            return {"cuda_available": False, "reason": "No CUDA devices found."}
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode(errors="ignore")
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        runtime_version = cp.cuda.runtime.runtimeGetVersion()
        return {
            "cuda_available": True,
            "device_name": name,
            "device_count": count,
            "cuda_runtime_version": runtime_version,
            "vram_total_mb": round(total_b / (1024 ** 2), 1),
            "vram_free_mb": round(free_b / (1024 ** 2), 1),
            "cupy_version": cp.__version__,
        }
    except Exception as exc:
        return {"cuda_available": False, "reason": f"CuPy/CUDA not usable: {exc.__class__.__name__}"}


@app.get("/api/gpu-status")
def gpu_status():
    return detect_gpu()


# ------------------------------------------------------------------------------
# GENUINE, REPRODUCIBLE LIVE SPARSE SpMV MICRO-BENCHMARK
# ------------------------------------------------------------------------------
class BenchmarkRequest(BaseModel):
    backend: str = "cpu"
    operations: int = 20


def build_demo_matrix(seed=42, rows=60000, cols=60000, density=0.00074):
    rng = np.random.default_rng(seed)
    mat = sp.random(rows, cols, density=density, format="csr", random_state=rng, dtype=np.float64)
    vec = rng.standard_normal(cols)
    return mat, vec


@app.post("/api/benchmark-live")
def benchmark_live(req: BenchmarkRequest):
    ops = max(1, min(req.operations, 500))
    matrix, vec = build_demo_matrix()

    gpu_info = detect_gpu()
    requested_backend = req.backend.lower()
    actual_backend = "cpu"
    fallback_reason = None
    transfer_ms = None

    if requested_backend == "gpu":
        if gpu_info.get("cuda_available"):
            actual_backend = "gpu"
        else:
            fallback_reason = gpu_info.get("reason", "GPU not available on this machine.")

    if actual_backend == "cpu":
        t0 = time.perf_counter()
        for _ in range(ops):
            _ = matrix @ vec
        elapsed_s = time.perf_counter() - t0
    else:
        import cupy as cp  # type: ignore
        import cupyx.scipy.sparse as cusparse  # type: ignore

        t_transfer0 = time.perf_counter()
        gpu_matrix = cusparse.csr_matrix(matrix)
        gpu_vec = cp.asarray(vec)
        cp.cuda.Stream.null.synchronize()
        transfer_ms = round((time.perf_counter() - t_transfer0) * 1000, 4)

        t0 = time.perf_counter()
        for _ in range(ops):
            _ = gpu_matrix @ gpu_vec
        cp.cuda.Stream.null.synchronize()
        elapsed_s = time.perf_counter() - t0

    return {
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "fallback_reason": fallback_reason,
        "operations": ops,
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "total_time_ms": round(elapsed_s * 1000, 4),
        "per_operation_ms": round((elapsed_s * 1000) / ops, 4),
        "transfer_ms": transfer_ms,
        "is_live_measurement": True,
        "label": "LIVE measurement on this machine (not the validated reference numbers)",
    }


@app.get("/api/benchmark-data")
def get_benchmark_data():
    return VALIDATED_BENCHMARK


# ------------------------------------------------------------------------------
# DISRUPTION DEMONSTRATION (staged pipeline, no fabricated final schedule)
# ------------------------------------------------------------------------------
@app.post("/api/trigger-disruption")
def trigger_disruption():
    stats = get_model_stats()
    stages = [
        {"id": "disruption", "title": "Disruption Injected",
         "detail": "Demonstration scenario: DEL-hub weather disruption affecting scheduled departures."},
        {"id": "model_update", "title": "Optimization Model Updated",
         "detail": f"Airline formulation regenerates the sparse model: {stats['variables']:,} variables, "
                   f"{stats['constraints']:,} constraints, {stats['non_zeros']:,} non-zeros."},
        {"id": "presolve", "title": "Generic Presolver",
         "detail": (f"{PRESOLVER_STATS['aircraft_candidates_raw']:,} raw aircraft candidates -> "
                    f"{PRESOLVER_STATS['aircraft_candidates_feasible']:,} feasible "
                    f"({PRESOLVER_STATS['aircraft_candidates_removed']:,} unreachable removed). "
                    f"{PRESOLVER_STATS['flights_with_feasible_candidate']}/{PRESOLVER_STATS['flights_total']} "
                    f"flights retain a feasible candidate.")},
        {"id": "indioptima", "title": "IndiOptima Dispatch",
         "detail": "Generic solver receives the presolved model; application-independent of the airline formulation."},
        {"id": "bnb_pdhg", "title": "Branch-and-Bound over PDHG LP Relaxations",
         "detail": "Integer decisions explored via Branch-and-Bound; each LP relaxation solved with PDHG."},
        {"id": "backend", "title": "CPU / CUDA Numerical Backend",
         "detail": "Sparse SpMV kernels dispatched to GPU when available and beneficial (see GPU Benchmark)."},
    ]
    return {
        "success": True,
        "solver_executed": False,
        "final_schedule_available": False,
        "message": ("This demonstration shows the real pipeline stages and real presolver statistics for the "
                    "789-flight model. The full 472,798-variable MILP is not solved on-demand from this API "
                    "(offline solves take minutes), so no fabricated final recovery schedule is shown."),
        "stages": stages,
        "model": stats,
        "presolver": PRESOLVER_STATS,
    }


# ------------------------------------------------------------------------------
# GENUINE SMALL SOLVES using the real solver pipeline (bundled tiny .mps files)
# ------------------------------------------------------------------------------
def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


EXAMPLES = {
    "tiny_milp": "examples/tiny_milp.mps",
    "tiny_binary": "examples/tiny_binary.mps",
    "tiny_max": "examples/tiny_max.mps",
    "afiro": "examples/afiro.mps",
}


@app.get("/api/examples")
def list_examples():
    import os
    return {"examples": [k for k in EXAMPLES if os.path.exists(EXAMPLES[k])]}


class SolveExampleRequest(BaseModel):
    example: str = "tiny_milp"
    backend: str = "cpu"


@app.post("/api/solve-example")
def solve_example(req: SolveExampleRequest):
    import os
    path = EXAMPLES.get(req.example)
    if not path or not os.path.exists(path):
        return {"success": False, "error": f"Unknown or missing example '{req.example}'"}

    gpu_info = detect_gpu()
    use_gpu = req.backend.lower() == "gpu" and gpu_info.get("cuda_available")
    fallback_reason = None
    if req.backend.lower() == "gpu" and not use_gpu:
        fallback_reason = gpu_info.get("reason", "GPU not available on this machine.")

    try:
        from backend.formats.mps_parser import MPSParser
        from backend.core.presolver import GenericPresolver
        from backend.core.milp_solver import MILPSolver

        t0 = time.perf_counter()
        model = MPSParser().parse(path)
        parse_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        presolve_result = GenericPresolver().presolve(model)
        presolved_model = presolve_result.model
        presolve_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        solver = MILPSolver(backend="gpu" if use_gpu else "cpu")
        result = solver.solve(presolved_model)
        solve_ms = (time.perf_counter() - t2) * 1000

        return _sanitize({
            "success": True,
            "example": req.example,
            "requested_backend": req.backend,
            "actual_backend": "gpu" if use_gpu else "cpu",
            "fallback_reason": fallback_reason,
            "status": getattr(result, "status", str(result)),
            "objective_value": getattr(result, "objective", None),
            "iterations": getattr(result, "iterations", None),
            "variables_removed": presolve_result.variables_removed,
            "constraints_removed": presolve_result.constraints_removed,
            "parse_ms": round(parse_ms, 3),
            "presolve_ms": round(presolve_ms, 3),
            "solve_ms": round(solve_ms, 3),
            "is_live_measurement": True,
        })
    except Exception as exc:
        return {"success": False, "error": f"{exc.__class__.__name__}: {exc}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
