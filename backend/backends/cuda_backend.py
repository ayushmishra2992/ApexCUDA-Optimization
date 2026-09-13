import time

import cupy as cp
import cupyx.scipy.sparse as cusparse
import numpy as np


class CUDABackend:
    """
    CUDA sparse backend using CuPy/cuSPARSE.

    Profiling is optional and disabled by default so normal solver
    execution is not affected.
    """

    def __init__(self, A, profiling=False):
        if not hasattr(A, "tocsr"):
            raise TypeError("A must be a SciPy sparse matrix.")

        self.profiling = bool(profiling)

        self.matvec_calls = 0
        self.rmatvec_calls = 0

        self.matvec_time = 0.0
        self.rmatvec_time = 0.0

        self.setup_time = 0.0
        self.sync_time = 0.0

        self.setup_memory_used = 0
        self.peak_memory_used = 0

        setup_start = time.perf_counter()

        # Convert once and keep the sparse matrix resident on GPU.
        self.A = cusparse.csr_matrix(A)

        # Make sure GPU setup is complete before recording setup time.
        cp.cuda.Stream.null.synchronize()

        self.setup_time = time.perf_counter() - setup_start

        self._update_memory_stats()

    def _update_memory_stats(self):
        """Record current CuPy memory-pool usage."""
        try:
            pool = cp.get_default_memory_pool()
            used = int(pool.used_bytes())

            self.peak_memory_used = max(
                self.peak_memory_used,
                used
            )

            if self.setup_memory_used == 0:
                self.setup_memory_used = used

        except Exception:
            # Profiling must never break the solver.
            pass

    def matvec(self, x):
        """
        Compute A @ x.

        When profiling is disabled this follows the normal fast path.
        """
        x_gpu = cp.asarray(x)

        if not self.profiling:
            return self.A @ x_gpu

        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()

        result = self.A @ x_gpu

        end.record()
        end.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start, end)

        self.matvec_calls += 1
        self.matvec_time += elapsed_ms / 1000.0

        self._update_memory_stats()

        return result

    def rmatvec(self, x):
        """
        Compute A.T @ x.

        When profiling is disabled this follows the normal fast path.
        """
        x_gpu = cp.asarray(x)

        if not self.profiling:
            return self.A.T @ x_gpu

        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()

        result = self.A.T @ x_gpu

        end.record()
        end.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start, end)

        self.rmatvec_calls += 1
        self.rmatvec_time += elapsed_ms / 1000.0

        self._update_memory_stats()

        return result

    def to_cpu(self, x):
        """Transfer a GPU array back to CPU."""
        if not self.profiling:
            return cp.asnumpy(x)

        start = time.perf_counter()

        result = cp.asnumpy(x)

        cp.cuda.Stream.null.synchronize()

        self.sync_time += time.perf_counter() - start

        return result

    def synchronize(self):
        """Synchronize the default CUDA stream."""
        start = time.perf_counter()

        cp.cuda.Stream.null.synchronize()

        if self.profiling:
            self.sync_time += time.perf_counter() - start

        self._update_memory_stats()

    def reset_profile(self):
        """Reset collected profiling counters."""
        self.matvec_calls = 0
        self.rmatvec_calls = 0

        self.matvec_time = 0.0
        self.rmatvec_time = 0.0

        self.sync_time = 0.0

    def profile_summary(self):
        """
        Return collected GPU profiling information.
        """
        total_spmv_time = (
            self.matvec_time +
            self.rmatvec_time
        )

        return {
            "profiling_enabled": self.profiling,
            "setup_time_seconds": float(self.setup_time),

            "matvec_calls": int(self.matvec_calls),
            "rmatvec_calls": int(self.rmatvec_calls),

            "matvec_time_seconds": float(self.matvec_time),
            "rmatvec_time_seconds": float(self.rmatvec_time),

            "total_spmv_time_seconds": float(total_spmv_time),

            "sync_time_seconds": float(self.sync_time),

            "setup_memory_bytes": int(self.setup_memory_used),
            "peak_memory_bytes": int(self.peak_memory_used),

            "gpu_name": cp.cuda.runtime.getDeviceProperties(
                cp.cuda.Device().id
            )["name"].decode(
                "utf-8"
            )
        }