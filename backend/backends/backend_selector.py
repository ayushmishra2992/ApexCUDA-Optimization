import importlib.util


class BackendSelector:
    """
    Cost-aware CPU/CUDA backend selector.

    The selector estimates the CPU and GPU cost of repeated sparse
    matrix-vector operations and chooses the cheaper backend.

    Modes:
        "cpu"   -> force CPU
        "cuda"  -> force CUDA
        "auto"  -> choose CPU or CUDA automatically
    """

    def __init__(
        self,
        min_gpu_nnz=100_000,
        min_gpu_operations=20,
        expected_iterations=100,
        gpu_setup_ms=1271.157700,
        cpu_spmv_ms_at_1m_nnz=1.720244,
        gpu_spmv_ms_at_1m_nnz=0.185820,
    ):
        # -------------------------------------------------
        # Basic workload thresholds
        # -------------------------------------------------

        self.min_gpu_nnz = int(min_gpu_nnz)
        self.min_gpu_operations = int(min_gpu_operations)

        # -------------------------------------------------
        # Expected number of iterative SpMV passes
        #
        # PDHG performs approximately:
        #
        #     A @ x
        #     A.T @ y
        #
        # per iteration.
        # -------------------------------------------------

        self.expected_iterations = int(expected_iterations)

        # -------------------------------------------------
        # Initial empirical calibration
        #
        # These values come from the RTX 3050 benchmark:
        #
        # ~1M NNZ:
        # CPU = 1.720244 ms
        # GPU = 0.185820 ms
        #
        # The setup value is deliberately treated as
        # "GPU setup + H2D", not pure H2D.
        # -------------------------------------------------

        self.gpu_setup_ms = float(gpu_setup_ms)

        self.cpu_spmv_ms_at_1m_nnz = float(
            cpu_spmv_ms_at_1m_nnz
        )

        self.gpu_spmv_ms_at_1m_nnz = float(
            gpu_spmv_ms_at_1m_nnz
        )

        # -------------------------------------------------
        # Reference NNZ for the calibration
        # -------------------------------------------------

        self.reference_nnz = 1_000_000

    # =====================================================
    # GPU availability
    # =====================================================

    def gpu_available(self):
        """
        Check whether CuPy and at least one CUDA device
        are available.

        CuPy is imported lazily so CPU-only execution does
        not require CUDA imports at module import time.
        """

        if importlib.util.find_spec("cupy") is None:
            return False

        try:
            import cupy as cp

            return cp.cuda.runtime.getDeviceCount() > 0

        except Exception:
            return False

    # =====================================================
    # Sparse workload estimation
    # =====================================================

    def estimate_operations(self, model):
        """
        Estimate sparse operations performed per iteration.

        One PDHG iteration performs approximately:

            A @ x
            A.T @ y

        Therefore we count two sparse matrix-vector
        operations.
        """

        nnz = int(getattr(model.A, "nnz", 0))

        return 2 * nnz

    # =====================================================
    # CPU cost estimation
    # =====================================================

    def estimate_cpu_spmv_ms(self, model):
        """
        Estimate the time for one CPU SpMV.

        The current model uses linear scaling with NNZ based
        on the measured ~1M-NNZ RTX 3050 benchmark.

        This is an initial empirical model and will be
        replaced/refined as more benchmark points are
        collected.
        """

        nnz = int(getattr(model.A, "nnz", 0))

        if nnz <= 0:
            return 0.0

        scale = nnz / self.reference_nnz

        return self.cpu_spmv_ms_at_1m_nnz * scale

    # =====================================================
    # GPU cost estimation
    # =====================================================

    def estimate_gpu_spmv_ms(self, model):
        """
        Estimate the time for one GPU SpMV.

        Uses the current empirical RTX 3050 calibration.
        """

        nnz = int(getattr(model.A, "nnz", 0))

        if nnz <= 0:
            return 0.0

        scale = nnz / self.reference_nnz

        return self.gpu_spmv_ms_at_1m_nnz * scale

    # =====================================================
    # Total CPU cost
    # =====================================================

    def estimate_cpu_cost_ms(self, model):
        """
        Estimate total CPU sparse-computation cost over
        the expected number of iterations.
        """

        spmv_time = self.estimate_cpu_spmv_ms(model)

        operations_per_iteration = self.estimate_operations(
            model
        )

        # Two SpMVs per iteration.
        iterations = self.expected_iterations

        return (
            spmv_time
            * 2
            * iterations
        )

    # =====================================================
    # Total GPU cost
    # =====================================================

    def estimate_gpu_cost_ms(self, model):
        """
        Estimate total GPU cost.

        Includes:

            GPU setup + initial transfer
            +
            repeated GPU SpMV operations
        """

        spmv_time = self.estimate_gpu_spmv_ms(model)

        iterations = self.expected_iterations

        repeated_compute_ms = (
            spmv_time
            * 2
            * iterations
        )

        return (
            self.gpu_setup_ms
            + repeated_compute_ms
        )

    # =====================================================
    # Select backend
    # =====================================================

    def select(self, model, mode="auto"):
        """
        Select CPU or CUDA backend.

        Parameters
        ----------
        model:
            OptimizationModel instance.

        mode:
            "cpu", "cuda", or "auto".

        Returns
        -------
        str
            "cpu" or "cuda"
        """

        mode = str(mode).lower()

        if mode not in {"cpu", "cuda", "auto"}:
            raise ValueError(
                "mode must be 'cpu', 'cuda', or 'auto'."
            )

        # -------------------------------------------------
        # Explicit CPU mode
        # -------------------------------------------------

        if mode == "cpu":
            return "cpu"

        # -------------------------------------------------
        # Explicit CUDA mode
        # -------------------------------------------------

        if mode == "cuda":
            if not self.gpu_available():
                raise RuntimeError(
                    "CUDA backend requested, but no CUDA "
                    "device is available."
                )

            return "cuda"

        # -------------------------------------------------
        # AUTO mode
        # -------------------------------------------------

        nnz = int(
            getattr(model.A, "nnz", 0)
        )

        operations = self.estimate_operations(model)

        # No GPU available
        if not self.gpu_available():
            return "cpu"

        # Very small workload
        if nnz < self.min_gpu_nnz:
            return "cpu"

        # Too little repeated work to justify GPU
        if operations < self.min_gpu_operations:
            return "cpu"

        # -------------------------------------------------
        # Cost comparison
        # -------------------------------------------------

        cpu_cost = self.estimate_cpu_cost_ms(model)
        gpu_cost = self.estimate_gpu_cost_ms(model)

        if gpu_cost < cpu_cost:
            return "cuda"

        return "cpu"

    # =====================================================
    # Explain decision
    # =====================================================

    def explain(self, model, mode="auto"):
        """
        Return detailed diagnostics explaining the backend
        decision.
        """

        backend = self.select(
            model,
            mode=mode,
        )

        nnz = int(
            getattr(model.A, "nnz", 0)
        )

        operations = self.estimate_operations(
            model
        )

        cpu_spmv_ms = self.estimate_cpu_spmv_ms(
            model
        )

        gpu_spmv_ms = self.estimate_gpu_spmv_ms(
            model
        )

        cpu_cost_ms = self.estimate_cpu_cost_ms(
            model
        )

        gpu_cost_ms = self.estimate_gpu_cost_ms(
            model
        )

        if mode.lower() == "cpu":
            reason = (
                "CPU explicitly requested."
            )

        elif mode.lower() == "cuda":
            reason = (
                "CUDA explicitly requested."
            )

        elif not self.gpu_available():
            reason = (
                "CUDA is unavailable; CPU selected."
            )

        elif nnz < self.min_gpu_nnz:
            reason = (
                "Sparse workload is below the GPU "
                "workload threshold; CPU selected."
            )

        elif operations < self.min_gpu_operations:
            reason = (
                "Expected sparse workload is too small "
                "to justify GPU execution."
            )

        elif gpu_cost_ms < cpu_cost_ms:
            reason = (
                "Estimated GPU execution cost is lower "
                "than CPU execution cost."
            )

        else:
            reason = (
                "Estimated CPU execution cost is lower "
                "than GPU execution cost."
            )

        return {
            "backend": backend,
            "mode": mode.lower(),

            "nnz": nnz,

            "estimated_sparse_operations_per_iteration": (
                int(operations)
            ),

            "expected_iterations": (
                self.expected_iterations
            ),

            "estimated_cpu_spmv_ms": (
                float(cpu_spmv_ms)
            ),

            "estimated_gpu_spmv_ms": (
                float(gpu_spmv_ms)
            ),

            "estimated_cpu_cost_ms": (
                float(cpu_cost_ms)
            ),

            "estimated_gpu_cost_ms": (
                float(gpu_cost_ms)
            ),

            "gpu_setup_ms": (
                float(self.gpu_setup_ms)
            ),

            "gpu_available": (
                self.gpu_available()
            ),

            "gpu_threshold_nnz": (
                self.min_gpu_nnz
            ),

            "reason": reason,
        }