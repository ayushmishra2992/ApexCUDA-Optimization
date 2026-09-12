import numpy as np


class BackendSelector:
    """
    Chooses CPU or CUDA execution based on optimization workload size.

    The selector is generic and independent of the airline application.
    """

    def __init__(
        self,
        min_gpu_nnz=1_000_000,
        min_gpu_operations=20
    ):
        self.min_gpu_nnz = min_gpu_nnz
        self.min_gpu_operations = min_gpu_operations

    def estimate_operations(self, model):
        """
        Estimate the number of sparse matrix operations likely to benefit
        from GPU execution.
        """
        # PDHG repeatedly performs A*x and A.T*y.
        return 2 * getattr(model.A, "nnz", 0)

    def select(self, model):
        """
        Return the recommended backend: 'cpu' or 'cuda'.
        """

        nnz = getattr(model.A, "nnz", 0)

        if nnz < self.min_gpu_nnz:
            return "cpu"

        return "cuda"

    def explain(self, model):
        """
        Return diagnostic information explaining the backend decision.
        """

        nnz = getattr(model.A, "nnz", 0)
        backend = self.select(model)

        return {
            "backend": backend,
            "nnz": int(nnz),
            "estimated_sparse_operations": int(self.estimate_operations(model)),
            "gpu_threshold_nnz": self.min_gpu_nnz,
            "reason": (
                "Large sparse model; GPU execution selected."
                if backend == "cuda"
                else "Small sparse model; CPU execution selected to avoid GPU overhead."
            )
        }