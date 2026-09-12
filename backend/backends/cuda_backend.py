import cupy as cp
import cupyx.scipy.sparse as cusparse
import numpy as np


class CUDABackend:
    """
    CUDA numerical backend for sparse matrix-vector operations.

    The optimization algorithm remains in the generic solver layer.
    This backend provides GPU-resident sparse linear algebra.
    """

    def __init__(self, A):
        if not hasattr(A, "tocsr"):
            raise TypeError("A must be a SciPy sparse matrix.")

        self.A = cusparse.csr_matrix(A)

    def matvec(self, x):
        return self.A @ cp.asarray(x)

    def rmatvec(self, x):
        return self.A.T @ cp.asarray(x)

    def to_cpu(self, x):
        return cp.asnumpy(x)

    def synchronize(self):
        cp.cuda.Stream.null.synchronize()