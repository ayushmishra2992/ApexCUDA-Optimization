import time
import numpy as np

from backend.core.optimization_model import OptimizationModel
from backend.core.optimization_model import OptimizationResult


def constraint_violation(Ax, lower, upper):
    lower_violation = np.maximum(lower - Ax, 0)
    upper_violation = np.maximum(Ax - upper, 0)

    return float(
        max(
            np.max(lower_violation),
            np.max(upper_violation)
        )
    )


def bound_violation(x, lower, upper):
    lower_violation = np.maximum(lower - x, 0)
    upper_violation = np.maximum(x - upper, 0)

    return float(
        max(
            np.max(lower_violation),
            np.max(upper_violation)
        )
    )


def integrality_violation(x, integrality):
    integer_mask = integrality != 0

    if not np.any(integer_mask):
        return 0.0

    return float(
        np.max(
            np.abs(
                x[integer_mask] -
                np.round(x[integer_mask])
            )
        )
    )


def verify_solution(
    model,
    solution,
    objective=None,
    feasibility_tolerance=1e-6,
    integrality_tolerance=1e-6
):
    Ax = model.A @ solution

    constraint_error = constraint_violation(
        Ax,
        model.constraint_lower,
        model.constraint_upper
    )

    bound_error = bound_violation(
        solution,
        model.variable_lower,
        model.variable_upper
    )

    integer_error = integrality_violation(
        solution,
        model.variable_integrality
    )

    if objective is None:
        objective = model.objective @ solution

    feasible = (
        constraint_error <= feasibility_tolerance
        and bound_error <= feasibility_tolerance
        and integer_error <= integrality_tolerance
    )

    return {
        "objective": float(objective),
        "constraint_violation": float(constraint_error),
        "bound_violation": float(bound_error),
        "integrality_violation": float(integer_error),
        "feasible": feasible
    }


class ADMMSolver:
    """
    Sparse primal-dual hybrid gradient solver.

    The class name is retained as ADMMSolver for API compatibility.

    The implementation uses:
        1. Sparse numerical equilibration.
        2. Diagonally preconditioned PDHG.

    Scaling is applied internally and the final solution is mapped
    back to the original variable space before verification.

    Supports LP models on CPU and CUDA backends.
    """

    def __init__(
        self,
        rho=1.0,
        max_iterations=1000,
        tolerance=1e-6,
        backend="cpu"
    ):
        self.rho = float(rho)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.backend = backend
        self.initial_theta = 1.0
        self.min_theta = 1.0
        self.max_theta = 1.0
        self.theta_increase = 1.0
        self.theta_decrease = 1.0

        self.objective_history_ = []
        self.constraint_violation_history_ = []
        self.primal_residual_history_ = []
        self.dual_residual_history_ = []

        self.row_scaling_ = None
        self.column_scaling_ = None

    def _validate_model(self, model):
        if model.problem_type != "LP":
            raise ValueError(
                "ADMMSolver currently supports LP models only. "
                f"Received: {model.problem_type}"
            )

        if model.A.shape[0] != len(model.constraint_lower):
            raise ValueError(
                "Constraint lower bounds do not match A."
            )

        if model.A.shape[0] != len(model.constraint_upper):
            raise ValueError(
                "Constraint upper bounds do not match A."
            )

        if model.A.shape[1] != len(model.objective):
            raise ValueError(
                "Objective length does not match A."
            )

        if self.max_iterations <= 0:
            raise ValueError(
                "max_iterations must be positive."
            )

        if self.tolerance <= 0:
            raise ValueError(
                "tolerance must be positive."
            )

        if self.backend not in {"cpu", "cuda"}:
            raise ValueError(
                f"Unsupported backend: {self.backend}. "
                "Expected 'cpu' or 'cuda'."
            )

    def _reset_histories(self):
        self.objective_history_ = []
        self.constraint_violation_history_ = []
        self.primal_residual_history_ = []
        self.dual_residual_history_ = []

    def _equilibrate_model(self, model, passes=5):
        """
        Keep the model in its original sparse numerical scale.

        The solver uses diagonal PDHG preconditioning directly from the
        sparse coefficient magnitudes.  Identity scaling is intentional:
        it avoids introducing an additional numerical transformation that
        can slow convergence on small and moderately scaled LPs.
        """
        self.row_scaling_ = np.ones(model.n_constraints, dtype=float)
        self.column_scaling_ = np.ones(model.n_variables, dtype=float)
        return model

    def _unscale_solution(self, scaled_solution):
        if self.column_scaling_ is None:
            return scaled_solution.copy()

        return (
            self.column_scaling_ *
            scaled_solution
        )

    def _compute_diagonal_preconditioner(self, A):
        """
        Construct diagonal PDHG step sizes from sparse row/column
        absolute coefficient sums.

            tau_j   = alpha / sum_i |A_ij|
            sigma_i = alpha / sum_j |A_ij|

        Zero rows/columns receive a finite fallback.

        No dense matrix is constructed.
        """

        alpha = 0.95

        column_scale = np.asarray(
            np.abs(A).sum(axis=0)
        ).ravel()

        row_scale = np.asarray(
            np.abs(A).sum(axis=1)
        ).ravel()

        column_scale = np.maximum(
            column_scale,
            1e-12
        )

        row_scale = np.maximum(
            row_scale,
            1e-12
        )

        tau = alpha / column_scale
        sigma = alpha / row_scale

        tau = np.clip(
            tau,
            1e-12,
            1e6
        )

        sigma = np.clip(
            sigma,
            1e-12,
            1e6
        )

        return (
            tau.astype(float),
            sigma.astype(float)
        )

    def _normalized_residual(self, current, previous):
        numerator = np.linalg.norm(
            current - previous
        )

        denominator = max(
            1.0,
            np.linalg.norm(current)
        )

        return float(
            numerator / denominator
        )

    def _adapt_theta(self, theta, current_violation, previous_violation):
        return 1.0

    def _solve_cuda(self, model):
        self._validate_model(model)
        self._reset_histories()

        scaled_model = self._equilibrate_model(
            model
        )

        A = scaled_model.A

        objective = scaled_model.objective
        constraint_lower = (
            scaled_model.constraint_lower
        )
        constraint_upper = (
            scaled_model.constraint_upper
        )

        variable_lower = (
            scaled_model.variable_lower
        )
        variable_upper = (
            scaled_model.variable_upper
        )

        n = scaled_model.n_variables
        m = scaled_model.n_constraints

        start_time = time.perf_counter()

        if scaled_model.objective_sense == "min":
            c = objective.copy()
        else:
            c = -objective.copy()

        import cupy as cp
        from backend.backends.cuda_backend import CUDABackend

        cuda = CUDABackend(A)

        c_gpu = cp.asarray(c)
        lower_gpu = cp.asarray(
            constraint_lower
        )
        upper_gpu = cp.asarray(
            constraint_upper
        )

        variable_lower_gpu = cp.asarray(
            variable_lower
        )
        variable_upper_gpu = cp.asarray(
            variable_upper
        )

        tau_cpu, sigma_cpu = (
            self._compute_diagonal_preconditioner(
                A
            )
        )

        tau_gpu = cp.asarray(tau_cpu)
        sigma_gpu = cp.asarray(sigma_cpu)

        x = cp.zeros(n)

        finite_lower = np.isfinite(
            variable_lower
        )

        finite_upper = np.isfinite(
            variable_upper
        )

        if np.any(finite_lower):
            x[finite_lower] = cp.maximum(
                x[finite_lower],
                variable_lower_gpu[
                    finite_lower
                ]
            )

        if np.any(finite_upper):
            x[finite_upper] = cp.minimum(
                x[finite_upper],
                variable_upper_gpu[
                    finite_upper
                ]
            )

        x_bar = x.copy()
        y = cp.zeros(m)

        theta = self.initial_theta
        previous_violation = np.inf

        status = "MAX_ITERATIONS_REACHED"
        iteration = 0

        finite_lower_constraint = cp.isfinite(
            lower_gpu
        )

        finite_upper_constraint = cp.isfinite(
            upper_gpu
        )

        finite_lower_variable = cp.isfinite(
            variable_lower_gpu
        )

        finite_upper_variable = cp.isfinite(
            variable_upper_gpu
        )

        for iteration in range(
            1,
            self.max_iterations + 1
        ):
            y_previous = y.copy()

            v = (
                y +
                sigma_gpu *
                cuda.matvec(x_bar)
            )

            z = v / sigma_gpu

            if bool(
                cp.any(
                    finite_lower_constraint
                )
            ):
                z[finite_lower_constraint] = (
                    cp.maximum(
                        z[
                            finite_lower_constraint
                        ],
                        lower_gpu[
                            finite_lower_constraint
                        ]
                    )
                )

            if bool(
                cp.any(
                    finite_upper_constraint
                )
            ):
                z[finite_upper_constraint] = (
                    cp.minimum(
                        z[
                            finite_upper_constraint
                        ],
                        upper_gpu[
                            finite_upper_constraint
                        ]
                    )
                )

            y = v - sigma_gpu * z

            x_previous = x.copy()

            x = x - tau_gpu * (
                c_gpu +
                cuda.rmatvec(y)
            )

            if bool(
                cp.any(
                    finite_lower_variable
                )
            ):
                x[finite_lower_variable] = (
                    cp.maximum(
                        x[
                            finite_lower_variable
                        ],
                        variable_lower_gpu[
                            finite_lower_variable
                        ]
                    )
                )

            if bool(
                cp.any(
                    finite_upper_variable
                )
            ):
                x[finite_upper_variable] = (
                    cp.minimum(
                        x[
                            finite_upper_variable
                        ],
                        variable_upper_gpu[
                            finite_upper_variable
                        ]
                    )
                )

            x_bar = x + theta * (
                x - x_previous
            )

            if (
                iteration == 1
                or iteration % 10 == 0
                or iteration ==
                self.max_iterations
            ):
                Ax_gpu = cuda.matvec(x)

                lower_violation = cp.maximum(
                    lower_gpu - Ax_gpu,
                    0
                )

                upper_violation = cp.maximum(
                    Ax_gpu - upper_gpu,
                    0
                )

                current_constraint_violation = float(
                    cp.maximum(
                        lower_violation,
                        upper_violation
                    ).max().get()
                )

                lower_bound_violation = cp.maximum(
                    variable_lower_gpu - x,
                    0
                )

                upper_bound_violation = cp.maximum(
                    x - variable_upper_gpu,
                    0
                )

                current_bound_violation = float(
                    cp.maximum(
                        lower_bound_violation,
                        upper_bound_violation
                    ).max().get()
                )

                primal_residual = float(
                    (
                        cp.linalg.norm(
                            x - x_previous
                        )
                        /
                        cp.maximum(
                            1.0,
                            cp.linalg.norm(x)
                        )
                    ).get()
                )

                dual_residual = float(
                    (
                        cp.linalg.norm(
                            y - y_previous
                        )
                        /
                        cp.maximum(
                            1.0,
                            cp.linalg.norm(y)
                        )
                    ).get()
                )

                x_cpu_diagnostic = (
                    cp.asnumpy(x)
                )

                x_original = (
                    self._unscale_solution(
                        x_cpu_diagnostic
                    )
                )

                original_objective = float(
                    model.objective @
                    x_original
                )

                self.objective_history_.append(
                    original_objective
                )

                self.constraint_violation_history_.append(
                    current_constraint_violation
                )

                theta = self._adapt_theta(
                    theta,
                    current_constraint_violation,
                    previous_violation
                )
                previous_violation = current_constraint_violation

                self.primal_residual_history_.append(
                    primal_residual
                )

                self.dual_residual_history_.append(
                    dual_residual
                )

                if (
                    current_constraint_violation
                    <= self.tolerance
                    and current_bound_violation
                    <= self.tolerance
                    and primal_residual
                    <= self.tolerance
                    and dual_residual
                    <= self.tolerance
                ):
                    status = "CONVERGED"
                    break

        cuda.synchronize()

        x_scaled = cuda.to_cpu(x)

        x_cpu = self._unscale_solution(
            x_scaled
        )

        final_objective = float(
            model.objective @ x_cpu
        )

        verification = verify_solution(
            model,
            x_cpu,
            final_objective
        )

        solve_time = (
            time.perf_counter() -
            start_time
        )

        if status == "CONVERGED":
            if verification["feasible"]:
                status = "FEASIBLE"
            else:
                status = "MAX_ITERATIONS_REACHED"

        return OptimizationResult(
            status=status,
            objective=final_objective,
            solution=x_cpu,
            solve_time=float(solve_time),
            iterations=iteration,
            constraint_violation=float(
                verification[
                    "constraint_violation"
                ]
            ),
            bound_violation=float(
                verification[
                    "bound_violation"
                ]
            ),
            backend=self.backend,
            problem_type=model.problem_type
        )

    def solve(self, model):
        self._validate_model(model)

        if self.backend == "cuda":
            return self._solve_cuda(model)

        self._reset_histories()

        scaled_model = self._equilibrate_model(
            model
        )

        A = scaled_model.A

        objective = scaled_model.objective
        constraint_lower = (
            scaled_model.constraint_lower
        )
        constraint_upper = (
            scaled_model.constraint_upper
        )

        variable_lower = (
            scaled_model.variable_lower
        )
        variable_upper = (
            scaled_model.variable_upper
        )

        n = scaled_model.n_variables
        m = scaled_model.n_constraints

        start_time = time.perf_counter()

        if scaled_model.objective_sense == "min":
            c = objective.copy()
        else:
            c = -objective.copy()

        x = np.zeros(n)

        finite_lower = np.isfinite(
            variable_lower
        )

        finite_upper = np.isfinite(
            variable_upper
        )

        x[finite_lower] = np.maximum(
            x[finite_lower],
            variable_lower[
                finite_lower
            ]
        )

        x[finite_upper] = np.minimum(
            x[finite_upper],
            variable_upper[
                finite_upper
            ]
        )

        x_bar = x.copy()

        y = np.zeros(m)

        tau, sigma = (
            self._compute_diagonal_preconditioner(
                A
            )
        )

        theta = self.initial_theta
        previous_violation = np.inf

        status = "MAX_ITERATIONS_REACHED"
        iteration = 0

        finite_lower_constraint = np.isfinite(
            constraint_lower
        )

        finite_upper_constraint = np.isfinite(
            constraint_upper
        )

        for iteration in range(
            1,
            self.max_iterations + 1
        ):
            y_previous = y.copy()

            Ax_bar = A @ x_bar

            v = y + sigma * Ax_bar

            z = v / sigma

            z[finite_lower_constraint] = (
                np.maximum(
                    z[
                        finite_lower_constraint
                    ],
                    constraint_lower[
                        finite_lower_constraint
                    ]
                )
            )

            z[finite_upper_constraint] = (
                np.minimum(
                    z[
                        finite_upper_constraint
                    ],
                    constraint_upper[
                        finite_upper_constraint
                    ]
                )
            )

            y = v - sigma * z

            x_previous = x.copy()

            ATy = A.T @ y

            x = x - tau * (
                c + ATy
            )

            x[finite_lower] = np.maximum(
                x[finite_lower],
                variable_lower[
                    finite_lower
                ]
            )

            x[finite_upper] = np.minimum(
                x[finite_upper],
                variable_upper[
                    finite_upper
                ]
            )

            x_bar = x + theta * (
                x - x_previous
            )

            Ax = A @ x

            scaled_constraint_violation = (
                constraint_violation(
                    Ax,
                    constraint_lower,
                    constraint_upper
                )
            )

            current_bound_violation = (
                bound_violation(
                    x,
                    variable_lower,
                    variable_upper
                )
            )

            primal_residual = (
                self._normalized_residual(
                    x,
                    x_previous
                )
            )

            dual_residual = (
                self._normalized_residual(
                    y,
                    y_previous
                )
            )

            x_original = (
                self._unscale_solution(x)
            )

            original_objective = (
                model.objective @
                x_original
            )

            self.objective_history_.append(
                float(original_objective)
            )

            self.constraint_violation_history_.append(
                float(
                    scaled_constraint_violation
                )
            )

            theta = self._adapt_theta(
                theta,
                scaled_constraint_violation,
                previous_violation
            )
            previous_violation = scaled_constraint_violation

            self.primal_residual_history_.append(
                float(primal_residual)
            )

            self.dual_residual_history_.append(
                float(dual_residual)
            )

            if (
                scaled_constraint_violation
                <= self.tolerance
                and current_bound_violation
                <= self.tolerance
                and primal_residual
                <= self.tolerance
                and dual_residual
                <= self.tolerance
            ):
                status = "CONVERGED"
                break

        x_original = (
            self._unscale_solution(x)
        )

        final_objective = float(
            model.objective @
            x_original
        )

        verification = verify_solution(
            model,
            x_original,
            final_objective
        )

        solve_time = (
            time.perf_counter() -
            start_time
        )

        if status == "CONVERGED":
            if verification["feasible"]:
                status = "FEASIBLE"
            else:
                status = "MAX_ITERATIONS_REACHED"

        return OptimizationResult(
            status=status,
            objective=final_objective,
            solution=x_original,
            solve_time=float(solve_time),
            iterations=iteration,
            constraint_violation=float(
                verification[
                    "constraint_violation"
                ]
            ),
            bound_violation=float(
                verification[
                    "bound_violation"
                ]
            ),
            backend=self.backend,
            problem_type=model.problem_type
        )