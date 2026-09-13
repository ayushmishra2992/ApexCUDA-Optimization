import time
import numpy as np

from backend.core.optimization_model import OptimizationModel
from backend.core.optimization_model import OptimizationResult
from backend.core.presolver import GenericPresolver
# NOTE: CUDABackend is imported lazily (inside the function that needs it)
# rather than at module load time. cuda_backend.py imports cupy, which is
# not installed on CPU-only machines; importing it eagerly here made the
# entire solver (including pure-CPU solves) fail to import without CuPy.

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
            np.abs(x[integer_mask] - np.round(x[integer_mask]))
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

    def __init__(
        self,
        rho=1.0,
        max_iterations=1000,
        tolerance=1e-6,
        backend="cpu"
    ):
        # Keep rho for API compatibility.
        # The new primal-dual method uses adaptive step sizes.
        self.rho = float(rho)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

        self.backend = str(backend).lower()

        if self.backend not in {"cpu", "cuda", "auto"}:
            raise ValueError(
                "backend must be 'cpu', 'cuda', or 'auto'."
            )

        self.objective_history_ = []
        self.constraint_violation_history_ = []
        self.primal_residual_history_ = []
        self.dual_residual_history_ = []

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

    def _estimate_operator_norm(self, A):
        """
        Estimate ||A||_2 using power iteration.

        This avoids explicitly forming A.T @ A.
        """
        n = A.shape[1]

        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)

        norm_x = np.linalg.norm(x)

        if norm_x == 0:
            return 1.0

        x /= norm_x

        for _ in range(20):
            y = A @ x
            z = A.T @ y

            norm_z = np.linalg.norm(z)

            if norm_z == 0:
                return 1.0

            x = z / norm_z

        Ax = A @ x
        norm_Ax = np.linalg.norm(Ax)

        if norm_Ax == 0:
            return 1.0

        return float(norm_Ax)

    def _solve_cuda(self, model):
        self._validate_model(model)


        A = model.A
        objective = model.objective
        constraint_lower = model.constraint_lower
        constraint_upper = model.constraint_upper
        variable_lower = model.variable_lower
        variable_upper = model.variable_upper

        n = model.n_variables
        m = model.n_constraints

        self.objective_history_ = []
        self.constraint_violation_history_ = []
        self.primal_residual_history_ = []
        self.dual_residual_history_ = []

        start_time = time.perf_counter()

        # Convert maximization into minimization.
        if model.objective_sense == "min":
            c = objective.copy()
        else:
            c = -objective.copy()

        # ------------------------------------------------------------
        # CUDA setup
        # ------------------------------------------------------------

        import cupy as cp
        from backend.backends.cuda_backend import CUDABackend

        cuda = CUDABackend(A)

        c_gpu = cp.asarray(c)
        lower_gpu = cp.asarray(constraint_lower)
        upper_gpu = cp.asarray(constraint_upper)
        variable_lower_gpu = cp.asarray(variable_lower)
        variable_upper_gpu = cp.asarray(variable_upper)

        # ------------------------------------------------------------
        # Initial point
        # ------------------------------------------------------------

        x = cp.zeros(n)

        finite_lower = np.isfinite(variable_lower)
        finite_upper = np.isfinite(variable_upper)

        x[finite_lower] = cp.maximum(
            x[finite_lower],
            variable_lower_gpu[finite_lower]
        )

        x[finite_upper] = cp.minimum(
            x[finite_upper],
            variable_upper_gpu[finite_upper]
        )

        x_bar = x.copy()

        # Dual variable
        y = cp.zeros(m)

        # ------------------------------------------------------------
        # Estimate operator norm on GPU
        # ------------------------------------------------------------

        rng = cp.random.default_rng(42)
        x_norm = rng.standard_normal(n)

        norm_x = cp.linalg.norm(x_norm)

        if norm_x == 0:
            operator_norm = 1.0
        else:
            x_norm /= norm_x

            for _ in range(20):
                y_norm = cuda.matvec(x_norm)
                z_norm = cuda.rmatvec(y_norm)

                norm_z = cp.linalg.norm(z_norm)

                if norm_z == 0:
                    break

                x_norm = z_norm / norm_z

            Ax_norm = cuda.matvec(x_norm)
            operator_norm = float(
                cp.linalg.norm(Ax_norm).get()
            )

            if operator_norm <= 1e-12:
                operator_norm = 1.0

        # ------------------------------------------------------------
        # PDHG step sizes
        # ------------------------------------------------------------

        tau = 0.9 / operator_norm
        sigma = 0.9 / operator_norm
        theta = 1.0

        status = "MAX_ITERATIONS_REACHED"

        # ------------------------------------------------------------
        # PDHG iterations
        # ------------------------------------------------------------

        for iteration in range(1, self.max_iterations + 1):

            # --------------------------------------------------------
            # Dual update
            # --------------------------------------------------------

            y_previous = y.copy()

            v = y + sigma * cuda.matvec(x_bar)

            z = v / sigma

            finite_lower_constraint = cp.isfinite(
                lower_gpu
            )
            finite_upper_constraint = cp.isfinite(
                upper_gpu
            )

            z[finite_lower_constraint] = cp.maximum(
                z[finite_lower_constraint],
                lower_gpu[finite_lower_constraint]
            )

            z[finite_upper_constraint] = cp.minimum(
                z[finite_upper_constraint],
                upper_gpu[finite_upper_constraint]
            )

            y = v - sigma * z

            # --------------------------------------------------------
            # Primal update
            # --------------------------------------------------------

            x_previous = x.copy()

            x = x - tau * (
                c_gpu + cuda.rmatvec(y)
            )

            # Project onto variable bounds.
            finite_lower_variable = cp.isfinite(
                variable_lower_gpu
            )
            finite_upper_variable = cp.isfinite(
                variable_upper_gpu
            )

            x[finite_lower_variable] = cp.maximum(
                x[finite_lower_variable],
                variable_lower_gpu[finite_lower_variable]
            )

            x[finite_upper_variable] = cp.minimum(
                x[finite_upper_variable],
                variable_upper_gpu[finite_upper_variable]
            )

            # --------------------------------------------------------
            # Extrapolation
            # --------------------------------------------------------

            x_bar = x + theta * (x - x_previous)

            # --------------------------------------------------------
            # Diagnostics
            #
            # Only synchronize every 10 iterations to reduce
            # CPU-GPU synchronization overhead.
            # --------------------------------------------------------

            if (
                iteration == 1
                or iteration % 10 == 0
                or iteration == self.max_iterations
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
                    cp.linalg.norm(
                        x - x_previous
                    ).get()
                )

                dual_residual = float(
                    cp.linalg.norm(
                        y - y_previous
                    ).get()
                )

                x_cpu_diagnostic = cp.asnumpy(x)

                original_objective = float(
                    objective @ x_cpu_diagnostic
                )

                self.objective_history_.append(
                    original_objective
                )

                self.constraint_violation_history_.append(
                    current_constraint_violation
                )

                self.primal_residual_history_.append(
                    primal_residual
                )

                self.dual_residual_history_.append(
                    dual_residual
                )

                if (
                    current_constraint_violation <= self.tolerance
                    and current_bound_violation <= self.tolerance
                    and primal_residual <= self.tolerance
                    and dual_residual <= self.tolerance
                ):
                    status = "CONVERGED"
                    break

        # ------------------------------------------------------------
        # Final verification
        # ------------------------------------------------------------

        cuda.synchronize()

        x_cpu = cuda.to_cpu(x)

        final_objective = float(
            objective @ x_cpu
        )

        verification = verify_solution(
            model,
            x_cpu,
            final_objective
        )

        solve_time = time.perf_counter() - start_time

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
                verification["constraint_violation"]
            ),
            bound_violation=float(
                verification["bound_violation"]
            ),
            backend=self.backend,
            problem_type=model.problem_type
        )

    def solve(self, model):

        self._validate_model(model)
                # ------------------------------------------------------------
        # Presolve
        # ------------------------------------------------------------

        presolver = GenericPresolver()
        presolve_result = presolver.presolve(model)

        if presolve_result.status == "INFEASIBLE":
            raise ValueError(
                f"Presolve detected an infeasible model: "
                f"{presolve_result.message}"
            )

        if presolve_result.status == "UNBOUNDED":
            raise ValueError(
                f"Presolve detected an unbounded model: "
                f"{presolve_result.message}"
            )

        if presolve_result.status == "SOLVED":
            solution = presolve_result.postsolve(
                np.array([], dtype=float)
            )

            objective = float(
                model.objective @ solution
            )

            verification = verify_solution(
                model,
                solution,
                objective
            )

            return OptimizationResult(
                status="FEASIBLE" if verification["feasible"] else "MAX_ITERATIONS_REACHED",
                objective=objective,
                solution=solution,
                solve_time=0.0,
                iterations=0,
                constraint_violation=float(
                    verification["constraint_violation"]
                ),
                bound_violation=float(
                    verification["bound_violation"]
                ),
                backend=self.backend,
                problem_type=model.problem_type
            )

        model = presolve_result.model

        # ------------------------------------------------------------
        # Backend selection
        #
        # cpu   -> force CPU
        # cuda  -> force CUDA
        # auto  -> cost-aware automatic selection
        # ------------------------------------------------------------

        requested_backend = self.backend

        selector = BackendSelector(
            expected_iterations=self.max_iterations
        )

        if requested_backend == "auto":
            self.backend = selector.select(
                model,
                mode="auto"
            )

        elif requested_backend == "cuda":
            # Validate that CUDA is actually available.
            self.backend = selector.select(
                model,
                mode="cuda"
            )

        else:
            self.backend = "cpu"

        if self.backend == "cuda":
            return self._solve_cuda(model)

        A = model.A
        objective = model.objective
        constraint_lower = model.constraint_lower
        constraint_upper = model.constraint_upper
        variable_lower = model.variable_lower
        variable_upper = model.variable_upper

        n = model.n_variables
        m = model.n_constraints

        self.objective_history_ = []
        self.constraint_violation_history_ = []
        self.primal_residual_history_ = []
        self.dual_residual_history_ = []

        start_time = time.perf_counter()

        # Convert maximization into minimization.
        if model.objective_sense == "min":
            c = objective.copy()
        else:
            c = -objective.copy()

        # ------------------------------------------------------------
        # Initial point
        # ------------------------------------------------------------

        x = np.zeros(n)

        finite_lower = np.isfinite(variable_lower)
        finite_upper = np.isfinite(variable_upper)

        x[finite_lower] = np.maximum(
            x[finite_lower],
            variable_lower[finite_lower]
        )

        x[finite_upper] = np.minimum(
            x[finite_upper],
            variable_upper[finite_upper]
        )

        x_bar = x.copy()

        # Dual variable for Ax belonging to [lower, upper].
        y = np.zeros(m)

        # ------------------------------------------------------------
        # Step sizes
        #
        # For primal-dual hybrid gradient:
        #
        #     tau * sigma * ||A||^2 < 1
        #
        # ------------------------------------------------------------

        operator_norm = self._estimate_operator_norm(A)

        if operator_norm <= 1e-12:
            operator_norm = 1.0

        tau = 0.9 / operator_norm
        sigma = 0.9 / operator_norm

        theta = 1.0

        status = "MAX_ITERATIONS_REACHED"

        final_constraint_violation = np.inf
        final_bound_violation = np.inf

        previous_x = x.copy()
        previous_y = y.copy()

        # ------------------------------------------------------------
        # PDHG iterations
        # ------------------------------------------------------------

        for iteration in range(1, self.max_iterations + 1):

            # --------------------------------------------------------
            # Dual update
            #
            # We need:
            #
            #     y = prox_{sigma f*}(y + sigma A x_bar)
            #
            # where f is the indicator function of:
            #
            #     lower <= z <= upper
            #
            # The conjugate proximal operator can be written through
            # Moreau's identity:
            #
            #     prox_{sigma f*}(v)
            #       = v - sigma * projection(v/sigma)
            #
            # --------------------------------------------------------

            y_previous = y.copy()

            v = y + sigma * (A @ x_bar)

            z = np.minimum(
                np.maximum(v / sigma, constraint_lower),
                constraint_upper
            )

            # Handle infinite bounds correctly.
            finite_lower_constraint = np.isfinite(constraint_lower)
            finite_upper_constraint = np.isfinite(constraint_upper)

            z = v / sigma

            z[finite_lower_constraint] = np.maximum(
                z[finite_lower_constraint],
                constraint_lower[finite_lower_constraint]
            )

            z[finite_upper_constraint] = np.minimum(
                z[finite_upper_constraint],
                constraint_upper[finite_upper_constraint]
            )

            y = v - sigma * z

            # --------------------------------------------------------
            # Primal update
            #
            # x = projection_bounds(
            #       x - tau * (c + A.T y)
            #     )
            # --------------------------------------------------------

            x_previous = x.copy()

            x = x - tau * (c + A.T @ y)

            # Project onto variable bounds.
            x = np.maximum(x, variable_lower)
            x = np.minimum(x, variable_upper)

            # --------------------------------------------------------
            # Extrapolation
            # --------------------------------------------------------

            x_bar = x + theta * (x - x_previous)

            # --------------------------------------------------------
            # Diagnostics
            # --------------------------------------------------------

            Ax = A @ x

            current_constraint_violation = constraint_violation(
                Ax,
                constraint_lower,
                constraint_upper
            )

            current_bound_violation = bound_violation(
                x,
                variable_lower,
                variable_upper
            )

            primal_residual = np.linalg.norm(
                x - x_previous
            )

            dual_residual = np.linalg.norm(
                y - y_previous
            )

            original_objective = objective @ x

            self.objective_history_.append(
                float(original_objective)
            )

            self.constraint_violation_history_.append(
                float(current_constraint_violation)
            )

            self.primal_residual_history_.append(
                float(primal_residual)
            )

            self.dual_residual_history_.append(
                float(dual_residual)
            )

            # --------------------------------------------------------
            # Convergence
            # --------------------------------------------------------

            if (
                current_constraint_violation <= self.tolerance
                and current_bound_violation <= self.tolerance
                and primal_residual <= self.tolerance
                and dual_residual <= self.tolerance
            ):
                status = "CONVERGED"

                final_constraint_violation = (
                    current_constraint_violation
                )

                final_bound_violation = (
                    current_bound_violation
                )

                break

        else:
            final_constraint_violation = (
                self.constraint_violation_history_[-1]
            )

            final_bound_violation = bound_violation(
                x,
                variable_lower,
                variable_upper
            )

        # ------------------------------------------------------------
        # Final verification
        # ------------------------------------------------------------

        final_objective = objective @ x

        verification = verify_solution(
            model,
            x,
            final_objective
        )

        solve_time = time.perf_counter() - start_time

        if status == "CONVERGED":

            if verification["feasible"]:
                status = "FEASIBLE"
            else:
                status = "MAX_ITERATIONS_REACHED"

        return OptimizationResult(
            status=status,
            objective=float(final_objective),
            solution=x,
            solve_time=float(solve_time),
            iterations=iteration,
            constraint_violation=float(
                verification["constraint_violation"]
            ),
            bound_violation=float(
                verification["bound_violation"]
            ),
            backend=self.backend,
            problem_type=model.problem_type
        )