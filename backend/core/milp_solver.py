import time
import numpy as np

from backend.core.optimization_model import OptimizationResult
from backend.core.solver import verify_solution


class MILPSolver:
    """
    Generic MILP solver using branch-and-bound.

    Each node solves an LP relaxation using the existing ADMM solver.
    Fractional integer variables are branched into two child nodes.
    """

    def __init__(
        self,
        rho=1.0,
        max_lp_iterations=100,
        max_nodes=100,
        backend="cpu"
    ):
        self.rho = rho
        self.max_lp_iterations = max_lp_iterations
        self.max_nodes = max_nodes
        self.backend = backend

    def solve(self, model):
        if model.problem_type != "MILP":
            raise ValueError(
                f"MILPSolver requires a MILP model. "
                f"Received: {model.problem_type}"
            )

        start_time = time.time()

        from backend.core.solver import ADMMSolver

        lp_solver = ADMMSolver(
            rho=self.rho,
            max_iterations=self.max_lp_iterations,
            backend=self.backend
        )

        integer_indices = np.where(
            model.variable_integrality != 0
        )[0]

        # Each node is represented by variable lower/upper bounds.
        root_lower = model.variable_lower.copy()
        root_upper = model.variable_upper.copy()

        nodes = [
            (root_lower, root_upper)
        ]

        incumbent_solution = None
        incumbent_objective = (
            np.inf if model.objective_sense == "min"
            else -np.inf
        )

        processed_nodes = 0

        while nodes and processed_nodes < self.max_nodes:

            node_lower, node_upper = nodes.pop()
            processed_nodes += 1

            # Ignore invalid nodes.
            if np.any(node_lower > node_upper):
                continue

            relaxation_model = self._make_relaxation(
                model,
                node_lower,
                node_upper
            )

            relaxation_result = lp_solver.solve(
                relaxation_model
            )

            if relaxation_result.status != "FEASIBLE":
                continue

            x = relaxation_result.solution
            relaxation_objective = model.objective @ x

            # Bound pruning.
            if incumbent_solution is not None:

                if model.objective_sense == "min":
                    if relaxation_objective >= incumbent_objective - 1e-9:
                        continue

                else:
                    if relaxation_objective <= incumbent_objective + 1e-9:
                        continue

            # Find a fractional integer variable.
            fractional_index = self._find_fractional_variable(
                x,
                integer_indices
            )

            # Integer solution found.
            if fractional_index is None:

                verification = verify_solution(
                    model,
                    x,
                    relaxation_objective
                )

                if not verification["feasible"]:
                    continue

                if (
                    incumbent_solution is None
                    or self._is_better(
                        verification["objective"],
                        incumbent_objective,
                        model.objective_sense
                    )
                ):
                    incumbent_solution = x.copy()
                    incumbent_objective = verification["objective"]

                continue

            # Branch on the fractional variable.
            value = x[fractional_index]

            floor_value = np.floor(value)
            ceil_value = np.ceil(value)

            # Left branch: x_i <= floor(value)
            left_lower = node_lower.copy()
            left_upper = node_upper.copy()

            left_upper[fractional_index] = min(
                left_upper[fractional_index],
                floor_value
            )

            if left_lower[fractional_index] <= left_upper[fractional_index]:
                nodes.append(
                    (left_lower, left_upper)
                )

            # Right branch: x_i >= ceil(value)
            right_lower = node_lower.copy()
            right_upper = node_upper.copy()

            right_lower[fractional_index] = max(
                right_lower[fractional_index],
                ceil_value
            )

            if right_lower[fractional_index] <= right_upper[fractional_index]:
                nodes.append(
                    (right_lower, right_upper)
                )

        solve_time = time.time() - start_time

        # No integer feasible solution was found.
        if incumbent_solution is None:
            return OptimizationResult(
                status="NO_INTEGER_SOLUTION",
                objective=np.inf,
                solution=np.zeros(model.n_variables),
                solve_time=solve_time,
                iterations=processed_nodes,
                constraint_violation=np.inf,
                bound_violation=np.inf,
                backend=self.backend,
                problem_type="MILP"
            )

        verification = verify_solution(
            model,
            incumbent_solution,
            incumbent_objective
        )

        status = "FEASIBLE"

        if nodes:
            status = "NODE_LIMIT"

        return OptimizationResult(
            status=status,
            objective=verification["objective"],
            solution=incumbent_solution,
            solve_time=solve_time,
            iterations=processed_nodes,
            constraint_violation=verification["constraint_violation"],
            bound_violation=verification["bound_violation"],
            backend=self.backend,
            problem_type="MILP"
        )

    def _make_relaxation(
        self,
        model,
        variable_lower,
        variable_upper
    ):
        """
        Create an LP relaxation for a branch-and-bound node.
        """

        from backend.core.optimization_model import OptimizationModel

        return OptimizationModel(
            A=model.A,
            constraint_lower=model.constraint_lower,
            constraint_upper=model.constraint_upper,
            objective=model.objective,
            variable_lower=variable_lower,
            variable_upper=variable_upper,
            variable_integrality=np.zeros(
                model.n_variables,
                dtype=int
            ),
            objective_sense=model.objective_sense
        )

    def _find_fractional_variable(
        self,
        solution,
        integer_indices,
        tolerance=1e-6
    ):
        """
        Find the first integer variable whose value
        is fractional.
        """

        for index in integer_indices:

            value = solution[index]

            if abs(value - round(value)) > tolerance:
                return int(index)

        return None

    def _is_better(
        self,
        objective,
        incumbent,
        sense
    ):
        if sense == "min":
            return objective < incumbent - 1e-9

        return objective > incumbent + 1e-9