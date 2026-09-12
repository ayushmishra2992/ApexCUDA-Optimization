import time
from dataclasses import dataclass

import numpy as np

from backend.core.optimization_model import OptimizationResult
from backend.core.solver import verify_solution


@dataclass
class BBNode:
    lower: np.ndarray
    upper: np.ndarray
    depth: int = 0
    bound: float = None
    solution: np.ndarray = None


class MILPSolver:
    """
    Generic MILP solver using branch-and-bound.

    Each node solves an LP relaxation using the existing LP solver.
    Fractional integer variables are branched into two child nodes.
    """

    def __init__(
        self,
        rho=1.0,
        max_lp_iterations=100,
        max_nodes=100,
        backend="cpu",
        time_limit=None,
        mip_gap=0.0
    ):
        self.rho = rho
        self.max_lp_iterations = max_lp_iterations
        self.max_nodes = max_nodes
        self.backend = backend
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.metrics = {}

    def solve(self, model):
        if model.problem_type != "MILP":
            raise ValueError(
                f"MILPSolver requires a MILP model. "
                f"Received: {model.problem_type}"
            )

        if self.mip_gap < 0:
            raise ValueError("mip_gap must be non-negative.")

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

        root_lower = model.variable_lower.copy()
        root_upper = model.variable_upper.copy()

        processed_nodes = 0
        pruned_nodes = 0
        integer_solutions = 0
        self._lp_relaxations = 0

        root_node = BBNode(
            lower=root_lower,
            upper=root_upper,
            depth=0
        )

        root_x = self._evaluate_node(
            model,
            root_node,
            lp_solver
        )

        if root_x is None:
            solve_time = time.time() - start_time

            self.metrics = {
                "nodes_processed": 0,
                "nodes_remaining": 0,
                "nodes_pruned": 0,
                "integer_solutions": 0,
                "lp_relaxations": self._lp_relaxations,
                "best_bound": None,
                "absolute_gap": None,
                "relative_gap": None,
            }

            return OptimizationResult(
                status="NO_INTEGER_SOLUTION",
                objective=np.inf,
                solution=np.zeros(model.n_variables),
                solve_time=solve_time,
                iterations=0,
                constraint_violation=np.inf,
                bound_violation=np.inf,
                backend=self.backend,
                problem_type="MILP"
            )

        nodes = [root_node]

        incumbent_solution = None
        incumbent_objective = (
            np.inf if model.objective_sense == "min"
            else -np.inf
        )

        termination_reason = None

        while nodes and processed_nodes < self.max_nodes:

            if (
                self.time_limit is not None
                and time.time() - start_time >= self.time_limit
            ):
                termination_reason = "TIME_LIMIT"
                break

            if model.objective_sense == "min":
                best_index = min(
                    range(len(nodes)),
                    key=lambda i: nodes[i].bound
                )
            else:
                best_index = max(
                    range(len(nodes)),
                    key=lambda i: nodes[i].bound
                )

            node = nodes.pop(best_index)

            processed_nodes += 1

            node_lower = node.lower
            node_upper = node.upper

            if np.any(node_lower > node_upper):
                pruned_nodes += 1
                continue

            x = node.solution

            if x is None:
                x = self._evaluate_node(
                    model,
                    node,
                    lp_solver
                )

                if x is None:
                    pruned_nodes += 1
                    continue

            relaxation_objective = float(
                model.objective @ x
            )

            # Bound pruning.
            if incumbent_solution is not None:

                if model.objective_sense == "min":
                    if relaxation_objective >= incumbent_objective - 1e-9:
                        pruned_nodes += 1
                        continue

                else:
                    if relaxation_objective <= incumbent_objective + 1e-9:
                        pruned_nodes += 1
                        continue

            # Find the most fractional integer variable.
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
                    pruned_nodes += 1
                    continue

                integer_solutions += 1

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

            # Branch on the selected fractional variable.
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

                left_node = BBNode(
                    lower=left_lower,
                    upper=left_upper,
                    depth=node.depth + 1
                )

                left_x = self._evaluate_node(
                    model,
                    left_node,
                    lp_solver
                )

                if left_x is not None:
                    nodes.append(left_node)

            # Right branch: x_i >= ceil(value)
            right_lower = node_lower.copy()
            right_upper = node_upper.copy()

            right_lower[fractional_index] = max(
                right_lower[fractional_index],
                ceil_value
            )

            if right_lower[fractional_index] <= right_upper[fractional_index]:

                right_node = BBNode(
                    lower=right_lower,
                    upper=right_upper,
                    depth=node.depth + 1
                )

                right_x = self._evaluate_node(
                    model,
                    right_node,
                    lp_solver
                )

                if right_x is not None:
                    nodes.append(right_node)

            # Check the current MIP gap after processing a node.
            if incumbent_solution is not None and nodes:

                best_bound = self._best_bound(
                    nodes,
                    model.objective_sense
                )

                absolute_gap, relative_gap = self._calculate_gap(
                    incumbent_objective,
                    best_bound,
                    model.objective_sense
                )

                if relative_gap <= self.mip_gap:
                    termination_reason = "MIP_GAP"
                    break

        solve_time = time.time() - start_time

        # No integer feasible solution was found.
        if incumbent_solution is None:

            if termination_reason == "TIME_LIMIT":
                status = "TIME_LIMIT"
            elif processed_nodes >= self.max_nodes and nodes:
                status = "NODE_LIMIT"
            else:
                status = "NO_INTEGER_SOLUTION"

            self.metrics = {
                "nodes_processed": processed_nodes,
                "nodes_remaining": len(nodes),
                "nodes_pruned": pruned_nodes,
                "integer_solutions": integer_solutions,
                "lp_relaxations": self._lp_relaxations,
                "best_bound": (
                    self._best_bound(
                        nodes,
                        model.objective_sense
                    )
                    if nodes
                    else None
                ),
                "absolute_gap": None,
                "relative_gap": None,
            }

            return OptimizationResult(
                status=status,
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

        if nodes:
            best_bound = self._best_bound(
                nodes,
                model.objective_sense
            )

            absolute_gap, relative_gap = self._calculate_gap(
                incumbent_objective,
                best_bound,
                model.objective_sense
            )
        else:
            best_bound = incumbent_objective
            absolute_gap = 0.0
            relative_gap = 0.0

        if not nodes:
            status = "OPTIMAL"

        elif termination_reason == "TIME_LIMIT":
            status = "TIME_LIMIT"

        elif termination_reason == "MIP_GAP":
            status = "MIP_GAP"

        elif processed_nodes >= self.max_nodes:
            status = "NODE_LIMIT"

        else:
            status = "NODE_LIMIT"

        self.metrics = {
            "nodes_processed": processed_nodes,
            "nodes_remaining": len(nodes),
            "nodes_pruned": pruned_nodes,
            "integer_solutions": integer_solutions,
            "lp_relaxations": self._lp_relaxations,
            "best_bound": float(best_bound),
            "absolute_gap": float(absolute_gap),
            "relative_gap": float(relative_gap),
        }

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

    def _evaluate_node(
        self,
        model,
        node,
        lp_solver
    ):
        if np.any(node.lower > node.upper):
            return None

        relaxation_model = self._make_relaxation(
            model,
            node.lower,
            node.upper
        )

        relaxation_result = lp_solver.solve(
            relaxation_model
        )

        self._lp_relaxations += 1

        if relaxation_result.status not in {"FEASIBLE", "OPTIMAL"}:
            return None

        x = relaxation_result.solution

        node.solution = x.copy()

        node.bound = float(
            model.objective @ x
        )

        return x

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
        Find the most fractional integer variable.
        """

        best_index = None
        best_fractionality = -1.0

        for index in integer_indices:

            value = solution[index]

            fractionality = abs(
                value - np.round(value)
            )

            if fractionality > tolerance:
                score = min(
                    fractionality,
                    1.0 - fractionality
                )

                if score > best_fractionality:
                    best_fractionality = score
                    best_index = int(index)

        return best_index

    def _best_bound(
        self,
        nodes,
        sense
    ):
        if not nodes:
            return None

        if sense == "min":
            return min(
                node.bound
                for node in nodes
            )

        return max(
            node.bound
            for node in nodes
        )

    def _calculate_gap(
        self,
        incumbent,
        best_bound,
        sense
    ):
        if best_bound is None:
            return 0.0, 0.0

        if sense == "min":
            absolute_gap = incumbent - best_bound
        else:
            absolute_gap = best_bound - incumbent

        absolute_gap = max(
            0.0,
            float(absolute_gap)
        )

        denominator = max(
            abs(float(incumbent)),
            1.0
        )

        relative_gap = absolute_gap / denominator

        return absolute_gap, relative_gap

    def _is_better(
        self,
        objective,
        incumbent,
        sense
    ):
        if sense == "min":
            return objective < incumbent - 1e-9

        return objective > incumbent + 1e-9
