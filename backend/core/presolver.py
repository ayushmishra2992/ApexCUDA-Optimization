from dataclasses import dataclass
import numpy as np

from backend.core.optimization_model import OptimizationModel


@dataclass
class PresolveResult:
    """
    Result returned by the presolver.

    Stores the reduced model together with enough information
    to reconstruct a solution in the original variable space.
    """

    model: OptimizationModel
    original_variables: int
    original_constraints: int
    reduced_variables: int
    reduced_constraints: int
    fixed_variable_indices: list
    fixed_variable_values: np.ndarray
    reduced_to_original: np.ndarray
    reduced_constraint_to_original: np.ndarray
    objective_offset: float = 0.0
    status: str = "OK"
    message: str = ""

    @property
    def variables_removed(self):
        return self.original_variables - self.reduced_variables

    @property
    def constraints_removed(self):
        return self.original_constraints - self.reduced_constraints

    def postsolve(self, reduced_solution):
        """
        Reconstruct a solution in the original variable space.
        """

        reduced_solution = np.asarray(
            reduced_solution,
            dtype=float
        )

        if len(reduced_solution) != self.reduced_variables:
            raise ValueError(
                "Reduced solution length does not match reduced model."
            )

        solution = np.empty(
            self.original_variables,
            dtype=float
        )

        if self.fixed_variable_indices:
            solution[
                self.fixed_variable_indices
            ] = self.fixed_variable_values

        solution[
            self.reduced_to_original
        ] = reduced_solution

        return solution


class GenericPresolver:
    """
    Generic sparse presolver for OptimizationModel.

    Performs:
    - fixed-variable elimination
    - empty-row infeasibility detection
    - redundant empty-row removal
    """

    def presolve(self, model):

        fixed_mask = (
            model.variable_lower == model.variable_upper
        )

        fixed_variable_indices = np.where(
            fixed_mask
        )[0]

        fixed_variable_values = model.variable_lower[
            fixed_mask
        ].copy()

        free_mask = ~fixed_mask

        # Contribution of fixed variables to every constraint.
        if len(fixed_variable_indices) > 0:

            fixed_matrix = model.A[:, fixed_mask]

            fixed_contribution = np.asarray(
                fixed_matrix @ fixed_variable_values
            ).reshape(-1)

        else:

            fixed_contribution = np.zeros(
                model.n_constraints,
                dtype=float
            )

        reduced_lower = (
            model.constraint_lower - fixed_contribution
        )

        reduced_upper = (
            model.constraint_upper - fixed_contribution
        )

        # Keep only non-fixed columns.
        reduced_A = model.A[:, free_mask].tocsr()

        # Detect rows that contain no remaining variables.
        row_nnz = np.diff(
            reduced_A.indptr
        )

        empty_rows = (
            row_nnz == 0
        )

        # An empty row represents:
        #
        #     lower <= 0 <= upper
        #
        # If zero is outside these bounds, the model is infeasible.
        infeasible_empty_rows = (
            empty_rows
            & (
                (reduced_lower > 0.0)
                | (reduced_upper < 0.0)
            )
        )

        if np.any(infeasible_empty_rows):

            return PresolveResult(
                model=model,
                original_variables=model.n_variables,
                original_constraints=model.n_constraints,
                reduced_variables=model.n_variables,
                reduced_constraints=model.n_constraints,
                fixed_variable_indices=list(
                    fixed_variable_indices
                ),
                fixed_variable_values=fixed_variable_values,
                reduced_to_original=np.arange(
                    model.n_variables,
                    dtype=int
                ),
                reduced_constraint_to_original=np.arange(
                    model.n_constraints,
                    dtype=int
                ),
                objective_offset=0.0,
                status="INFEASIBLE",
                message=(
                    "An empty constraint row is infeasible "
                    "after fixed-variable substitution."
                )
            )

        # Remove feasible empty rows because they impose no restriction.
        keep_rows = ~empty_rows

        reduced_A = reduced_A[
            keep_rows
        ].tocsr()

        reduced_lower = reduced_lower[
            keep_rows
        ]

        reduced_upper = reduced_upper[
            keep_rows
        ]

        reduced_constraint_to_original = np.where(
            keep_rows
        )[0].astype(int)

        reduced_objective = model.objective[
            free_mask
        ].copy()

        reduced_variable_lower = model.variable_lower[
            free_mask
        ].copy()

        reduced_variable_upper = model.variable_upper[
            free_mask
        ].copy()

        reduced_integrality = model.variable_integrality[
            free_mask
        ].copy()

        reduced_to_original = np.where(
            free_mask
        )[0].astype(int)

        objective_offset = float(
            model.objective[fixed_mask]
            @ fixed_variable_values
        )

        reduced_model = OptimizationModel(
            A=reduced_A,
            constraint_lower=reduced_lower,
            constraint_upper=reduced_upper,
            objective=reduced_objective,
            variable_lower=reduced_variable_lower,
            variable_upper=reduced_variable_upper,
            variable_integrality=reduced_integrality,
            objective_sense=model.objective_sense
        )

        return PresolveResult(
            model=reduced_model,
            original_variables=model.n_variables,
            original_constraints=model.n_constraints,
            reduced_variables=reduced_model.n_variables,
            reduced_constraints=reduced_model.n_constraints,
            fixed_variable_indices=list(
                fixed_variable_indices
            ),
            fixed_variable_values=fixed_variable_values,
            reduced_to_original=reduced_to_original,
            reduced_constraint_to_original=(
                reduced_constraint_to_original
            ),
            objective_offset=objective_offset,
            status="OK",
            message=""
        )