from dataclasses import dataclass
import numpy as np

from backend.core.optimization_model import OptimizationModel


@dataclass
class PresolveResult:
    """
    Result returned by the presolver.

    Stores the final reduced model together with enough
    information to reconstruct a solution in original
    variable space.
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
        Reconstruct a solution in original variable space.
        """

        reduced_solution = np.asarray(
            reduced_solution,
            dtype=float
        )

        if len(reduced_solution) != self.reduced_variables:
            raise ValueError(
                "Reduced solution length does not match reduced model."
            )

        solution = np.full(
            self.original_variables,
            np.nan,
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

    Performs safe reductions:

    - singleton constraint bound propagation
    - fixed-variable elimination
    - empty-row infeasibility detection
    - redundant empty-row removal
    - empty-column elimination
    - repeated presolve passes until a fixed point
    """

    def __init__(self, max_passes=100):
        self.max_passes = max_passes

    def _propagate_singleton_bounds(self, model):
        """
        Tighten variable bounds from singleton constraint rows.

        A singleton row has exactly one non-zero coefficient:

            lower <= a*x <= upper

        which gives bounds on x.
        """

        A = model.A.tocsr()

        row_nnz = np.diff(A.indptr)

        singleton_rows = np.where(
            row_nnz == 1
        )[0]

        if len(singleton_rows) == 0:
            return model, False, "OK", ""

        lower = model.variable_lower.copy()
        upper = model.variable_upper.copy()

        changed = False

        for row in singleton_rows:

            start = A.indptr[row]
            end = A.indptr[row + 1]

            variable_index = int(
                A.indices[start]
            )

            coefficient = float(
                A.data[start]
            )

            if coefficient == 0.0:
                continue

            constraint_lower = (
                model.constraint_lower[row]
            )

            constraint_upper = (
                model.constraint_upper[row]
            )

            if coefficient > 0.0:

                implied_lower = (
                    constraint_lower / coefficient
                )

                implied_upper = (
                    constraint_upper / coefficient
                )

            else:

                implied_lower = (
                    constraint_upper / coefficient
                )

                implied_upper = (
                    constraint_lower / coefficient
                )

            if model.variable_integrality[
                variable_index
            ] != 0:

                if np.isfinite(implied_lower):
                    implied_lower = np.ceil(
                        implied_lower
                    )

                if np.isfinite(implied_upper):
                    implied_upper = np.floor(
                        implied_upper
                    )

            new_lower = max(
                lower[variable_index],
                implied_lower
            )

            new_upper = min(
                upper[variable_index],
                implied_upper
            )

            if new_lower > new_upper:
                return (
                    model,
                    changed,
                    "INFEASIBLE",
                    (
                        "Singleton constraint produced "
                        "inconsistent variable bounds."
                    )
                )

            if new_lower != lower[variable_index]:
                changed = True

            if new_upper != upper[variable_index]:
                changed = True

            lower[variable_index] = new_lower
            upper[variable_index] = new_upper

        if not changed:
            return model, False, "OK", ""

        tightened_model = OptimizationModel(
            A=model.A,
            constraint_lower=model.constraint_lower.copy(),
            constraint_upper=model.constraint_upper.copy(),
            objective=model.objective.copy(),
            variable_lower=lower,
            variable_upper=upper,
            variable_integrality=model.variable_integrality.copy(),
            objective_sense=model.objective_sense
        )

        return tightened_model, True, "OK", ""

    def _make_result(
        self,
        model,
        original_variables,
        original_constraints,
        fixed_indices,
        fixed_values,
        reduced_to_original,
        reduced_constraint_to_original,
        objective_offset,
        status,
        message
    ):
        return PresolveResult(
            model=model,
            original_variables=original_variables,
            original_constraints=original_constraints,
            reduced_variables=model.n_variables,
            reduced_constraints=model.n_constraints,
            fixed_variable_indices=list(
                fixed_indices
            ),
            fixed_variable_values=np.asarray(
                fixed_values,
                dtype=float
            ),
            reduced_to_original=np.asarray(
                reduced_to_original,
                dtype=int
            ),
            reduced_constraint_to_original=np.asarray(
                reduced_constraint_to_original,
                dtype=int
            ),
            objective_offset=float(
                objective_offset
            ),
            status=status,
            message=message
        )

    def presolve(self, model):

        model.validate()

        original_variables = model.n_variables
        original_constraints = model.n_constraints

        current_model = model

        current_to_original_variables = np.arange(
            original_variables,
            dtype=int
        )

        current_to_original_constraints = np.arange(
            original_constraints,
            dtype=int
        )

        fixed_indices = []
        fixed_values = []

        objective_offset = 0.0

        for _ in range(self.max_passes):

            changed = False

            # -----------------------------------------------------
            # 1. Singleton constraint propagation
            # -----------------------------------------------------

            (
                current_model,
                singleton_changed,
                status,
                message
            ) = self._propagate_singleton_bounds(
                current_model
            )

            if status == "INFEASIBLE":
                return self._make_result(
                    current_model,
                    original_variables,
                    original_constraints,
                    fixed_indices,
                    fixed_values,
                    current_to_original_variables,
                    current_to_original_constraints,
                    objective_offset,
                    "INFEASIBLE",
                    message
                )

            if singleton_changed:
                changed = True

            # -----------------------------------------------------
            # 2. Fixed-variable elimination
            # -----------------------------------------------------

            fixed_mask = (
                current_model.variable_lower
                == current_model.variable_upper
            )

            current_fixed_indices = np.where(
                fixed_mask
            )[0]

            if len(current_fixed_indices) > 0:

                current_fixed_values = (
                    current_model.variable_lower[
                        fixed_mask
                    ].copy()
                )

                original_fixed_indices = (
                    current_to_original_variables[
                        current_fixed_indices
                    ]
                )

                fixed_objective = (
                    current_model.objective[
                        current_fixed_indices
                    ]
                    @ current_fixed_values
                )

                fixed_indices.extend(
                    original_fixed_indices.tolist()
                )

                fixed_values.extend(
                    current_fixed_values.tolist()
                )

                objective_offset += float(
                    fixed_objective
                )

                fixed_contribution = np.asarray(
                    current_model.A[
                        :,
                        fixed_mask
                    ]
                    @ current_fixed_values
                ).reshape(-1)

                new_lower = (
                    current_model.constraint_lower
                    - fixed_contribution
                )

                new_upper = (
                    current_model.constraint_upper
                    - fixed_contribution
                )

                free_mask = ~fixed_mask

                current_model = OptimizationModel(
                    A=current_model.A[
                        :,
                        free_mask
                    ].tocsr(),
                    constraint_lower=new_lower,
                    constraint_upper=new_upper,
                    objective=current_model.objective[
                        free_mask
                    ].copy(),
                    variable_lower=current_model.variable_lower[
                        free_mask
                    ].copy(),
                    variable_upper=current_model.variable_upper[
                        free_mask
                    ].copy(),
                    variable_integrality=current_model.variable_integrality[
                        free_mask
                    ].copy(),
                    objective_sense=current_model.objective_sense
                )

                current_to_original_variables = (
                    current_to_original_variables[
                        free_mask
                    ]
                )

                changed = True

            # -----------------------------------------------------
            # 3. Empty-row processing
            # -----------------------------------------------------

            row_nnz = np.diff(
                current_model.A.indptr
            )

            empty_rows = (
                row_nnz == 0
            )

            if np.any(empty_rows):

                infeasible_empty_rows = (
                    empty_rows
                    & (
                        current_model.constraint_lower
                        > 0.0
                    )
                    | (
                        empty_rows
                        & (
                            current_model.constraint_upper
                            < 0.0
                        )
                    )
                )

                if np.any(infeasible_empty_rows):
                    return self._make_result(
                        current_model,
                        original_variables,
                        original_constraints,
                        fixed_indices,
                        fixed_values,
                        current_to_original_variables,
                        current_to_original_constraints,
                        objective_offset,
                        "INFEASIBLE",
                        (
                            "An empty constraint row is infeasible "
                            "after presolve reductions."
                        )
                    )

                keep_rows = ~empty_rows

                current_model = OptimizationModel(
                    A=current_model.A[
                        keep_rows
                    ].tocsr(),
                    constraint_lower=current_model.constraint_lower[
                        keep_rows
                    ],
                    constraint_upper=current_model.constraint_upper[
                        keep_rows
                    ],
                    objective=current_model.objective.copy(),
                    variable_lower=current_model.variable_lower.copy(),
                    variable_upper=current_model.variable_upper.copy(),
                    variable_integrality=current_model.variable_integrality.copy(),
                    objective_sense=current_model.objective_sense
                )

                current_to_original_constraints = (
                    current_to_original_constraints[
                        keep_rows
                    ]
                )

                changed = True

            # -----------------------------------------------------
            # 4. Empty-column processing
            # -----------------------------------------------------

            column_nnz = np.diff(
                current_model.A.tocsc().indptr
            )

            empty_columns = (
                column_nnz == 0
            )

            empty_column_indices = np.where(
                empty_columns
            )[0]

            if len(empty_column_indices) > 0:

                empty_objective = (
                    current_model.objective[
                        empty_columns
                    ]
                )

                empty_lower = (
                    current_model.variable_lower[
                        empty_columns
                    ]
                )

                empty_upper = (
                    current_model.variable_upper[
                        empty_columns
                    ]
                )

                selected_values = np.empty(
                    len(empty_column_indices),
                    dtype=float
                )

                for i in range(
                    len(empty_column_indices)
                ):

                    c = empty_objective[i]
                    lower = empty_lower[i]
                    upper = empty_upper[i]

                    if current_model.objective_sense == "min":

                        if c < 0.0:
                            if not np.isfinite(upper):
                                return self._make_result(
                                    current_model,
                                    original_variables,
                                    original_constraints,
                                    fixed_indices,
                                    fixed_values,
                                    current_to_original_variables,
                                    current_to_original_constraints,
                                    objective_offset,
                                    "UNBOUNDED",
                                    (
                                        "An unconstrained variable can "
                                        "improve the objective without bound."
                                    )
                                )

                            value = upper

                        elif c > 0.0:
                            if not np.isfinite(lower):
                                return self._make_result(
                                    current_model,
                                    original_variables,
                                    original_constraints,
                                    fixed_indices,
                                    fixed_values,
                                    current_to_original_variables,
                                    current_to_original_constraints,
                                    objective_offset,
                                    "UNBOUNDED",
                                    (
                                        "An unconstrained variable can "
                                        "improve the objective without bound."
                                    )
                                )

                            value = lower

                        else:
                            if np.isfinite(lower):
                                value = lower
                            elif np.isfinite(upper):
                                value = upper
                            else:
                                value = 0.0

                    else:

                        if c > 0.0:
                            if not np.isfinite(upper):
                                return self._make_result(
                                    current_model,
                                    original_variables,
                                    original_constraints,
                                    fixed_indices,
                                    fixed_values,
                                    current_to_original_variables,
                                    current_to_original_constraints,
                                    objective_offset,
                                    "UNBOUNDED",
                                    (
                                        "An unconstrained variable can "
                                        "improve the objective without bound."
                                    )
                                )

                            value = upper

                        elif c < 0.0:
                            if not np.isfinite(lower):
                                return self._make_result(
                                    current_model,
                                    original_variables,
                                    original_constraints,
                                    fixed_indices,
                                    fixed_values,
                                    current_to_original_variables,
                                    current_to_original_constraints,
                                    objective_offset,
                                    "UNBOUNDED",
                                    (
                                        "An unconstrained variable can "
                                        "improve the objective without bound."
                                    )
                                )

                            value = lower

                        else:
                            if np.isfinite(lower):
                                value = lower
                            elif np.isfinite(upper):
                                value = upper
                            else:
                                value = 0.0

                    if (
                        value < lower
                        or value > upper
                    ):
                        return self._make_result(
                            current_model,
                            original_variables,
                            original_constraints,
                            fixed_indices,
                            fixed_values,
                            current_to_original_variables,
                            current_to_original_constraints,
                            objective_offset,
                            "INFEASIBLE",
                            (
                                "An unconstrained variable has no "
                                "feasible value within its bounds."
                            )
                        )

                    selected_values[i] = value

                original_empty_indices = (
                    current_to_original_variables[
                        empty_column_indices
                    ]
                )

                fixed_indices.extend(
                    original_empty_indices.tolist()
                )

                fixed_values.extend(
                    selected_values.tolist()
                )

                objective_offset += float(
                    empty_objective
                    @ selected_values
                )

                keep_columns = ~empty_columns

                current_model = OptimizationModel(
                    A=current_model.A[
                        :,
                        keep_columns
                    ].tocsr(),
                    constraint_lower=current_model.constraint_lower.copy(),
                    constraint_upper=current_model.constraint_upper.copy(),
                    objective=current_model.objective[
                        keep_columns
                    ].copy(),
                    variable_lower=current_model.variable_lower[
                        keep_columns
                    ].copy(),
                    variable_upper=current_model.variable_upper[
                        keep_columns
                    ].copy(),
                    variable_integrality=current_model.variable_integrality[
                        keep_columns
                    ].copy(),
                    objective_sense=current_model.objective_sense
                )

                current_to_original_variables = (
                    current_to_original_variables[
                        keep_columns
                    ]
                )

                changed = True

            # -----------------------------------------------------
            # 5. Fixed point / solved detection
            # -----------------------------------------------------

            if current_model.n_variables == 0:

                return self._make_result(
                    current_model,
                    original_variables,
                    original_constraints,
                    fixed_indices,
                    fixed_values,
                    current_to_original_variables,
                    current_to_original_constraints,
                    objective_offset,
                    "SOLVED",
                    "Presolve fixed all variables."
                )

            if not changed:
                break

        # ---------------------------------------------------------
        # Final reduced model
        # ---------------------------------------------------------

        fixed_indices = np.asarray(
            fixed_indices,
            dtype=int
        )

        fixed_values = np.asarray(
            fixed_values,
            dtype=float
        )

        if len(fixed_indices) > 0:

            order = np.argsort(
                fixed_indices
            )

            fixed_indices = (
                fixed_indices[order]
            )

            fixed_values = (
                fixed_values[order]
            )

        return self._make_result(
            current_model,
            original_variables,
            original_constraints,
            fixed_indices,
            fixed_values,
            current_to_original_variables,
            current_to_original_constraints,
            objective_offset,
            "OK",
            ""
        )