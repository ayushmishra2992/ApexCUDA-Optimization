from dataclasses import dataclass
import numpy as np

from backend.core.optimization_model import OptimizationModel


@dataclass
class PresolveResult:
    """
    Result returned by a generic presolver.
    """

    model: OptimizationModel
    original_variables: int
    original_constraints: int
    reduced_variables: int
    reduced_constraints: int
    fixed_variable_indices: list

    @property
    def variables_removed(self):
        return self.original_variables - self.reduced_variables

    @property
    def constraints_removed(self):
        return self.original_constraints - self.reduced_constraints


class GenericPresolver:
    """
    Generic presolve interface for OptimizationModel.

    Currently detects fixed variables without modifying
    the model or its variable indexing.
    """

    def presolve(self, model):

        fixed_variables = (
            model.variable_lower == model.variable_upper
        )

        fixed_variable_indices = list(
            np.where(fixed_variables)[0]
        )

        return PresolveResult(
            model=model,
            original_variables=model.n_variables,
            original_constraints=model.n_constraints,
            reduced_variables=model.n_variables,
            reduced_constraints=model.n_constraints,
            fixed_variable_indices=fixed_variable_indices
        )