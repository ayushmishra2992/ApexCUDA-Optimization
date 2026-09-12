from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class OptimizationModel:
    """
    Generic optimization model representation.

    Supports the common mathematical structure needed for
    LP/MILP/QP-style solver interfaces.
    """

    A: csr_matrix
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    objective: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    variable_integrality: np.ndarray
    objective_sense: str = "min"

    def __post_init__(self):
        # Convert sparse matrix to CSR representation
        self.A = self.A.tocsr()

        # Convert arrays to NumPy arrays
        self.constraint_lower = np.asarray(
            self.constraint_lower,
            dtype=float
        )

        self.constraint_upper = np.asarray(
            self.constraint_upper,
            dtype=float
        )

        self.objective = np.asarray(
            self.objective,
            dtype=float
        )

        self.variable_lower = np.asarray(
            self.variable_lower,
            dtype=float
        )

        self.variable_upper = np.asarray(
            self.variable_upper,
            dtype=float
        )

        self.variable_integrality = np.asarray(
            self.variable_integrality,
            dtype=int
        )

        self.validate()

        self.objective_sense = self.objective_sense.lower()

        if self.objective_sense not in ["min", "max"]:
            raise ValueError(
                "Objective sense must be either 'min' or 'max'."
            )

    @property
    def n_constraints(self):
        return self.A.shape[0]

    @property
    def n_variables(self):
        return self.A.shape[1]

    @property
    def nnz(self):
        return self.A.nnz

    @property
    def n_integer_variables(self):
        return int(np.sum(self.variable_integrality != 0))

    @property
    def n_continuous_variables(self):
        return int(np.sum(self.variable_integrality == 0))

    @property
    def problem_type(self):
        if self.n_integer_variables > 0:
            return "MILP"

        return "LP"

    def summary(self):
        return {
            "problem_type": self.problem_type,
            "variables": self.n_variables,
            "constraints": self.n_constraints,
            "nonzeros": self.nnz,
            "integer_variables": self.n_integer_variables,
            "continuous_variables": self.n_continuous_variables,
            "objective_sense": self.objective_sense,
        }

    def validate(self):
        if np.any(self.constraint_lower > self.constraint_upper):
            raise ValueError("Inverted constraint bounds: lower bound cannot exceed upper bound.")
        if np.any(self.variable_lower > self.variable_upper):
            raise ValueError("Inverted variable bounds: lower bound cannot exceed upper bound.")

        if self.A.ndim != 2:
            raise ValueError(
                "A must be a 2-dimensional sparse matrix."
            )

        if len(self.objective) != self.n_variables:
            raise ValueError(
                "Objective length must equal number of variables."
            )

        if len(self.variable_lower) != self.n_variables:
            raise ValueError(
                "Variable lower bounds must match number of variables."
            )

        if len(self.variable_upper) != self.n_variables:
            raise ValueError(
                "Variable upper bounds must match number of variables."
            )

        if len(self.variable_integrality) != self.n_variables:
            raise ValueError(
                "Variable integrality must match number of variables."
            )

        if len(self.constraint_lower) != self.n_constraints:
            raise ValueError(
                "Constraint lower bounds must match number of constraints."
            )

        if len(self.constraint_upper) != self.n_constraints:
            raise ValueError(
                "Constraint upper bounds must match number of constraints."
            )

        if np.any(self.variable_lower > self.variable_upper):
            raise ValueError(
                "Some variable lower bounds exceed upper bounds."
            )

        if not np.all(
            np.isin(self.variable_integrality, [0, 1])
        ):
            raise ValueError(
                "Variable integrality must contain only 0 or 1."
            )

        if np.isnan(self.objective).any():
            raise ValueError(
                "Objective contains NaN values."
            )

        if np.isnan(self.variable_lower).any():
            raise ValueError(
                "Variable lower bounds contain NaN values."
            )

        if np.isnan(self.variable_upper).any():
            raise ValueError(
                "Variable upper bounds contain NaN values."
            )

        if np.isnan(self.constraint_lower).any():
            raise ValueError(
                "Constraint lower bounds contain NaN values."
            )

        if np.isnan(self.constraint_upper).any():
            raise ValueError(
                "Constraint upper bounds contain NaN values."
            )

        if np.isnan(self.A.data).any():
            raise ValueError(
                "Constraint matrix contains NaN values."
            )


@dataclass
class OptimizationResult:
    """
    Standardized result returned by an IndiOptima solver.
    """

    status: str
    objective: float
    solution: np.ndarray
    solve_time: float
    iterations: int
    constraint_violation: float
    bound_violation: float
    backend: str
    problem_type: str