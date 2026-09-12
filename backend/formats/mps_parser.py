import numpy as np
from scipy.sparse import coo_matrix

from backend.core.optimization_model import OptimizationModel

class MPSParser:
    """
    Generic MPS parser for LP/MILP models.

    Supports the core MPS sections:
        NAME
        ROWS
        COLUMNS
        RHS
        BOUNDS
        ENDATA
    """

    def parse(self, filepath):

        rows = {}
        row_order = []

        columns = {}
        rhs = {}
        bounds = {}
        integer_variables = set()
        binary_variables = set()
        objective_sense = "min"

        section = None
        current_variable = None

        with open(filepath, "r") as f:

            for raw_line in f:

                line = raw_line.strip()

                if not line:
                    continue

                # Section headers
                if line.startswith("NAME"):
                    section = "NAME"
                    continue

                if line == "ROWS":
                    section = "ROWS"
                    continue

                if line == "COLUMNS":
                    section = "COLUMNS"
                    continue

                if line == "RHS":
                    section = "RHS"
                    continue

                if line == "BOUNDS":
                    section = "BOUNDS"
                    continue

                if line == "OBJSENSE":
                    section = "OBJSENSE"
                    continue

                if line == "ENDATA":
                    break

                parts = line.split()

                if section == "ROWS":

                    row_type = parts[0]
                    row_name = parts[1]

                    rows[row_name] = row_type
                    row_order.append(row_name)

                elif section == "COLUMNS":

                    variable = parts[0]

                    # MPS integer marker
                    # Example:
                    # MARK0000  'MARKER'  'INTORG'
                    # MARK0001  'MARKER'  'INTEND'
                    if len(parts) >= 3 and parts[1] == "'MARKER'":

                        if parts[2] == "'INTORG'":
                            current_variable = "__INTORG__"

                        elif parts[2] == "'INTEND'":
                            current_variable = "__INTEND__"

                        continue

                    if variable not in columns:
                        columns[variable] = {}

                    # Variables appearing between INTORG and INTEND
                    if current_variable == "__INTORG__":
                        integer_variables.add(variable)

                    # COLUMNS may contain one or two row/value pairs
                    for i in range(1, len(parts), 2):

                        row_name = parts[i]
                        value = float(parts[i + 1])

                        columns[variable][row_name] = value

                elif section == "OBJSENSE":

                    sense = parts[0].upper()

                    if sense == "MIN":
                        objective_sense = "min"

                    elif sense == "MAX":
                        objective_sense = "max"

                    else:
                        raise ValueError(
                            f"Unsupported objective sense: {sense}"
                        )

                elif section == "RHS":

                    for i in range(1, len(parts), 2):

                        row_name = parts[i]
                        value = float(parts[i + 1])

                        rhs[row_name] = value

                elif section == "BOUNDS":

                    bound_type = parts[0]
                    variable = parts[2]

                    if variable not in bounds:
                        bounds[variable] = {
                            "lower": 0.0,
                            "upper": np.inf
                        }

                    if bound_type == "LO":
                        bounds[variable]["lower"] = float(parts[3])

                    elif bound_type == "LI":
                        bounds[variable]["lower"] = float(parts[3])
                        integer_variables.add(variable)

                    elif bound_type == "UP":
                        bounds[variable]["upper"] = float(parts[3])

                    elif bound_type == "UI":
                        bounds[variable]["upper"] = float(parts[3])
                        integer_variables.add(variable)

                    elif bound_type == "FX":
                        value = float(parts[3])
                        bounds[variable]["lower"] = value
                        bounds[variable]["upper"] = value

                    elif bound_type == "FR":
                        bounds[variable]["lower"] = -np.inf
                        bounds[variable]["upper"] = np.inf

                    elif bound_type == "MI":
                        bounds[variable]["lower"] = -np.inf

                    elif bound_type == "PL":
                        bounds[variable]["upper"] = np.inf

                    elif bound_type == "BV":
                        bounds[variable]["lower"] = 0.0
                        bounds[variable]["upper"] = 1.0
                        binary_variables.add(variable)

        # Objective row
        objective_row = None

        for row_name in row_order:
            if rows[row_name] == "N":
                objective_row = row_name
                break

        # Constraint rows
        constraint_rows = [
            row_name
            for row_name in row_order
            if rows[row_name] != "N"
        ]

        # Variable ordering
        variable_names = list(columns.keys())
        variable_index = {
            name: i
            for i, name in enumerate(variable_names)
        }

        row_index = {
            name: i
            for i, name in enumerate(constraint_rows)
        }

        # Sparse matrix construction
        row_indices = []
        col_indices = []
        data = []

        objective = np.zeros(len(variable_names))

        for variable, entries in columns.items():

            j = variable_index[variable]

            for row_name, value in entries.items():

                if row_name == objective_row:
                    objective[j] = value

                elif row_name in row_index:

                    row_indices.append(row_index[row_name])
                    col_indices.append(j)
                    data.append(value)

        A = coo_matrix(
            (
                data,
                (row_indices, col_indices)
            ),
            shape=(
                len(constraint_rows),
                len(variable_names)
            )
        ).tocsr()

        # Constraint bounds
        constraint_lower = np.full(
            len(constraint_rows),
            -np.inf
        )

        constraint_upper = np.full(
            len(constraint_rows),
            np.inf
        )

        for i, row_name in enumerate(constraint_rows):

            value = rhs.get(row_name, 0.0)

            if rows[row_name] == "L":
                constraint_upper[i] = value

            elif rows[row_name] == "G":
                constraint_lower[i] = value

            elif rows[row_name] == "E":
                constraint_lower[i] = value
                constraint_upper[i] = value

        # Variable bounds
        variable_lower = np.zeros(len(variable_names))
        variable_upper = np.full(
            len(variable_names),
            np.inf
        )

        for variable, variable_bounds in bounds.items():

            j = variable_index[variable]

            variable_lower[j] = variable_bounds["lower"]
            variable_upper[j] = variable_bounds["upper"]

        # Initial implementation: all variables continuous
        variable_integrality = np.zeros(
            len(variable_names),
            dtype=int
        )

        for variable in integer_variables:
            j = variable_index[variable]
            variable_integrality[j] = 1

        for variable in binary_variables:
            j = variable_index[variable]
            variable_integrality[j] = 1
            
        return OptimizationModel(
            A=A,
            constraint_lower=constraint_lower,
            constraint_upper=constraint_upper,
            objective=objective,
            variable_lower=variable_lower,
            variable_upper=variable_upper,
            variable_integrality=variable_integrality,
            objective_sense=objective_sense
        )