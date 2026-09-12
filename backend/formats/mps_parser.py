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
        OBJSENSE
        ENDATA
    """

    SUPPORTED_BOUND_TYPES = {
        "LO",
        "LI",
        "UP",
        "UI",
        "FX",
        "FR",
        "MI",
        "PL",
        "BV",
    }

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
        integer_mode = False
        saw_endata = False

        with open(filepath, "r") as f:
            for line_number, raw_line in enumerate(f, start=1):
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
                    saw_endata = True
                    break

                parts = line.split()

                if section is None:
                    raise ValueError(
                        f"Unexpected data before a valid MPS section "
                        f"at line {line_number}: {line}"
                    )

                # ---------------------------------------------------------
                # ROWS
                # ---------------------------------------------------------
                if section == "ROWS":

                    if len(parts) < 2:
                        raise ValueError(
                            f"Malformed ROWS entry at line {line_number}: {line}"
                        )

                    row_type = parts[0].upper()
                    row_name = parts[1]

                    if row_type not in {"N", "L", "G", "E"}:
                        raise ValueError(
                            f"Unsupported row type '{row_type}' "
                            f"at line {line_number}"
                        )

                    if row_name in rows:
                        raise ValueError(
                            f"Duplicate row '{row_name}' "
                            f"at line {line_number}"
                        )

                    rows[row_name] = row_type
                    row_order.append(row_name)

                # ---------------------------------------------------------
                # COLUMNS
                # ---------------------------------------------------------
                elif section == "COLUMNS":

                    if len(parts) < 3:
                        raise ValueError(
                            f"Malformed COLUMNS entry at line "
                            f"{line_number}: {line}"
                        )

                    variable = parts[0]

                    # MPS integer markers
                    #
                    # Example:
                    # MARK0000 'MARKER' 'INTORG'
                    # MARK0001 'MARKER' 'INTEND'
                    if len(parts) >= 3 and parts[1] == "'MARKER'":

                        marker = parts[2].upper()

                        if marker == "'INTORG'":
                            integer_mode = True

                        elif marker == "'INTEND'":
                            integer_mode = False

                        else:
                            raise ValueError(
                                f"Unsupported MPS marker '{parts[2]}' "
                                f"at line {line_number}"
                            )

                        continue

                    if (len(parts) - 1) % 2 != 0:
                        raise ValueError(
                            f"Malformed COLUMNS row/value pairs "
                            f"at line {line_number}: {line}"
                        )

                    if variable not in columns:
                        columns[variable] = {}

                    if integer_mode:
                        integer_variables.add(variable)

                    # COLUMNS may contain one or two row/value pairs.
                    #
                    # If the same variable/row appears more than once,
                    # MPS semantics require the values to be accumulated.
                    for i in range(1, len(parts), 2):

                        row_name = parts[i]

                        try:
                            value = float(parts[i + 1])
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid numeric value '{parts[i + 1]}' "
                                f"at line {line_number}"
                            ) from exc

                        if row_name not in rows:
                            raise ValueError(
                                f"COLUMNS references unknown row "
                                f"'{row_name}' at line {line_number}"
                            )

                        columns[variable][row_name] = (
                            columns[variable].get(row_name, 0.0)
                            + value
                        )

                # ---------------------------------------------------------
                # OBJECTIVE SENSE
                # ---------------------------------------------------------
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

                # ---------------------------------------------------------
                # RHS
                # ---------------------------------------------------------
                elif section == "RHS":

                    if len(parts) < 3:
                        raise ValueError(
                            f"Malformed RHS entry at line "
                            f"{line_number}: {line}"
                        )

                    if (len(parts) - 1) % 2 != 0:
                        raise ValueError(
                            f"Malformed RHS row/value pairs "
                            f"at line {line_number}: {line}"
                        )

                    for i in range(1, len(parts), 2):

                        row_name = parts[i]

                        if row_name not in rows:
                            raise ValueError(
                                f"RHS references unknown row "
                                f"'{row_name}' at line {line_number}"
                            )

                        try:
                            value = float(parts[i + 1])
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid RHS value '{parts[i + 1]}' "
                                f"at line {line_number}"
                            ) from exc

                        rhs[row_name] = value

                # ---------------------------------------------------------
                # BOUNDS
                # ---------------------------------------------------------
                elif section == "BOUNDS":

                    if len(parts) < 3:
                        raise ValueError(
                            f"Malformed BOUNDS entry at line "
                            f"{line_number}: {line}"
                        )

                    bound_type = parts[0].upper()
                    variable = parts[2]

                    if bound_type not in self.SUPPORTED_BOUND_TYPES:
                        raise ValueError(
                            f"Unsupported bound type '{bound_type}' "
                            f"at line {line_number}"
                        )

                    if variable not in columns:
                        raise ValueError(
                            f"BOUNDS references unknown variable "
                            f"'{variable}' at line {line_number}"
                        )

                    if variable not in bounds:
                        bounds[variable] = {
                            "lower": 0.0,
                            "upper": np.inf,
                        }

                    # Bounds requiring a numeric value.
                    if bound_type in {"LO", "LI", "UP", "UI", "FX"}:

                        if len(parts) < 4:
                            raise ValueError(
                                f"Bound type '{bound_type}' requires "
                                f"a numeric value at line {line_number}"
                            )

                        try:
                            value = float(parts[3])
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid bound value '{parts[3]}' "
                                f"at line {line_number}"
                            ) from exc

                    if bound_type == "LO":
                        bounds[variable]["lower"] = value

                    elif bound_type == "LI":
                        bounds[variable]["lower"] = value
                        integer_variables.add(variable)

                    elif bound_type == "UP":
                        bounds[variable]["upper"] = value

                    elif bound_type == "UI":
                        bounds[variable]["upper"] = value
                        integer_variables.add(variable)

                    elif bound_type == "FX":
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

        # -------------------------------------------------------------
        # Required structure validation
        # -------------------------------------------------------------

        if not rows:
            raise ValueError("MPS file is missing the ROWS section")

        if not columns:
            raise ValueError("MPS file is missing the COLUMNS section")

        if not saw_endata:
            raise ValueError("MPS file is missing ENDATA")

        # -------------------------------------------------------------
        # Objective row
        # -------------------------------------------------------------

        objective_rows = [
            row_name
            for row_name in row_order
            if rows[row_name] == "N"
        ]

        if not objective_rows:
            raise ValueError(
                "MPS file is missing an objective row"
            )

        if len(objective_rows) > 1:
            raise ValueError(
                "MPS file contains multiple objective rows; "
                "only one objective row is supported"
            )

        objective_row = objective_rows[0]

        # -------------------------------------------------------------
        # Constraint rows
        # -------------------------------------------------------------

        constraint_rows = [
            row_name
            for row_name in row_order
            if rows[row_name] != "N"
        ]

        # -------------------------------------------------------------
        # Variable ordering
        # -------------------------------------------------------------

        variable_names = list(columns.keys())

        variable_index = {
            name: i
            for i, name in enumerate(variable_names)
        }

        row_index = {
            name: i
            for i, name in enumerate(constraint_rows)
        }

        # -------------------------------------------------------------
        # Sparse matrix construction
        # -------------------------------------------------------------

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
                (row_indices, col_indices),
            ),
            shape=(
                len(constraint_rows),
                len(variable_names),
            ),
        ).tocsr()

        # -------------------------------------------------------------
        # Constraint bounds
        # -------------------------------------------------------------

        constraint_lower = np.full(
            len(constraint_rows),
            -np.inf,
        )

        constraint_upper = np.full(
            len(constraint_rows),
            np.inf,
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

        # -------------------------------------------------------------
        # Variable bounds
        # -------------------------------------------------------------

        variable_lower = np.zeros(len(variable_names))

        variable_upper = np.full(
            len(variable_names),
            np.inf,
        )

        for variable, variable_bounds in bounds.items():

            j = variable_index[variable]

            variable_lower[j] = variable_bounds["lower"]
            variable_upper[j] = variable_bounds["upper"]

        # -------------------------------------------------------------
        # Validate variable bounds before creating the model
        # -------------------------------------------------------------

        for variable in variable_names:

            j = variable_index[variable]

            lower = variable_lower[j]
            upper = variable_upper[j]

            if lower > upper:
                raise ValueError(
                    f"Invalid bounds for variable '{variable}': "
                    f"lower bound {lower} is greater than "
                    f"upper bound {upper}"
                )

        # -------------------------------------------------------------
        # Variable integrality
        # -------------------------------------------------------------

        variable_integrality = np.zeros(
            len(variable_names),
            dtype=int,
        )

        for variable in integer_variables:

            j = variable_index[variable]
            variable_integrality[j] = 1

        for variable in binary_variables:

            j = variable_index[variable]
            variable_integrality[j] = 1

        # -------------------------------------------------------------
        # Build OptimizationModel
        # -------------------------------------------------------------

        return OptimizationModel(
            A=A,
            constraint_lower=constraint_lower,
            constraint_upper=constraint_upper,
            objective=objective,
            variable_lower=variable_lower,
            variable_upper=variable_upper,
            variable_integrality=variable_integrality,
            objective_sense=objective_sense,
        )