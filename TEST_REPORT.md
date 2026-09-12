# 🧪 IndiOptima Verification & Test Report
**Tester:** Riya Katiyar  
**Branch:** `generalization`  
**Date:** September 11, 2026  
**Status:** 20 PASSED, 1 DEFECT DISCOVERED (Runtime: ~0.4s)  

---

## 📊 Summary by Component

| Priority | Component | Test File | Status | Notes |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **OptimizationModel** | `tests/test_part1_model.py` | ⚠️ **BUG FOUND** | Inverted bounds (`lower > upper`) not rejected. |
| **2** | **MPS Parser** | `tests/test_part2_mps_parser.py` | ✅ **PASS** | `tiny_test`, `tiny_milp`, `tiny_max`, `fractional_milp` all pass. |
| **3** | **Generic Presolver** | `tests/test_part3_presolver.py` | ✅ **PASS** | Fixed-variable detection passes. |
| **4** | **Solution Verification**| `tests/test_part4_verification.py`| ✅ **PASS** | Constraint, bound, and integrality checks pass. |
| **5** | **LP Solver (ADMM)** | `tests/test_part5_lp_solver.py` | ✅ **PASS** | Small LP instances solve without crashing. |
| **6** | **MILP Solver (B&B)** | `tests/test_part6_milp_solver.py` | ✅ **PASS** | Branch-and-bound solves `fractional_milp.mps` within node limits. |
| **7** | **Generic Dispatcher**| `tests/test_part7_dispatcher.py` | ✅ **PASS** | Routes LP problems and rejects invalid backends properly. |
| **8** | **Benchmark Runner** | `tests/test_part8_benchmark.py` | ✅ **PASS** | Returns valid schema with required result metrics. |
| **9** | **Airline Formulation**| `tests/test_part9_airline_struct.py`| ✅ **PASS** | Scale verified: 472,798 vars, 3,443,145 constraints, 7,898,666 nonzeros. |
| **10**| **Regression Suite** | `tests/test_part10_regression.py`| ✅ **PASS** | Zero-row matrix edge case handled correctly. |

---

## 🐛 Bug Findings & Defect Report for Manya

### 🐞 Bug 1: `OptimizationModel.validate()` does not check for inverted bounds
* **File:** `backend/optimization_model.py`
* **Test Case:** `tests/test_part1_model.py::test_invalid_bounds_lower_greater_than_upper`
* **Reproduction:**
  ```python
  b_lower = np.array([10.0])
  b_upper = np.array([5.0])  # lower > upper (impossible empty feasible space)
  model = OptimizationModel(A, b_lower, b_upper, c, xl, xu, integ)
  model.validate()  # Passes silently without raising ValueError!
  ```
* **Expected Behavior:** `model.validate()` should raise `ValueError('Inverted bounds')`.
* **Suggested Fix for Manya in `backend/optimization_model.py`:**
  ```python
  if np.any(self.constraint_lower > self.constraint_upper):
      raise ValueError('constraint_lower cannot exceed constraint_upper')
  if np.any(self.variable_lower > self.variable_upper):
      raise ValueError('variable_lower cannot exceed variable_upper')
  ```
