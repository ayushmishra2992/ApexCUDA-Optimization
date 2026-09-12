# 🚀 ApexCUDA: GPU-Accelerated Optimization Solver

> **Smart India Hackathon 2026** | **Problem Statement:** SIH26119  
> **Organization:** Mangalore Refinery and Petrochemicals Limited (MRPL)  
> **Team:** INDIOPTIMA

---

## 👥 Team & Contribution Matrix

| Team Member | GitHub Handle | Core Contributions |
| :--- | :--- | :--- |
| **Manya Aggarwal** 🌟 | [@manya0306](https://github.com/manya0306) | **Team Leader** · Problem Formulation, Presolver Pipeline, Constraint Reduction |
| **Aashna Das** | [@AashnaDas](https://github.com/AashnaDas) | **Solver Engineer** · Matrix-Free ADMM / PDHG Solver, Primal/Dual Residual Convergence |
| **Ayush Mishra** | [@ayushmishra2992](https://github.com/ayushmishra2992) | **GPU Systems** · NVIDIA CUDA & cuSPARSE SpMV Kernel Optimization |
| **Arnav Gupta** | [@arnavg19](https://github.com/arnavg19) | **Data Engineering** · 1,000-Flight Workload Generation, Fleet & Crew Disruption Modeling |
| **Riya Katiyar** | [@riyakatiyar123](https://github.com/riyakatiyar123) | **Full-Stack Engineer** · Next.js Mission Control Digital Twin, FastAPI Telemetry Server |
| **Vanshi Aneja** | [@vanshianeja](https://github.com/vanshianeja) | **Presentation & Documentation** · SIH 2026 Pitch Deck Architecture, Visual Layouts & Specifications |

## 💡 What is ApexCUDA?

ApexCUDA is a **GPU-accelerated optimization solver** designed to solve large and complex optimization problems efficiently.

The project focuses on problems such as **flight scheduling and resource allocation**, where thousands of decisions and millions of constraints may need to be processed.

Instead of relying only on traditional CPU-based solvers, ApexCUDA uses **NVIDIA GPUs and CUDA** to speed up computationally expensive operations.

---

## 🎯 What Problem Are We Solving?

Large industrial optimization problems can contain:

- Thousands of decision variables
- Millions of constraints
- Very large and sparse matrices
- High computational and memory requirements

Traditional approaches can become slow and memory-intensive.

**Our goal:** build an efficient, GPU-powered solver that can handle these large problems faster while reducing memory usage.

---

## ⚙️ How ApexCUDA Works

Our system is divided into several major parts:

### 1. Presolver

Reduces the size of the optimization problem by removing unnecessary or redundant constraints and variables.

**Result:** A smaller problem that is faster to solve.

### 2. Sparse Matrix Representation

Large optimization matrices are usually mostly empty.

Instead of storing all the empty values, we use **CSR (Compressed Sparse Row)** representation to store only the important values.

**Result:** Significant reduction in memory usage.

### 3. GPU Acceleration

Computationally expensive matrix operations are executed on an **NVIDIA GPU using CUDA/cuSPARSE**.

**Result:** Faster mathematical operations compared with CPU execution.

### 4. ADMM Solver

We use **ADMM (Alternating Direction Method of Multipliers)** to solve the optimization problem iteratively.

The solver monitors:

- Primal residual
- Dual residual
- Convergence

This helps determine whether the solution is approaching an optimal or feasible state.

### 5. Benchmark & Verification

We compare the performance of our solver against CPU-based approaches and reference solutions.

This allows us to measure:

- Execution time
- GPU speedup
- Memory usage
- Solution accuracy

### 6. Mission Control Dashboard

Our Next.js frontend provides a visual interface to monitor the solver.

It displays information such as:

- Solver status
- Flight and optimization data
- GPU performance
- Benchmark results
- Telemetry and system activity

---

## 📊 Key Results

| Metric | Result |
|---|---:|
| Dense problem size | **12.98 TB** |
| Sparse CSR representation | **108.6 MB** |
| Memory reduction | **~120,000×** |
| GPU speedup | **35.79×** |
| Flights tested | **1,000** |
| Constraints processed | **3.44 million** |
| Best reported solve time | **0.078 seconds** |

> These results demonstrate the potential of GPU acceleration and sparse matrix techniques for large-scale optimization problems.

---

## 🧰 Technologies Used

### Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS

### Backend

- Python
- FastAPI / Flask
- Optimization algorithms

### GPU Computing

- NVIDIA CUDA
- cuSPARSE
- Sparse Matrix Operations

### Mathematical Optimization

- ADMM
- MILP Presolver
- CSR Matrix Representation

### Development

- Git
- GitHub
- Jupyter Notebooks

---

## 🏗️ Project Structure

```text
ApexCUDA/
│
├── frontend/          # Mission Control dashboard
│
├── backend/           # Backend and solver integration
│
├── notebooks/         # Presolver and ADMM experiments
│
├── assets/            # Project assets
│
├── README.md
│
└── .gitignore