import os
import pytest
from backend.benchmark import run_benchmark

def test_benchmark_returns_required_keys():
    path = "examples/tiny_test.mps"
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
        
    data = run_benchmark(path)
    required_keys = [
        "file", "problem_type", "variables", "constraints", 
        "nonzeros", "solve_time", "status", "objective"
    ]
    for key in required_keys:
        assert key in data, f"Missing required key: {key}"
