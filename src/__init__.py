"""
Home Office Optimization
========================

Módulos para simulação, otimização e avaliação da alocação
de dias de home office.
"""

from .generate_data import generate_simulated_dataset
from .load_data import load_responses, load_restrictions
from .optimizer import optimize_assignment
from .random_assignment import generate_random_assignment
from .metrics import calculate_metrics

__all__ = [
    "generate_simulated_dataset",
    "load_responses",
    "load_restrictions",
    "optimize_assignment",
    "generate_random_assignment",
    "calculate_metrics",
]