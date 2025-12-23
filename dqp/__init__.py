"""
dQP: Differentiation Through Black-Box Quadratic Programming Solvers

This package provides a modular framework for differentiating the solution to a 
quadratic programming problem (QP) with respect to its parameters, enabling the 
seamless integration of QPs into machine learning architectures and bilevel optimization.

Basic usage:
    from dqp import dQP
    from dqp.sparse_helper import initialize_torch_from_npz
    
    # Build settings for your solver
    settings = dQP.build_settings(solve_type="sparse", qp_solver="osqp", lin_solver="scipy SPLU")
    
    # Create the differentiable QP layer
    layer = dQP.dQP_layer(settings=settings)
    
    # Solve QP: min 1/2 x^T P x + q^T x s.t. Cx <= d, Ax = b
    z_star, lambda_star, mu_star, _, _ = layer(P, q, C, d, A, b)
    
    # Backpropagate through the solution
    z_star.sum().backward()
"""

from dqp.dQP import dQP_layer, build_settings
from dqp.sparse_helper import initialize_torch_from_npz, csc_torch_to_scipy, csc_scipy_to_torch
from dqp.lin_solvers import get_dense_solvers, get_sparse_solvers
from dqp.qp_diagnostic import benchmark

__version__ = "0.1.2"
__all__ = [
    "dQP_layer",
    "build_settings",
    "initialize_torch_from_npz",
    "csc_torch_to_scipy",
    "csc_scipy_to_torch",
    "get_dense_solvers",
    "get_sparse_solvers",
]
