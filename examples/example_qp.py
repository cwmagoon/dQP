import torch
import numpy as np
from dqp import dQP
import qpsolvers

def qp_2d(qp_solver):
    # -----------------------------------------------------------------------------
    #   minimize     (1/2) zᵀ P z + qᵀ z
    #   subject to   C z ≤ d
    #
    #   P = 2 I₂
    #   q = [0, 0]
    #   C = [ [-1 -1],
    #         [-1  0],
    #         [ 0 -1] ]
    #   d = [-1, 0, 0]
    #
    #   z* = [1/2, 1/2]
    #   μ* = [1, 0, 0] (first constraint active)
    # -----------------------------------------------------------------------------

    print("\n" + "="*60)
    print(f"Using qp solver: {qp_solver}")
    print("="*60)

    # -------------------------------------------------------------------------
    # Define differentiable problem parameters (sparse COO)
    # -------------------------------------------------------------------------
    rows_P = torch.LongTensor([0, 1])
    cols_P = torch.LongTensor([0, 1])
    vals_P = torch.tensor([2.0, 2.0], dtype=torch.float64, requires_grad=True)
    P = torch.sparse_coo_tensor(
        torch.stack([rows_P, cols_P]),
        vals_P,
        size=(2, 2),
        dtype=torch.float64,
    )
    q = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)

    rows_C = torch.LongTensor([0, 0, 1, 2])
    cols_C = torch.LongTensor([0, 1, 0, 1])
    vals_C = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float64, requires_grad=True)
    C = torch.sparse_coo_tensor(
        torch.stack([rows_C, cols_C]),
        vals_C,
        size=(3, 2),
        dtype=torch.float64,
    )
    d = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)

    # -------------------------------------------------------------------------
    # Initialize and apply dQP
    # -------------------------------------------------------------------------
    settings = dQP.build_settings(
        solve_type="sparse",
        qp_solver=qp_solver,
        lin_solver="qdldl",
        empty_batch=False,
    )
    layer = dQP.dQP_layer(settings=settings)

    z_star, lambda_star, mu_star, _, _ = layer(P.to_sparse_csc(), q, C.to_sparse_csc(), d)

    z_exact = np.array([0.5, 0.5])
    mu_exact = np.array([1.0, 0.0, 0.0])

    print("-"*80)
    print("dQP active set J:", layer.active)
    print("dQP z*:", z_star.detach().numpy())
    print("Analytical z*:", z_exact)
    print("dQP μ* :", mu_star.detach().numpy())
    print("Analytical μ*:", mu_exact)
    print("-"*80)

    # -----------------------------------------------------------------------------
    # Backpropagate (differentiate) through some scalar-valued loss. 
    # A simple example is the optimal value function:
    #       p*(θ) = f(z*(θ),θ) = (1/2) z*ᵀ P z* + qᵀ z*
    #
    # In this case, the Envelope theorem https://en.wikipedia.org/wiki/Envelope_theorem 
    # yields a simple expression to use as reference, 
    # by simplifying the term ∇_z f • ∇_θ z* using Lagrangian stationarity at z*(θ), 
    #       ∇_P p = (1/2) z* z*ᵀ 
    #       ∇_q p = z*
    #       ∇_C p = μ* z*ᵀ  
    #       ∇_d p = -μ*
    # -----------------------------------------------------------------------------
    L = 0.5 * z_star @ (torch.sparse.mm(P, z_star.unsqueeze(1)).squeeze(1)) + q @ z_star
    L.backward()

    print("dQP ∇_P L:", vals_P.grad.detach().numpy())
    grad_P_exact = 0.5 * np.outer(z_exact, z_exact)
    print("Analytical ∇_P L:", grad_P_exact[rows_P, cols_P])

    print("dQP ∇_q L:", q.grad.detach().numpy())
    print("Analytical ∇_q L:", z_exact)

    print("dQP ∇_C L:", vals_C.grad.detach().numpy())
    grad_C_exact = np.outer(mu_exact, z_exact)
    print("Analytical ∇_C L:", grad_C_exact[rows_C, cols_C])

    print("dQP ∇_d L:", d.grad.detach().numpy())
    print("Analytical ∇_d L:", -mu_exact)
    print("-"*80)


if __name__ == "__main__":
    for solver in qpsolvers.sparse_solvers: 
        try:
            qp_2d(solver)
        except Exception as e:
            print(e)