## dQP
<b>Differentiation Through Black-Box Quadratic Programming Solvers</b> [<a href="https://papers.nips.cc/paper_files/paper/2025/file/9bd9aa8c2326b67321eb1f5f281ba618-Paper-Conference.pdf">Paper</a>] [<a href="https://neurips.cc/virtual/2025/loc/san-diego/poster/119180">Video</a>] [<a href="https://neurips.cc/media/PosterPDFs/NeurIPS%202025/119180.png?t=1762316861.5409572">Poster</a>] <br> 
<a href="https://cwmagoon.github.io/">Magoon<sup>\*</sup></a>, <a href="https://563925743.github.io/">Yang<sup>\*</sup></a>, <a href="https://noamaig.github.io/">Aigerman</a>, <a href="https://shaharkov.github.io/">Kovalsky</a><br>
<i>NeurIPS (2025)</i>   


<p align=center>
  <img src="https://github.com/cwmagoon/dQP/blob/main/images/figure_introduction_increasing_structure_light.png?raw=true" alt="teaser" width="300" />
  <span style="display:inline-block; width: 100px;"></span> <!-- Spacer element -->
  <img src="https://github.com/cwmagoon/dQP/blob/main/images/figure_introduction_geometry_ant_light.png?raw=true" alt="teaser" width="300"  />
</p>

- [Installation](#installation)
- [Usage](#usage)
- [Options](#options)
- [Experiments](#development-setup-and-experiments)
- [Author Notes](#author-notes)

**dQP** is a modular framework for differentiating the solution to a quadratic programming problem (QP) with respect to its parameters, enabling the seamless integration of QPs into machine learning architectures and bilevel optimization. 
**dQP** supports <u>over 15 state-of-the-art QP solvers</u> by attaching a custom PyTorch layer to the open-source QP interface [[qpsolvers]](https://github.com/qpsolvers/qpsolvers).
It solves problems of the form,

<p align="center">
<img src="https://github.com/cwmagoon/dQP/blob/main/images/QP_formulation.png?raw=true" alt="QP" width="500"/>
</p>

The key mathematical structure we use to aid modularity is the special polyhedral geometry of a QP. 
This is illustrated below, where a QP is <u>locally</u> equivalent to a purely equality constrained problem, <i>i.e.</i>, both their solutions and derivatives match.

<p align="center">
<img src="https://github.com/cwmagoon/dQP/blob/main/images/figure_methods_schematic_light.png?raw=true" alt="teaser" width="500"/>
</p>


## Installation
    
[![PyPI version](https://img.shields.io/pypi/v/libdqp.svg)](https://pypi.org/project/libdqp/)
```bash
pip install libdqp
```

**dQP** is distributed via [[PyPI]](https://pypi.org/project/libdqp/) with easily pip-installable solvers included. See [[qpsolvers]](https://github.com/qpsolvers/qpsolvers) for information on including more. Solvers like PyPardiso are OS dependent and are excluded if necessary on the target platform. QP solvers like Gurobi are commercial but offer [[academic licenses]](https://www.gurobi.com/academia/academic-program-and-licenses/). 


## Usage

The following example applies [[OSQP]](https://github.com/osqp/osqp) for the forward pass and sparse symmetric indefinite linear solver [[QDLDL]](https://github.com/osqp/qdldl-python) for the backward pass. A companion code in the examples folder iterates through all loaded QP solvers on the same problem.

```python
import torch
import numpy as np

from dqp import dQP

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

# -----------------------------------------------------------------------------
# Define differentiable problem parameters
# -----------------------------------------------------------------------------

rows_P = torch.LongTensor([0, 1])
cols_P = torch.LongTensor([0, 1])
# sparse matrix nonzero values are differentiable (0s assumed fixed)
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

# -----------------------------------------------------------------------------
# Initialize and apply dQP
# -----------------------------------------------------------------------------
settings = dQP.build_settings(
    solve_type="sparse",
    qp_solver="osqp",
    lin_solver="qdldl",
    empty_batch=False,
)

layer = dQP.dQP_layer(settings=settings)

# Solve QP (forward pass)
z_star, lambda_star, mu_star, _, _ = layer(
    P.to_sparse_csc(), q, C.to_sparse_csc(), d
)

print("-"*80)
z_exact = np.array([0.5, 0.5])
mu_exact = np.array([1.0, 0.0, 0.0])
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

# Evaluate differentiable loss and backpropogate using torch autograd
L = 0.5 * z_star @ (torch.sparse.mm(P, z_star.unsqueeze(1)).squeeze(1)) + q @ z_star
L.backward()

print("-"*80)
# gradient projected onto subspace of nonzero values
print("dQP ∇_P L:", vals_P.grad.detach().numpy()) 
grad_P_exact = 0.5 * np.outer(z_exact, z_exact)
print("Analytical ∇_P L:", grad_P_exact[rows_P, cols_P]) 

print("dQP ∇_q L:     ", q.grad.detach().numpy())
print("Analytical ∇_q L:     ", z_exact)

print("dQP ∇_C L:", vals_C.grad.detach().numpy())
grad_C_exact = np.outer(mu_exact, z_exact)
print("Analytical ∇_C L:", grad_C_exact[rows_C, cols_C])

print("dQP ∇_d L:     ", d.grad.detach().numpy())
print("Analytical ∇_d L:     ", -mu_exact)
print("-"*80)
```

## Options

**dQP** supports a wide range of tunable settings.

|          Option          |                                                  Meaning                                                   |
|:------------------------:|:----------------------------------------------------------------------------------------------------------:|
|        qp_solver         | QP solver (supported by [[qpsolvers]](https://github.com/qpsolvers/qpsolvers?tab=readme-ov-file#solvers)). |
|        lin_solver        |                              Linear solver for the derivative (algorithm 1).                               |
|        solve_type        |                                       Dense or sparse (CSC format).                                        |
|         eps_abs          |                            Absolute tolerance on feasibility, optimality, etc.                             |
|         eps_rel          |                                            Relative tolerance.                                             |
|        eps_active        |                             Tolerance to decide the active set of constraints.                             |
|      dual_available      |                    Assert whether a given solver outputs duals, if not, solve for them.                    |
|  normalize_constraints   |                     Normalize each row of the constraints, so residuals are distances.                     |
|      refine_active       |                                Use heuristic active set refinement (Fig. 7)                                |
| warm_start_from_previous |               Save previous solution and use to warm-start (useful in bilevel optimization)                |
|       omp_parallel       |                        Use a simple parallel for loop for the forward over batches.                        |
|       empty_batch        |                            Include an empty batch dimension, even for batch 1.                             |
|    qp_solver_keywords    |                                     Additional keywords for qpsolvers.                                     |
|         verbose          |                                      Display additional information.                                       |
|           time           |                                              Display timings.                                              |
|        check_PSD         |                      Verify that Q is positive semi-definite (dense only), is costly.                      |

<b> Which solver do I choose for my problem? </b> We suggest perusing open-source benchmarks and reading about the basic classes of QP solver (e.g. first and second order). **dQP** includes a crude diagnostic tool for quickly evaluating the performance of QP solvers on a user-provided example (see examples folder).

## Development Setup and Experiments

<details>
<summary>
This script sets up the environment distributed with PyPI:
</summary>

```bash
git clone https://github.com/cwmagoon/dQP
cd dQP
conda create -y --name dQP python=3.9
conda activate dQP
pip install -e .
```


This script includes this and provides additional dependencies used in the experiments:

```bash
git clone https://github.com/cwmagoon/dQP
cd dQP
conda create -y --name dQP python=3.9
conda activate dQP

pip install torch==2.3.0+cpu -f https://download.pytorch.org/whl/torch_stable.html scipy numpy qpsolvers 
pip install clarabel cvxopt daqp ecos gurobipy highspy mosek osqp piqp proxsuite qpalm quadprog scs 
pip install qdldl pypardiso 

pip install optnet qpth cvxpylayers proxsuite
    
pip install matplotlib tensorboard pandas

pip install setproctitle

pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install libigl polyscope shapely robust_laplacian torchvision==0.18
conda install -c conda-forge ffmpeg
```
</details>



<br>We provide the code for our experiments in the experiments folder, including some additional directions on running them at the following:

* [Benchmark Experiment](./experiments/mega_test/README.md) 
* [Sudoku Experiment](./experiments/sudoku/README.md)
* [Geometry Experiment](./experiments/geometry/README.md)

## Author Notes

We greatly appreciate users applying our tool to their problems. Feel free to contact Connor or Fengyu with questions, issues, and suggestions; issues and forks on the repository page are welcome.

If you find **dQP** useful, please consider citing the associated paper:

```bibtex
@inproceedings{NEURIPS2025_9bd9aa8c,
 author = {Magoon, Connor and Yang, Fengyu and Aigerman, Noam and Kovalsky, Shahar},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 pages = {108486--108517},
 publisher = {Curran Associates, Inc.},
 title = {Differentiation Through Black-Box Quadratic Programming Solvers},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/9bd9aa8c2326b67321eb1f5f281ba618-Paper-Conference.pdf},
 volume = {38},
 year = {2025}
}
```