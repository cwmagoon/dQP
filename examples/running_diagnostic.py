from dqp import benchmark
import numpy as np
import scipy as sp

def generate_random_qp(dim, nIneq, nEq):    
    # Generate symmetric positive definite matrix Q
    P = np.random.rand(dim, dim)
    Q = np.matmul(P, P.T) + 1e-4 * np.eye(dim)
    
    # Generate linear term q
    q = np.random.rand(dim)
    
    # Inequality constraints Gx <= h
    G = np.random.rand(nIneq, dim)
    z0 = np.ones((dim, 1))
    s0 = np.ones((nIneq, 1))
    h = np.matmul(G, z0) + s0  # guarantee there exists feasible z0 with Gz0 < h
    h = h.squeeze(-1)
    
    # Equality constraints Ax = b
    A = np.random.rand(nEq, dim)
    b = np.matmul(A, z0).squeeze(-1)

    return Q, q, G, h, A, b

Q,q,G,h,A,b = generate_random_qp(200,20,20)
# Q, G, A = [sp.sparse.csc_matrix(M) for M in (Q, G, A)] # to debug sparse solvers (recommend plugging real sparse problem data)

# last argument is reference solver from which solution errors are measured; 
# 2nd order methods heuristically expected to be a better reference
benchmark("random",Q,q,G,h,A,b,"gurobi") 
