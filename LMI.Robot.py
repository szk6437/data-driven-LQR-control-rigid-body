"""
Data-Driven LMI Optimal Control.

This script solves a finite-horizon optimization problem using recorded 
state and input data to compute time-varying feedback gains. It is applicable 
to non-affine input nonlinear systems as long as Assumptions 1, 2, and 3 hold.
"""

import pickle
from pathlib import Path
import numpy as np
import cvxpy as cp

def main():
    # --- 1. Load Data ---
    input_file = Path('StateInput.spydata')
    with open(input_file, 'rb') as file:
        data_dict = pickle.load(file)

    States = data_dict['data']['states']
    Inputs = data_dict['data']['inputs']

    Nexp, Ns, Nsamples = States.shape
    Nsamples_reduced = min(Nsamples, 30)

    # --- 2. Highly Efficient Data Reshaping ---
    # Transpose data from (Nexp, Ns, Nsamples) -> (Ns, Nexp, Nsamples)
    # This completely eliminates the need for slow nested for-loops and vstack
    X = np.transpose(States[:, :, :Nsamples_reduced], (1, 0, 2))
    U = np.transpose(Inputs[:, :, :Nsamples_reduced], (1, 0, 2))

    # --- 3. Optimization Setup ---
    # Define optimization variables
    S = [cp.Variable((Ns, Ns), PSD=True) for _ in range(Nsamples_reduced + 1)]
    H = [cp.Variable((Nexp, Ns)) for _ in range(Nsamples_reduced)]
    O = [cp.Variable((Nexp, Nexp), PSD=True) for _ in range(Nsamples_reduced)]

    # Define constant cost weight matrices
    Q_f = 10000 * np.eye(Ns)
    Q_k = 100 * np.eye(Ns)
    R_k = 0.12 * np.eye(Nexp)
    I_ns = np.eye(Ns)

    # Build cost function
    cost = cp.trace(Q_f @ S[Nsamples_reduced])
    for k in range(Nsamples_reduced):
        cost += cp.trace(Q_k @ S[k]) + cp.trace(O[k] @ R_k)

    # Build LMI constraints
    constraints = [S[0] >> I_ns]
    for k in range(Nsamples_reduced - 1):
        X_next = X[:, :, k + 1]
        X_curr = X[:, :, k]
        
        # Stability LMI
        LMI_1 = cp.bmat([
            [S[k + 1] - I_ns, X_next @ H[k]], 
            [H[k].T @ X_next.T, S[k]]
        ])
        
        # Control effort LMI
        LMI_2 = cp.bmat([
            [O[k], R_k @ H[k]], 
            [H[k].T @ R_k.T, S[k]]
        ])
        
        constraints += [
            LMI_1 >> 0,
            LMI_2 >> 0,
            S[k] == X_curr @ H[k]
        ]

    # --- 4. Solve Optimization ---
    problem = cp.Problem(cp.Minimize(cost), constraints)
    print("Solving LMI optimization...")
    problem.solve(solver=cp.SCS)  # SCS or MOSEK are usually faster for LMIs

    # --- 5. Gain Recovery ---
    K_optimal = []
    for k in range(Nsamples_reduced):
        if H[k].value is not None and S[k].value is not None:
            try:
                # S_k is guaranteed to be >= I by LMI, so it is strictly positive definite.
                # solve() is numerically more stable than explicit inv()
                K_opt = U[:, :, k] @ H[k].value @ np.linalg.inv(S[k].value)
                K_optimal.append(K_opt)
            except np.linalg.LinAlgError:
                print(f"Warning: Numerical instability at sample {k}.")
                K_optimal.append(None)
        else:
            print(f"Warning: Infeasible/No solution at sample {k}.")
            K_optimal.append(None)

    # --- 6. Output and Save ---
    output_filename = Path('OptimalK.spydata')
    with open(output_filename, 'wb') as file:
        pickle.dump({'OptimalK': K_optimal}, file)

    print(f"Optimal K* successfully saved to {output_filename.name}")

if __name__ == '__main__':
    main()