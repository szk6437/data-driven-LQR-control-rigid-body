"""
Data Generation for Rigid Body Rotational Dynamics.

Note: Applicable to non-affine input nonlinear systems as long as Assumptions 1, 2, and 3 hold.
"""

import pickle
from pathlib import Path
import numpy as np

def main():
    # --- System Parameters ---
    T = 0.1
    input_interval = 20
    L = 6
    decay_rate = 5.0
    
    J = np.array([
        [2.5, 0.5, 0.7],
        [0.5, 1.8, 1.1],
        [0.7, 1.1, 1.7]
    ])
    J_inv = np.linalg.inv(J)
    
    # Damping matrix
    D = np.zeros((3, 3))
    
    # Pre-defined scaling matrix for inputs
    EI = np.array([
        [-0.1, -0.4, -0.6],
        [-0.3, -0.1, -0.2],
        [-0.2, -0.3, -0.4],
        [ 0.1,  0.2,  0.3],
        [ 0.2,  0.3,  0.4],
        [ 0.1,  0.2,  0.4]
    ]).T

    # --- Highly Efficient Vectorized Data Generation ---
    
    # 1. Generate all inputs simultaneously via broadcasting
    time_vector = np.arange(input_interval) * T
    Tau_results = EI.T[:, :, np.newaxis] * np.exp(-decay_rate * time_vector)
    
    # 2. Initialize state trajectories
    Omega_results = np.zeros((L, 3, input_interval))
    
    # Set random initial conditions for all L experiments at once
    np.random.seed(42) 
    Omega_results[:, :, 0] = np.random.rand(L, 3)
    
    # 3. Vectorized Simulation (Simulating all L experiments simultaneously)
    for k in range(1, input_interval):
        # Extract states and inputs for current step across all experiments
        omega_k = Omega_results[:, :, k-1]  # Shape: (L, 3)
        u_k = Tau_results[:, :, k-1]        # Shape: (L, 3)
        
        # Vectorized cross-product to replace S(omega) matrix multiplication
        J_omega = omega_k @ J.T
        cross_term = np.cross(omega_k, J_omega)
        
        # Forward Euler update for all L trajectories at once
        Omega_results[:, :, k] = (
            omega_k 
            - T * (cross_term @ J_inv.T) 
            - T * (omega_k @ D.T) 
            + T * (u_k @ J_inv.T)
        )

    # --- Save Data ---
    filename = Path('StateInput.spydata')
    data_dict = {'data': {'states': Omega_results, 'inputs': Tau_results}}
    
    with open(filename, 'wb') as file:
        pickle.dump(data_dict, file)
        
    print(f'Successfully generated and saved data to {filename.name}')

if __name__ == '__main__':
    main()
