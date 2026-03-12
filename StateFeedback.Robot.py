"""
Closed-Loop Simulation with Optimal State Feedback.

This script simulates the rigid body dynamics using the optimal 
data-driven feedback gains. Note: This approach is applicable to 
non-affine input nonlinear systems as long as Assumptions 1, 2, and 3 hold.
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(time: np.ndarray, data: np.ndarray, title: str, ylabel: str, labels: list):
    """Utility function for plotting step responses of states and inputs."""
    plt.figure(figsize=(8, 5))
    for i, label in enumerate(labels):
        plt.step(time, data[i, :], label=label, where='post')
    
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # --- 1. System Parameters ---
    T = 0.1
    t_total = 2.0
    num_steps = int(t_total / T)
    time = np.linspace(0, t_total, num_steps)
    
    J = np.array([
        [2.5, 0.5, 0.7],
        [0.5, 1.8, 1.1],
        [0.7, 1.1, 1.7]
    ])
    J_inv = np.linalg.inv(J)
    D = np.zeros((3, 3))
    
    # --- 2. Load Optimal Gains ---
    filename = Path('OptimalK.spydata')
    try:
        with open(filename, 'rb') as file:
            K_optimal = pickle.load(file)['OptimalK']
    except FileNotFoundError:
        print(f"Error: {filename.name} not found. Please run the LMI optimization first.")
        return

    # --- 3. Highly Efficient Simulation Setup ---
    # Precompute constant matrix operations outside the loop
    T_J_inv = T * J_inv
    T_D = T * D
    
    Omega = np.zeros((3, num_steps))
    U = np.zeros((3, num_steps))
    
    # Prespecified initial condition
    Omega[:, 0] = np.array([-0.1, 0.35, 0.5])

    # --- 4. Fast Closed-Loop Simulation ---
    for k in range(1, num_steps):
        omega_prev = Omega[:, k-1]
        
        # Apply optimal control input if available
        if k-1 < len(K_optimal) and K_optimal[k-1] is not None:
            U[:, k-1] = K_optimal[k-1] @ omega_prev
        else:
            U[:, k-1] = 0
        
        # Vectorized cross-product replaces the S(Omega) matrix multiplication
        J_omega = J @ omega_prev
        cross_term = np.cross(omega_prev, J_omega)
        
        # State update via Forward Euler
        Omega[:, k] = omega_prev - (T_J_inv @ cross_term) - (T_D @ omega_prev) + (T_J_inv @ U[:, k-1])

    # --- 5. Plot Results ---
    plot_trajectories(
        time, Omega, 
        title='Closed-Loop State Trajectories', 
        ylabel='States (Angular Velocity)', 
        labels=['State 1', 'State 2', 'State 3']
    )
    
    plot_trajectories(
        time, U, 
        title='Control Input Trajectories', 
        ylabel='Inputs (Torque)', 
        labels=['Tau 1', 'Tau 2', 'Tau 3']
    )

if __name__ == '__main__':
    main()