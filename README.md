# Data-Driven Optimal Control for Rigid Body Dynamics

This repository provides a Python implementation for data-driven optimal control utilizing Semidefinite Programming (SDP). It is designed to compute optimal control laws directly from collected input-state data, circumventing the need for explicit system identification. 

While this specific implementation demonstrates the approach on rigid body rotational dynamics, the proposed data-driven computation is applicable to non-affine input nonlinear systems as long as Assumptions hold.

## Overview
This methodology relies on solving a finite-horizon linear quadratic regulation (LQR) problem where the system dynamics are assumed to be unknown. Information about the system is instead extracted from a finite set of input-state data, requiring the injected input to be persistently exciting.

By framing the LQR problem as a one-shot semidefinite program, the control law is computed entirely from data. This avoids the traditional "indirect" approach, where a model is first determined from data and the control law is subsequently designed based on that model.

## Repository Structure
* `DataGeneration.Rigid.py`: Generates the necessary state and input trajectories by simulating multiple experiments with varying initial conditions and exponentially decaying control inputs. 
* `LMI.Rigid.py`: Processes the sequence of historical inputs and states to formulate and solve the Linear Matrix Inequalities (LMIs) using CVXPY. It recovers the optimal time-varying state feedback gains ($K^*$).
* `StateFeedback.Rigid.py`: Simulates the closed-loop physical dynamics of the system over time, demonstrating the stabilization achieved by the computed optimal feedback gains.

## Dependencies
* `numpy`
* `matplotlib`
* `cvxpy`

## Reference
The LMI formulation and core data-driven methodology implemented in this repository are based on the work by Rotulo, De Persis, and Tesi:

> Rotulo, M., De Persis, C., & Tesi, P. (2020). Data-driven Linear Quadratic Regulation via Semidefinite Programming. *IFAC PapersOnLine*, 53(2), 3995-4000. https://doi.org/10.1016/j.ifacol.2020.12.2264
