# MTSP Solver

This project provides a framework for multi-agent route planning and target assignment using clustering and optimization techniques. It leverages OR-Tools for solving routing problems and includes custom logic for agent clustering, target assignment, and collision-free path planning.

## Features

- Multi-agent environment simulation (`gridworld_env`)
- Clustering-based target assignment
- Open-ended Vehicle Routing Problem (VRP) solver using OR-Tools
- Collision-free path planning with CBS (Conflict-Based Search)
- Data buffers for supervised learning
- Logging to file and console for debugging and analysis

## Folder Structure

- `clustering_solver.py`: Main logic for clustering, target assignment, and trajectory generation
- `additional_functions.py`: Utility functions (e.g., distance calculations)
- `clustering_route_planner.py`: Clustering and TSP/VRP solver
- `CBSsolver.py`: Conflict-Based Search path planner
- `grid_world_env.py`: Environment simulation
- `buffers.py`: Data buffer for storing trajectories
- `params.py`: Parameter configuration

## Requirements

- Python 3.7+
- [OR-Tools](https://developers.google.com/optimization)
- NumPy
