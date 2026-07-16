# Capacitated Vehicle Routing Problem (CVRP): Deep RL vs. Combinatorial Solvers

This repository contains a comparative study benchmarking Deep Reinforcement Learning against traditional mathematical solvers for finding optimal vehicle routes in a multi-agent system. 

## Project Overview

Unoptimized multi-vehicle routing causes uneven load distribution, delayed deliveries, and inflated logistics costs. Standard neural networks often struggle with the massive **combinatorial explosion** inherent in strict routing scenarios, frequently getting trapped in **local minima**. 

This project aims to automate and dynamically assign delivery nodes by evaluating the computational efficiency and accuracy of neural network routing against deterministic solvers.

### Key Findings
The **Guided Local Search** algorithm (via Google OR-Tools) bypassed the limitations of the PPO Reinforcement Learning agent, consistently achieving mathematically perfect load distributions across the fleet in significantly less time.

## Repository Structure

```text
vrp_models/
│
├── envs/
│   └── custom_mvrp_env.py      # Custom Gymnasium environment with capacity/distance constraints
│
├── models/
│   └── rl_agent_training.py    # PPO agent training and testing pipeline
│
├── or_tools_1.ipynb            # Google OR-Tools Guided Local Search implementation 
├── or_tools_2.ipynb            # Advanced OR-Tools configurations and testing
├── requirements.txt            # Python dependencies
└── README.md
```
## Methodology

This comparative analysis is split into two distinct testing environments to benchmark performance, accuracy, and computational efficiency:

1. **Deep Reinforcement Learning (PPO):** Developed a custom `gymnasium` environment tracking dynamic spatial arrays and load constraints.
   * Trained a Proximal Policy Optimization (PPO) agent using `stable-baselines3` to navigate the penalty structures associated with invalid routing parameters (e.g., exceeding vehicle capacity or revisiting nodes).

2. **Deterministic Logistics Routing (OR-Tools):**
   * Engineered a mathematical routing solver using the Google OR-Tools library.
   * Applied strict distance matrix and demand array callbacks to formulate exact capacity constraints, utilizing a Guided Local Search metaheuristic to pinpoint the mathematically optimal route.

## Installation & Usage

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/shirsh008/vrp_models.git
cd vrp_models
pip install -r requirements.txt
