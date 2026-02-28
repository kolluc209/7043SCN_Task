# 🧠 Reinforcement Learning in Chef's Hat Gym

## Sparse / Delayed Reward Variant (ID mod 7 = 2)

------------------------------------------------------------------------

## 1. Overview

This project investigates reinforcement learning in the Chef's Hat Gym
multi-agent card game environment.

**Student ID mod 7 = 2 → Sparse / Delayed Reward Variant**

The objective is to analyse how delayed match-level rewards affect
learning dynamics and evaluate reward-shaping strategies as alternative
credit-assignment mechanisms.

Chef's Hat exhibits: - Sequential decision-making\
- Large discrete action space (200 actions)\
- Sparse terminal rewards\
- Multi-agent non-stationarity

The project evaluates whether reward shaping improves learning compared
to a baseline PPO agent trained under sparse reward conditions.

------------------------------------------------------------------------

## 2. Environment Details

Environment: Chef's Hat Gym\
GitHub: https://github.com/pablovin/ChefsHatGYM\
API: OpenAI Gym compatible

### Observation Space (after preprocessing)

Box(0.0, 13.0, (28,), float32)

-   17-dimensional hand vector\
-   11-dimensional board vector\
    Total: 28 continuous features

### Action Space

Discrete(200)

Each action corresponds to a card-play configuration or pass.

------------------------------------------------------------------------

## 3. RL Algorithm

Algorithm: Proximal Policy Optimization (PPO)\
Library: Stable-Baselines3

Configuration: - Learning rate: 3e-4\
- Gamma: 0.99\
- Batch size: 64\
- n_steps: 2048\
- Total timesteps: 20,000\
- Device: CPU

------------------------------------------------------------------------

## 4. Experiments

### Baseline (Sparse Reward Only)

  Metric       Value
  ------------ --------
  Avg Reward   -73.68
  Win Rate     0.00
  Min Reward   -90
  Max Reward   -58

Sparse terminal rewards caused severe credit assignment problems.

------------------------------------------------------------------------

### Reward Shaping v1 (Hand Reduction)

+0.5 reward for reducing hand size.

Training reward improved, but evaluation win rate remained 0.00.\
Agent optimised proxy behaviour instead of winning.

------------------------------------------------------------------------

### Reward Shaping v2 (Terminal Scaling)

+10 for winning\
-10 for losing

Resulted in unstable optimisation and degraded training reward (\~ -82).

------------------------------------------------------------------------

## 5. Comparative Summary

  Model       Avg Reward   Win Rate   Behaviour
  ----------- ------------ ---------- --------------------
  Baseline    -73.68       0.00       Sparse stagnation
  Shaped v1   -76.00       0.00       Proxy optimisation
  Shaped v2   -82.00       0.00       Reward instability

------------------------------------------------------------------------

## 6. Key Insight

This project demonstrates: - The difficulty of delayed reward
environments\
- Limitations of naive reward shaping\
- Sensitivity of PPO to reward scaling\
- Importance of reward alignment with final objective

------------------------------------------------------------------------

## 7. How to Run

### Create Virtual Environment

py -3.10 -m venv rl_env\
rl_env`\Scripts`{=tex}`\activate  `{=tex}

### Install Dependencies

pip install numpy==1.23.5\
pip install gym==0.26.2\
pip install stable-baselines3\[extra\]\
pip install matplotlib seaborn pandas\
pip install git+https://github.com/pablovin/ChefsHatGYM.git

### Train

python train_baseline.py\
python train_shaped.py

### Evaluate

python evaluate_agent.py

------------------------------------------------------------------------

End of README.
