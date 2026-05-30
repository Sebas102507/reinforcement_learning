<div align="center">

# Car Racing — PPO with PyTorch

**Solving `CarRacing-v3` from scratch using Proximal Policy Optimization + CNN**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.x-008080?logo=openai&logoColor=white)](https://gymnasium.farama.org/environments/box2d/car_racing/)

</div>

---

## Overview

A from-scratch implementation of **PPO-Clip** with a **CNN actor-critic** trained on the top-down car racing environment from [Gymnasium](https://gymnasium.farama.org/environments/box2d/car_racing/). The agent learns to drive solely from raw pixel observations.

---

## The Environment

A top-down racing track generated randomly every episode. The car starts at rest in the center of the road.

| Property | Detail |
|----------|--------|
| **Observation** | `Box(0, 255, (96, 96, 3), uint8)` — RGB pixel frame |
| **Action** | `Box(-1, 1, (3,), float32)` — steering · gas · brake |
| **Reward** | `+1000/N` per track tile visited − `0.1` per frame |
| **Episode end** | All tiles visited, or agent exits the playfield (−100) |

```
Action vector:
  [0] steering  ∈ [-1, +1]   (-1 = full left, +1 = full right)
  [1] gas       ∈ [ 0,  1]
  [2] brake     ∈ [ 0,  1]
```

---

## Algorithm — PPO-Clip with CNN

```
Observation (96×96×3)
       │
  ┌────▼────────────────┐
  │  CNN Feature        │  3 conv layers → 512-d feature vector
  │  Extractor          │
  └────┬───────┬────────┘
       │       │
  ┌────▼──┐ ┌──▼────┐
  │ Actor │ │Critic │
  │ μ, σ  │ │ V(s)  │
  └────┬──┘ └───────┘
       │
  Gaussian(μ, σ) → tanh → action ∈ [-1,1]³
```

| Component | Detail |
|-----------|--------|
| **CNN** | 3 conv layers (32→64→64 filters) + FC 512 |
| **Actor** | Gaussian policy — outputs mean (tanh) + log_std |
| **Critic** | State-value estimate V(s) |
| **Entropy bonus** | Encourages exploration |
| **Gradient clipping** | Norm ≤ 0.5 for stability |

---

## Project Structure

```
car-racing/
├── car_racing.ipynb    # Full PPO + CNN implementation
├── requirements.txt    # Python dependencies
├── .venv/              # Virtual environment
└── README.md
```

---

## Setup

```bash
# Activate the venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Then open `car_racing.ipynb` in Jupyter.

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.9+ | |
| PyTorch | 2.x | Neural networks |
| Gymnasium[box2d] | 1.x | CarRacing environment |
| Numpy | 2.x | Array ops |
| Matplotlib | any | Plots |
| Pygame | 2.x | Rendering |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| `rollout_length` | 512 | Steps per epoch (buffer size) |
| `k_epochs` | 4 | PPO gradient steps per buffer |
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon` | 0.2 | PPO clip range |
| `entropy_coef` | 0.01 | Entropy bonus weight |
| `n_updates` | 500 | Total training epochs |
