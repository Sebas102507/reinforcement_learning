<div align="center">

# Frozen Lake — PPO with PyTorch

**Solving `FrozenLake-v1` from scratch using Proximal Policy Optimization**


<img width="872" height="586" alt="Screenshot 2026-05-28 at 10 08 33 PM" src="https://github.com/user-attachments/assets/fb16e96e-f7da-4465-8925-ce415db6c697" />


[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.x-008080?logo=openai&logoColor=white)](https://gymnasium.farama.org/)

</div>

---

## Overview

A clean, from-scratch implementation of **PPO-Clip** (Proximal Policy Optimization) trained on the deterministic 4×4 FrozenLake grid from [Gymnasium](https://gymnasium.farama.org/). The notebook includes a live training window — game on the left, real-time charts on the right.

---

## The Environment

```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

| Symbol | Meaning | Reward |
|:------:|---------|:------:|
| `S` | Start tile | — |
| `F` | Frozen (safe to walk) | 0 |
| `H` | Hole — episode ends | 0 |
| `G` | Goal | +1 |

- **Observation space:** `Discrete(16)` — encoded as `row × 4 + col`
- **Action space:** `Discrete(4)` — `0` Left · `1` Down · `2` Right · `3` Up

---

## Algorithm — PPO-Clip

PPO trains an **actor-critic** network end-to-end using a clipped surrogate objective to keep policy updates stable.

```
┌───────────────────────────────────────┐
│  Collect rollout  (512 steps)         │  ← actor explores environment
│         ↓                             │
│  Compute returns & advantages (GAE)   │  ← critic estimates value
│         ↓                             │
│  K = 4 PPO gradient steps            │  ← clipped ratio prevents big updates
└───────────────────────────────────────┘
         repeat × 300 epochs
```

| Component | Role |
|-----------|------|
| **Actor** | Softmax policy π(a\|s) over 4 actions |
| **Critic** | State-value estimate V(s) |
| **Clipped objective** | Keeps the new policy close to the old one |
| **GAE** | Reduces variance in advantage estimates |

---

## Project Structure

```
frozen-lake/
├── frozen_lake.ipynb   # Full implementation — PPO, live renderer, plots
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Then open `frozen_lake.ipynb` in Jupyter and run all cells.

| Section | What it does |
|---------|-------------|
| §1 Imports | Libraries + device check |
| §2 Networks | Actor & Critic definitions |
| §3 PPO Agent | Training loop, GAE, PPO-clip update |
| §4 Train | Headless training + plots |
| §5 Live Window | Split pygame window with live charts |

---

## Requirements

| Package | Version |
|---------|---------|
| Python | 3.9+ |
| PyTorch | 2.x (MPS / CUDA / CPU) |
| Gymnasium | 1.x |
| Matplotlib | any recent |
| Pygame | 2.x |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| `rollout_length` | 512 | Steps collected per epoch (buffer size) |
| `k_epochs` | 4 | PPO gradient steps per buffer |
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon` | 0.2 | PPO clip range |
| `hidden` | 128 | Hidden layer width |
| `n_updates` | 300 | Total training epochs |
