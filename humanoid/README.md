<div align="center">

# Humanoid — PPO with PyTorch

**Teaching a 3D bipedal robot to walk using Proximal Policy Optimization**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-orange?logo=openai&logoColor=white)](https://gymnasium.farama.org/environments/mujoco/humanoid/)

</div>

---

## Overview

A from-scratch implementation of **PPO-Clip** with a **Gaussian MLP actor-critic** trained on the MuJoCo Humanoid environment from [Gymnasium](https://gymnasium.farama.org/environments/mujoco/humanoid/). The 3D bipedal robot must learn to walk forward as fast as possible without falling.

---

## The Environment

A 3D humanoid with a torso, two legs (thigh → shin → foot), and two arms (upper arm → forearm). Hip tendons connect to the knees for stability.

| Property | Detail |
|----------|--------|
| **Observation** | `Box(-∞, ∞, (348,), float64)` — positions, velocities, inertias, forces |
| **Action** | `Box(-0.4, 0.4, (17,), float32)` — joint torques |
| **Episode length** | 1000 timesteps |
| **Termination** | Torso height outside `[1.0, 2.0]` m (falls over) |

```
Reward = healthy_reward (5.0)
       + forward_reward (1.25 × Δx/Δt)
       − ctrl_cost     (0.1  × ‖action‖²)
       − contact_cost  (5e-7 × ‖contact_forces‖²)
```

---

## Body & Joints

```
         [head/torso]
        /            \
  [r_arm]            [l_arm]
  [r_fore]           [l_fore]
         |
     [abdomen]
    /         \
[r_thigh]  [l_thigh]
[r_shin]   [l_shin]
[r_foot]   [l_foot]
```

17 controllable hinge joints: 3 abdomen + 4×hip + 2×knee + 4×shoulder + 2×elbow

---

## Algorithm — PPO-Clip with MLP

```
Observation (348,)
      │
 ┌────▼──────────────────┐
 │   MLP  256 → 256      │  shared feature trunk
 └────┬──────────┬───────┘
      │          │
 ┌────▼──┐  ┌────▼──┐
 │ Actor │  │Critic │
 │ μ, σ  │  │ V(s)  │
 └────┬──┘  └───────┘
      │
 Normal(μ, σ) → tanh → scale → action ∈ [-0.4, 0.4]^17
```

| Component | Detail |
|-----------|--------|
| **MLP** | 2 hidden layers (256 units, Tanh activation) |
| **Actor** | Gaussian policy — outputs mean + log_std per joint |
| **Critic** | State-value estimate V(s) |
| **Entropy bonus** | Encourages joint exploration |
| **Gradient clipping** | Norm ≤ 0.5 |

---

## Project Structure

```
humanoid/
├── humanoid.ipynb      # Full PPO + MLP implementation
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

Then open `humanoid.ipynb` in Jupyter.

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.9+ | |
| PyTorch | 2.x | Neural networks |
| Gymnasium[mujoco] | 1.x | Humanoid environment |
| MuJoCo | 3.x | Physics simulation |
| Numpy | 2.x | Array ops |
| Matplotlib | any | Plots |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| `rollout_length` | 2048 | Steps per epoch (larger = more stable for MuJoCo) |
| `k_epochs` | 10 | PPO gradient steps per buffer |
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing factor |
| `epsilon` | 0.2 | PPO clip range |
| `entropy_coef` | 0.0 | No entropy bonus (dense reward) |
| `n_updates` | 1000 | Total training epochs |
