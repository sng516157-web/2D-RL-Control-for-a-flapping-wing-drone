# 2D Flapping-Wing MAV + RL Control (PyTorch + Pygame)

This project is a lightweight sandbox for experimenting with reinforcement learning on a **2D flapping-wing micro air vehicle (MAV)**. It includes:

- A simplified flapping-wing physics model  
- A PyTorch REINFORCE RL controller  
- A **moveable target set-point** using WASD keys  
- Pygame visualisation with trajectory trails and metrics  

It focuses on rapid iteration and intuition — not full aero fidelity.

---

## 1️⃣ Flapping-Wing Physics

State vector:
```
(x, y, θ, vx, vy, ω, base_f, freq_delta)
```

Wing frequencies:
```
f_L = base_f + freq_delta
f_R = base_f - freq_delta
```

Thrust (simple linear model):
```
T_L = k * f_L
T_R = k * f_R
T_total = T_L + T_R
```

Differential torque:
```
τ = arm * (T_R - T_L)
```

Unsteady sinusoidal + random noise torque simulates flutter.  
Euler integration performs state updates.

---

## 2️⃣ RL Formulation

### Actions
```
u = (u_thrust, u_yaw) ∈ [-1, 1]²
```
Mapped to frequency control:
```
base_f    += THRUST_GAIN * u_thrust
freq_delta += YAW_GAIN   * u_yaw
```

### Moveable set-point (keyboard)
| Key | Action |
|-----|--------|
| W/S | Target moves up/down |
| A/D | Target moves left/right |

Tracking error inside state:
```
ex = x - x_target
ey = y - y_target
```
Goal: converge `ex → 0`, `ey → 0` while staying upright.

---

## 3️⃣ Reward Function

Encourages stability + accuracy:
```
reward = -(
  5.0 * θ² +
  2.0 * ey² +
  1.0 * ex² +
  0.1 * (vx² + vy²) +
  0.1 * ω²
)
```

Episode termination if:
- |θ| > 90°
- Too far from target (safety bounds)
- Max steps reached

---

## 4️⃣ RL Algorithm: REINFORCE

Monte-Carlo policy gradient:
```
loss = -(log_probs * returns_normalised).mean()
```

Advantages:
- Minimal implementation
- Good for experimentation

Limitations:
- High variance → unstable
- Training results vary per run

Planned upgrade: **Advantage Actor–Critic (A2C)**

---

## 5️⃣ Visualisation & Runtime Metrics

On-screen visualisation includes:
- MAV body + wings
- **Blue line**: MAV trajectory history  
- **Green line**: Target trajectory
- HUD with:
  - Position & tracking error
  - Orientation & angular rate
  - Frequencies (base_f, freq_delta)
  - Reward
  - Control effort metrics:
    ```
    E_action = u_thrust² + u_yaw²
    E_freq   = base_f² + freq_delta²
    ```

---

## 6️⃣ Installation

```
pip install pygame torch numpy matplotlib
```

---

## 7️⃣ Usage

### Train the policy
Set in script:
```
MODE = "train"
```
Run:
```
python flapper_rl.py
```
Weights saved:
```
flapper_policy.pt
```

### Run simulator
```
MODE = "play"
python flapper_rl.py
```

Controls:
| Key | Action |
|------|--------|
| W/S | Move target up/down |
| A/D | Move target left/right |
| ESC | Quit |

---

## 8️⃣ Files

| File | Description |
|------|-------------|
| flapper_rl.py | Simulator + RL + Visualisation |
| flapper_policy.pt | Trained policy network |
| README.md | This document |

---

## 9️⃣ Future Enhancements

- Improved aerodynamic model (**T ∝ f²**)
- Move to **A2C / PPO**
- Enhanced reward shaping
- Learning curve plots
- Possible 3-D extension

---

### License
MIT — free to use, learn from, and build on.
