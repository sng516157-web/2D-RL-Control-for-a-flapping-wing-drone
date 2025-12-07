"""
flapper_rl.py

2D flapping wing MAV with PyTorch control.

Modes:
- MODE = "train": runs a headless RL loop and saves flapper_policy.pt
- MODE = "play" : loads flapper_policy.pt and runs a interactive pygame visualisation
"""

import math
import random
import pygame
import numpy as np  # for convenient vector ops
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# PHYSICAL PARAMETERS
# =========================
m = 0.05          # mass [kg]
g = 9.81          # gravity [m/s^2]
I = 2e-4          # moment of inertia about CoM [kg m^2]
arm = 0.1         # half-distance between wings [m]
k_thrust = 0.02   # thrust coefficient [N/Hz], T = k_thrust * f

unsteady_rel_amp = 0.01    # relative amplitude of unsteady torque
unsteady_noise_scale = 0.5 # reduced noise level on unsteady torque (tunable)

# Hovering flapping frequency (approx): 2 * k_thrust * f_hover ≈ m * g
BASE_F_HOVER = (m * g) / (2 * k_thrust)

# Control / normalisation scales
BASE_F_SCALE = 100.0       # scale to map base_f into ~[-1,1]
FREQ_DELTA_SCALE = 5.0     # scale to map freq_delta into ~[-1,1]
THRUST_GAIN = 1.0          # Hz per time step from u_thrust in [-1,1]
YAW_GAIN = 0.2             # Hz per time step from u_yaw in [-1,1]

# Bounds for control variables
BASE_F_MIN, BASE_F_MAX = 0.0, 100.0
FREQ_DELTA_MIN, FREQ_DELTA_MAX = -5.0, 5.0

# Pack parameters for convenience
PARAMS = {
    "m": m,
    "g": g,
    "I": I,
    "arm": arm,
    "k_thrust": k_thrust,
    "unsteady_rel_amp": unsteady_rel_amp,
    "unsteady_noise_scale": unsteady_noise_scale,
}


# =========================
# DYNAMICS FUNCTION
# =========================
def flapper_step(state, base_f, freq_delta, dt, params):
    """
    One physics step of the flapping MAV.

    state   : (x, y, theta, vx, vy, omega, t)
              x, y      -> position [m]
              theta     -> orientation [rad] (0 = pointing up)
              vx, vy    -> velocities [m/s]
              omega     -> angular velocity [rad/s]
              t         -> simulation time [s]
    base_f  : base flapping frequency [Hz] (collective thrust)
    freq_delta : frequency difference [Hz] (yaw control)
    dt      : time step [s]
    params  : dict with physical parameters

    returns next_state, base_f, freq_delta
    """
    x, y, theta, vx, vy, omega, t = state

    m = params["m"]
    g = params["g"]
    I = params["I"]
    arm = params["arm"]
    k_thrust = params["k_thrust"]
    unsteady_rel_amp = params["unsteady_rel_amp"]
    unsteady_noise_scale = params["unsteady_noise_scale"]

    # --- flapping frequencies for left and right wings ---
    f_L = max(0.0, base_f + freq_delta)
    f_R = max(0.0, base_f - freq_delta)

    # Thrusts per wing
    T_L = k_thrust * f_L
    T_R = k_thrust * f_R
    T_total = T_L + T_R

    # --- Forces in world frame ---
    # Orientation: thrust acts along body "up" direction.
    # Choose body "up" = (sin theta, cos theta) in world coordinates.
    Fx = T_total * math.sin(theta)
    Fy = T_total * math.cos(theta) - m * g  # subtract gravity

    # --- Base torque from thrust differential ---
    # Right wing on +arm, left wing on -arm -> tau ~ arm * (T_R - T_L)
    tau = arm * (T_R - T_L)

    # --- Unsteady / flutter torque ---
    f_flap = 0.5 * (f_L + f_R)  # average flapping frequency
    if f_flap > 0.0:
        # Amplitude proportional to total thrust * lever arm
        base_amp = unsteady_rel_amp * arm * T_total
        # Sinusoidal component at flapping frequency
        flutter_core = math.sin(2.0 * math.pi * f_flap * t)
        # Additive Gaussian noise
        flutter_noise = unsteady_noise_scale * random.gauss(0.0, 1.0)
        tau_unsteady = base_amp * (flutter_core + flutter_noise)
    else:
        tau_unsteady = 0.0

    tau_total = tau + tau_unsteady

    # --- Damping and integration (Euler) ---
    lin_damp = 0.05    # linear damping coefficient [N s / m] (tunable)
    ang_damp = 0.001   # angular damping coefficient [N m s / rad] (tunable)

    ax = (Fx - lin_damp * vx) / m
    ay = (Fy - lin_damp * vy) / m
    alpha = (tau_total - ang_damp * omega) / I

    vx += ax * dt
    vy += ay * dt
    omega += alpha * dt

    x += vx * dt
    y += vy * dt
    theta += omega * dt
    t += dt

    # Wrap angle into [-pi, pi] for numerical sanity
    if theta > math.pi:
        theta -= 2 * math.pi
    elif theta < -math.pi:
        theta += 2 * math.pi

    next_state = (x, y, theta, vx, vy, omega, t)
    return next_state, base_f, freq_delta


# =========================
# RL ENVIRONMENT
# =========================
class FlapperEnv:
    """
    Minimal RL environment for the flapping MAV.
    The agent controls:
      - u_thrust in [-1,1] -> changes base_f (collective thrust)
      - u_yaw    in [-1,1] -> changes freq_delta (differential thrust)

    Goal: keep y near y_target, theta near 0, and x near 0.
    """

    def __init__(self, params, dt=0.02, y_target=0.0):
        self.params = params
        self.dt = dt
        self.max_steps = 5000
        self.step_count = 0

        # New: position setpoint (x_target, y_target)
        self.x_target = 0.0
        self.y_target = y_target

        self.cum_action_effort = 0.0
        self.cum_freq_effort = 0.0

        self.reset()

    def reset(self):
        """Reset state to a near-hover condition with small perturbations."""
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.05 * (2 * random.random() - 1)  # small tilt
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.t = 0.0

        self.base_f = BASE_F_HOVER
        self.freq_delta = 0.0

        self.step_count = 0

        # Reset control effort accumulators
        self.cum_action_effort = 0.0
        self.cum_freq_effort = 0.0

        return self._get_state()

    def _get_state(self):
        """
        Return state as a list for the policy:

        [ex, ey, theta, vx, vy, omega, base_f_norm, freq_delta_norm]

        where ex = x - x_target, ey = y - y_target
        """
        ex = self.x - self.x_target
        ey = self.y - self.y_target
        base_f_norm = self.base_f / BASE_F_SCALE
        freq_delta_norm = self.freq_delta / FREQ_DELTA_SCALE
        return [
            ex, ey, self.theta,
            self.vx, self.vy, self.omega,
            base_f_norm, freq_delta_norm
        ]

    def step(self, action):
        """
        Step the env with an action = [u_thrust, u_yaw] in [-1,1].

        Returns: next_state, reward, done, info
        """
        self.step_count += 1
        u_thrust, u_yaw = action

        # --- CONTROL EFFORT METRICS ---
        action_effort = u_thrust**2 + u_yaw**2
        self.cum_action_effort += action_effort

        # Apply control to flapping frequencies
        self.base_f += THRUST_GAIN * float(u_thrust)
        self.freq_delta += YAW_GAIN * float(u_yaw)

        # Clamp control variables
        self.base_f = max(BASE_F_MIN, min(BASE_F_MAX, self.base_f))
        self.freq_delta = max(FREQ_DELTA_MIN, min(FREQ_DELTA_MAX, self.freq_delta))

        # Per-step "frequency effort" (proxy for power usage)
        freq_effort = self.base_f**2 + self.freq_delta**2
        self.cum_freq_effort += freq_effort

        # Integrate physics
        state = (self.x, self.y, self.theta, self.vx, self.vy, self.omega, self.t)
        (self.x, self.y, self.theta,
         self.vx, self.vy, self.omega, self.t), self.base_f, self.freq_delta = flapper_step(
            state, self.base_f, self.freq_delta, self.dt, self.params
        )

        # Tracking errors relative to target
        ex = self.x - self.x_target
        ey = self.y - self.y_target

        # Reward: penalise tilt, tracking errors, velocities
        theta_err = self.theta
        y_err = ey
        x_err = ex

        reward = -(
            5.0 * theta_err**2 +
            2.0 * y_err**2 +
            1.0 * x_err**2 +
            0.1 * (self.vx**2 + self.vy**2) +
            0.1 * self.omega**2
        )

        # Termination conditions
        done = False

        if abs(self.theta) > math.pi / 2:       # flipped more than 90 deg
            done = True
        if abs(ey) > 1.0:                       # too far vertically from target
            done = True
        if abs(ex) > 1.0:                       # too far sideways from target
            done = True
        if self.step_count >= self.max_steps:
            done = True

        # --- RUNTIME METRICS IN INFO ---
        dist_to_target = math.sqrt(ex**2 + ey**2)
        speed = math.sqrt(self.vx**2 + self.vy**2)

        info = {
            "ex": ex,
            "ey": ey,
            "dist_to_target": dist_to_target,
            "theta": self.theta,
            "abs_theta": abs(self.theta),
            "omega": self.omega,
            "speed": speed,
            "action_effort": action_effort,
            "freq_effort": freq_effort,
            "cum_action_effort": self.cum_action_effort,
            "cum_freq_effort": self.cum_freq_effort,
        }

        return self._get_state(), reward, done, info, {}


# =========================
# POLICY NETWORK
# =========================
class PolicyNet(nn.Module):
    """
    Simple MLP policy: state -> action in [-1,1]^2
    """

    def __init__(self, state_dim=8, hidden=64, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()   # outputs in [-1,1]
        )

    def forward(self, x):
        return self.net(x)


# =========================
# REINFORCE TRAINING
# =========================
def rollout_episode(env, policy, device="cpu", gamma=0.99, std=0.3):
    """
    Run one episode in the env using a stochastic policy.

    Returns:
        log_probs: tensor [T]
        returns : tensor [T]
        ep_return: float (sum of rewards)
        ep_metrics: dict with summary stats for this episode
    """
    state = env.reset()
    log_probs = []
    rewards = []

    # For metrics
    dists = []
    abs_thetas = []
    action_efforts = []
    freq_efforts = []

    done = False
    while not done:
        # state -> tensor [1, state_dim]
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # Policy outputs mean actions in [-1,1]
        mean_action = policy(s)  # [1,2]
        dist = torch.distributions.Normal(mean_action, std)
        action = dist.sample()   # [1,2]
        log_prob = dist.log_prob(action).sum(dim=-1)  # [1]

        # Clip action to [-1,1] before using in env
        a_clipped = torch.tanh(action)[0].cpu().numpy().tolist()
        next_state, reward, done, info, _ = env.step(a_clipped)

        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state

        # Collect runtime metrics from info
        dists.append(info["dist_to_target"])
        abs_thetas.append(info["abs_theta"])
        action_efforts.append(info["action_effort"])
        freq_efforts.append(info["freq_effort"])

    # Compute discounted returns
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()

    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    log_probs = torch.stack(log_probs).squeeze(-1)  # [T]

    ep_return = float(sum(rewards))
    ep_len = len(rewards)

    # Episode-level summary metrics
    if ep_len > 0:
        mean_dist = float(np.mean(dists))
        mean_abs_theta = float(np.mean(abs_thetas))
        mean_action_effort = float(np.mean(action_efforts))
        mean_freq_effort = float(np.mean(freq_efforts))
    else:
        mean_dist = mean_abs_theta = mean_action_effort = mean_freq_effort = 0.0

    ep_metrics = {
        "length": ep_len,
        "return": ep_return,
        "mean_dist": mean_dist,
        "mean_abs_theta": mean_abs_theta,
        "mean_action_effort": mean_action_effort,
        "mean_freq_effort": mean_freq_effort,
    }

    return log_probs, returns, ep_return, ep_metrics


def train_policy(num_episodes=500, lr=1e-3, gamma=0.99, std=0.3):
    """
    Train the policy with vanilla REINFORCE.
    """
    device = "cpu"
    env = FlapperEnv(PARAMS, dt=0.02, y_target=0.0)
    policy = PolicyNet().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    metrics_history = []

    for ep in range(num_episodes):
        log_probs, returns, ep_return, ep_metrics = rollout_episode(env, policy,
                                                        device=device,
                                                        gamma=gamma,
                                                        std=std)
        # Normalise returns for variance reduction
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        loss = -(log_probs * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Attach episode index and store metrics
        ep_metrics["episode"] = ep
        metrics_history.append(ep_metrics)

        if (ep + 1) % 20 == 0:
            print(
                f"Episode {ep+1:4d}/{num_episodes}, "
                f"return={ep_metrics['return']:8.3f}, "
                f"len={ep_metrics['length']:4d}, "
                f"mean_dist={ep_metrics['mean_dist']:.3f}, "
                f"mean_action_effort={ep_metrics['mean_action_effort']:.3f}, "
                f"loss={loss.item():.4f}"
            )

    torch.save(policy.state_dict(), "flapper_policy.pt")
    print("Saved trained policy to flapper_policy.pt")

    # Plot learning curves at the end
    plot_training_curves(metrics_history)

    return policy

def plot_training_curves(metrics_history):
    """
    Plot basic learning curves:
      - episode return
      - mean distance to target
      - mean action effort
    """
    if not metrics_history:
        print("No metrics to plot.")
        return

    episodes = [m["episode"] for m in metrics_history]
    returns = [m["return"] for m in metrics_history]
    mean_dists = [m["mean_dist"] for m in metrics_history]
    mean_action_efforts = [m["mean_action_effort"] for m in metrics_history]

    plt.figure(figsize=(12, 8))

    # 1) Return
    plt.subplot(3, 1, 1)
    plt.plot(episodes, returns, label="Episode return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()

    # 2) Mean distance to target
    plt.subplot(3, 1, 2)
    plt.plot(episodes, mean_dists, label="Mean dist to target", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Mean dist [m]")
    plt.grid(True)
    plt.legend()

    # 3) Mean action effort
    plt.subplot(3, 1, 3)
    plt.plot(episodes, mean_action_efforts, label="Mean action effort", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Mean u^2")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# =========================
# PYGAME VISUALISATION (SETPOINT CONTROL WITH WASD)
# =========================
def run_pygame(policy=None):
    """
    Visualise the flapping MAV in pygame.

    If `policy` is provided, it controls the MAV.
    WASD control the *setpoint*, not direct thrust/yaw:
        W / S : move target up / down
        A / D : move target left / right

    Press 'O' to toggle an automatic oval setpoint routine.
    """
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Flapping MAV RL (setpoint control with WASD + oval routine)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    SCALE = 200.0
    origin_x = WIDTH // 2
    origin_y = HEIGHT // 2

    # Create environment with same dynamics
    env = FlapperEnv(PARAMS, dt=0.02, y_target=0.0)
    state = env.reset()

    # Setpoint step (metres per key press)
    TARGET_STEP = 0.05

    # Tail tracking
    traj_history = []
    target_history = []
    MAX_HISTORY = 200

    # ===== OVAL ROUTINE STATE (NEW) =====
    routine_active = False
    routine_time = 0.0
    routine_center_x = 0.0
    routine_center_y = 0.0
    oval_a = 0.9      # semi-major axis in x [m]
    oval_b = 0.5      # semi-minor axis in y [m]
    oval_omega = 0.4  # rad/s (speed around the oval)

    running = True
    while running:
        dt_ms = clock.tick(60)
        _dt = dt_ms / 1000.0  # not used in physics; env has its own dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Toggle oval routine (NEW)
                elif event.key == pygame.K_o:
                    routine_active = not routine_active
                    if routine_active:
                        # Start oval around current target
                        routine_time = 0.0
                        routine_center_x = env.x_target
                        routine_center_y = env.y_target

                # Manual setpoint moves only when routine is OFF
                elif not routine_active:
                    # Move setpoint with WASD
                    if event.key == pygame.K_w:
                        env.y_target += TARGET_STEP
                    elif event.key == pygame.K_s:
                        env.y_target -= TARGET_STEP
                    elif event.key == pygame.K_d:
                        env.x_target += TARGET_STEP
                    elif event.key == pygame.K_a:
                        env.x_target -= TARGET_STEP

        # ===== UPDATE OVAL ROUTINE TARGET (NEW) =====
        if routine_active:
            routine_time += env.dt  # use env's physics timestep
            env.x_target = routine_center_x + oval_a * math.cos(oval_omega * routine_time)
            env.y_target = routine_center_y + oval_b * math.sin(oval_omega * routine_time)

        # === Control ===
        if policy is not None:
            # PyTorch policy control
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean_action = policy(s)
            action = torch.tanh(mean_action)[0].cpu().numpy().tolist()
        else:
            # No policy: keep zero action (hover-ish around target)
            action = [0.0, 0.0]

        state, reward, done, info,_ = env.step(action)
        ex, ey, theta, vx, vy, omega, base_f_norm, freq_delta_norm = state
        x = env.x
        y = env.y

        if done:
            state = env.reset()
            # When we reset, keep the target where it was
            # (no change needed; env.x_target / y_target stay as is)
            traj_history.clear()
            target_history.clear()

        traj_history.append((x, y))
        target_history.append((env.x_target, env.y_target))
        if len(traj_history) > MAX_HISTORY:
            traj_history.pop(0)
        if len(target_history) > MAX_HISTORY:
            target_history.pop(0)

        # === Drawing ===
        screen.fill((30, 30, 30))

        def world_to_screen(wx, wy):
            sx = origin_x + int(wx * SCALE)
            sy = origin_y - int(wy * SCALE)
            return sx, sy

        if len(traj_history) > 1:
            traj_points = [world_to_screen(px, py) for (px, py) in traj_history]
            pygame.draw.lines(screen, (100, 100, 255), False, traj_points, 2)  # actual MAV path

        if len(target_history) > 1:
            target_points = [world_to_screen(px, py) for (px, py) in target_history]
            pygame.draw.lines(screen, (0, 200, 0), False, target_points, 2)    # target path

        # Body parameters for display
        body_len = 0.12
        half_body = body_len / 2.0

        # Body x-axis direction in world coords
        bx = math.cos(theta)
        by = -math.sin(theta)  # so theta=0 points up

        # Body endpoints
        front = (x + bx * half_body, y + by * half_body)
        back = (x - bx * half_body, y - by * half_body)

        front_scr = world_to_screen(*front)
        back_scr = world_to_screen(*back)

        # Draw body
        pygame.draw.line(screen, (200, 200, 200), back_scr, front_scr, 3)

        # Wing positions at +/- arm along body x-axis
        left_wing_pos = (x - bx * arm, y - by * arm)
        right_wing_pos = (x + bx * arm, y + by * arm)
        left_wing_scr = world_to_screen(*left_wing_pos)
        right_wing_scr = world_to_screen(*right_wing_pos)

        pygame.draw.circle(screen, (0, 150, 255), left_wing_scr, 6)
        pygame.draw.circle(screen, (255, 150, 0), right_wing_scr, 6)

        # Draw target as a small cross
        target_scr = world_to_screen(env.x_target, env.y_target)
        pygame.draw.line(screen, (0, 255, 0),
                         (target_scr[0] - 5, target_scr[1]),
                         (target_scr[0] + 5, target_scr[1]), 2)
        pygame.draw.line(screen, (0, 255, 0),
                         (target_scr[0], target_scr[1] - 5),
                         (target_scr[0], target_scr[1] + 5), 2)

        # Text HUD
        base_f = BASE_F_SCALE * base_f_norm
        freq_delta = FREQ_DELTA_SCALE * freq_delta_norm
        dist_to_target = info["dist_to_target"]
        action_effort = info["action_effort"]
        freq_effort = info["freq_effort"]
        cum_action_effort = info["cum_action_effort"]
        cum_freq_effort = info["cum_freq_effort"]
        info_lines = [
            f"x: {x: .2f} m, y: {y: .2f} m",
            f"x_target: {env.x_target: .2f}, y_target: {env.y_target: .2f}",
            f"theta: {math.degrees(theta): .1f} deg",
            f"base_f: {base_f: .1f} Hz, freq_delta: {freq_delta: .2f} Hz",
            f"reward (last step): {reward: .3f}",
            f"oval routine: {'ON' if routine_active else 'OFF'}",
            f"effort_step (action, freq^2): ({action_effort: .3f}, {freq_effort: .1f})",
            f"effort_cum  (action, freq^2): ({cum_action_effort: .1f}, {cum_freq_effort: .1f})",
            "ESC to quit.",
            "W/S: move target up/down, A/D: move target left/right (manual mode).",
            "O: toggle automatic oval setpoint routine.",
            "Blue line: MAV trajectory (last 200 steps).",
            "Green line: target trajectory (last 200 steps)."
        ]
        for i, line in enumerate(info_lines):
            text_surf = font.render(line, True, (220, 220, 220))
            screen.blit(text_surf, (10, 10 + 20 * i))

        pygame.display.flip()

    pygame.quit()



# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Choose mode: "train" or "play"
    MODE = "play"   # set to "play" after training

    if MODE == "train":
        train_policy(num_episodes=1000)
    elif MODE == "play":
        # Load trained policy and visualise
        policy = PolicyNet()
        policy.load_state_dict(torch.load("flapper_policy.pt", map_location="cpu"))
        policy.eval()
        run_pygame(policy=policy)
    else:
        print("Unknown MODE. Use 'train' or 'play'.")
