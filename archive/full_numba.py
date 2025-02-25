import math
import random
import time

import numpy as np
import pygame
from numba import njit

# ---------------------------
# Simulation Parameters
# ---------------------------
NX, NY = 200, 100  # Grid size
cell_size = 4  # Pixels per cell
steps_per_frame = 5  # LBM steps per animation frame

# D2Q9 Lattice Directions and Weights
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
weights = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float32,
)

# Fluid and Thermal Parameters
viscosity = 0.04
tau = viscosity * 3 + 0.5
damping = 0.02
# dpdt = 300.0
dpdt = 30000.0
k_const = 0.0007
vc = 1.0
d_max = 0.001
seasonal_effect = 0.0  # Change this dynamically for seasons

# ---------------------------
# Data Arrays
# ---------------------------
f = np.ones((NY, NX, 9), dtype=np.float32) * weights  # LBM distribution functions
f_temp = np.empty_like(f)
base_map = np.zeros((NY, NX), dtype=np.float32)
edit_map = np.zeros((NY, NX), dtype=np.float32)
seasonal_map = np.zeros((NY, NX), dtype=np.float32)
air_temp = np.zeros((NY, NX), dtype=np.float32)
land_map = np.zeros((NY, NX), dtype=np.uint8)  # 0=Water, 1=Land
particles = []
hurricanes = []
next_hurricane_id = 1


# ---------------------------
# Numba-Accelerated LBM & Thermal Update
# ---------------------------
@njit
def step_lbm(
    f,
    f_temp,
    air_temp,
    base_map,
    edit_map,
    seasonal_map,
    seasonal_effect,
    tau,
    damping,
    dpdt,
    d_max,
    k_const,
    vc,
    ex,
    ey,
    weights,
    NX,
    NY,
    dt_step,
):
    feq = np.empty((NY, NX, 9), dtype=np.float32)
    u, v = np.zeros((NY, NX), dtype=np.float32), np.zeros((NY, NX), dtype=np.float32)

    # Compute macroscopic variables
    for j in range(NY):
        for i in range(NX):
            rho = f[j, i, :].sum()
            mom_x = np.sum(f[j, i, :] * ex)
            mom_y = np.sum(f[j, i, :] * ey)
            u[j, i], v[j, i] = mom_x / rho, mom_y / rho
            u_sq = u[j, i] ** 2 + v[j, i] ** 2
            for k in range(9):
                e_dot_u = ex[k] * u[j, i] + ey[k] * v[j, i]
                feq[j, i, k] = (
                    weights[k] * rho * (1 + 3 * e_dot_u + 4.5 * e_dot_u**2 - 1.5 * u_sq)
                )

    # Compute temperature forcing
    S_array = np.zeros((NY, NX), dtype=np.float32)
    for j in range(NY):
        for i in range(NX):
            t_ground = (
                base_map[j, i] + edit_map[j, i] + seasonal_effect * seasonal_map[j, i]
            )
            t_diff = k_const * (t_ground - air_temp[j, i])
            S_val = t_diff * dpdt
            if S_val > d_max:
                S_val = d_max
            elif S_val < -d_max:
                S_val = -d_max
            S_array[j, i] = S_val

    # Collision and Streaming
    for j in range(NY):
        for i in range(NX):
            for k in range(9):
                f[j, i, k] = (
                    f[j, i, k]
                    - (f[j, i, k] - feq[j, i, k]) / tau
                    + S_array[j, i] * weights[k]
                )

    for j in range(NY):
        for i in range(NX):
            for k in range(9):
                ni, nj = (i + ex[k]) % NX, (j + ey[k]) % NY
                f_temp[nj, ni, k] = f[j, i, k]

    f[:, :, :] = f_temp
    air_temp[:, :] += S_array


# ---------------------------
# Hurricanes
# ---------------------------
def update_hurricanes(dt):
    global next_hurricane_id
    if random.random() < 0.01:
        x, y = random.randint(0, NX - 1), random.randint(0, NY - 1)
        if air_temp[y, x] > 0 and land_map[y, x] == 0:
            hurricanes.append({"id": next_hurricane_id, "x": x, "y": y, "radius": 5.0})
            next_hurricane_id += 1

    for h in hurricanes[:]:
        i, j = int(h["x"]), int(h["y"])
        if land_map[j, i] == 0:
            h["radius"] += 0.01 * air_temp[j, i]
        else:
            h["radius"] -= 0.01
        if h["radius"] < 2.5:
            hurricanes.remove(h)


# ---------------------------
# Particles (Flow Visualization)
# ---------------------------
def spawn_particle():
    particles.append(
        {
            "x": random.uniform(0, NX * cell_size),
            "y": random.uniform(0, NY * cell_size),
            "trail": [],
        }
    )


def update_particles(dt):
    for p in particles[:]:
        gx, gy = int(p["x"] / cell_size), int(p["y"] / cell_size)
        if 0 <= gx < NX and 0 <= gy < NY:
            ux = np.sum(f[gy, gx, :] * ex) / np.sum(f[gy, gx, :])
            uy = np.sum(f[gy, gx, :] * ey) / np.sum(f[gy, gx, :])
            p["x"] += ux * dt * 200
            p["y"] += uy * dt * 200
            if not (0 <= p["x"] < NX * cell_size and 0 <= p["y"] < NY * cell_size):
                particles.remove(p)
    if random.random() < 0.5:
        spawn_particle()


# ---------------------------
# Pygame Setup
# ---------------------------
pygame.init()
screen = pygame.display.set_mode((NX * cell_size, NY * cell_size))
pygame.display.set_caption("LBM Fluid Simulation with Numba")
clock = pygame.time.Clock()

running = True
paused = False
last_time = time.time()

# ---------------------------
# Main Simulation Loop
# ---------------------------
while running:
    dt = time.time() - last_time
    last_time = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            paused = not paused

    if not paused:
        for _ in range(steps_per_frame):
            step_lbm(
                f,
                f_temp,
                air_temp,
                base_map,
                edit_map,
                seasonal_map,
                seasonal_effect,
                tau,
                damping,
                dpdt,
                d_max,
                k_const,
                vc,
                ex,
                ey,
                weights,
                NX,
                NY,
                dt,
            )
        update_hurricanes(dt)
        update_particles(dt)

    # Drawing
    for j in range(NY):
        for i in range(NX):
            col = int(((base_map[j, i] + edit_map[j, i]) * 255) % 256)
            pygame.draw.rect(
                screen,
                (col, col, col),
                (i * cell_size, j * cell_size, cell_size, cell_size),
            )

    for h in hurricanes:
        pygame.draw.circle(
            screen,
            (255, 0, 0),
            (int(h["x"] * cell_size), int(h["y"] * cell_size)),
            int(h["radius"] * cell_size),
            1,
        )

    for p in particles:
        pygame.draw.circle(screen, (0, 255, 0), (int(p["x"]), int(p["y"])), 1)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
