import math
import random
import sys

import numba
import numpy as np
import pygame

# Constants
NX, NY = 200, 100
CELL_SIZE = 4
STEPS_PER_FRAME = 1
FPS = 30
DT = 0.0001

# D2Q9 lattice directions and weights
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
weights = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
cs2 = 1 / 3

# Initialize LBM arrays
f = np.ones((NX, NY, 9)) * weights[np.newaxis, np.newaxis, :]
f_temp = np.zeros_like(f)

# Initialize ground temperature maps
base_map = np.zeros((NX, NY))
edit_map = np.zeros((NX, NY))
seasonal_map = np.zeros((NX, NY))

# Initialize air temperature
air_temp = np.zeros((NX, NY))

# Initialize particles
particles = []
particle_lifetime = 20
particle_spawn_accumulator = 0

# Simulation parameters
viscosity = 0.01
tau = viscosity * 3 + 0.5
damping = 0
noise_amplitude = 0.1
noise_scale = 1.0
tail_length = 100.0
spawn_rate = 50
seasonal_blend = 0.0
D_const = 0.0
k_const = 0.005
dpdt = 300
vc = 1.0
d_max = 0.01

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((NX * CELL_SIZE, NY * CELL_SIZE))
clock = pygame.time.Clock()


import math
import random


def perlin_noise(x, y, octaves=4, persistence=0.5):
    """
    Generates Perlin noise with multiple octaves.

    Parameters:
        x, y (float): Coordinates for noise generation.
        octaves (int): Number of noise layers (octaves).
        persistence (float): Amplitude decay factor for each octave.

    Returns:
        float: Combined noise value.
    """
    total = 0
    frequency = 1.0
    amplitude = 1.0
    max_value = 0  # Used for normalizing the result to [0, 1]

    for _ in range(octaves):
        # Generate noise for this octave
        noise_value = (
            math.sin(x * frequency * 0.1) + math.sin(y * frequency * 0.1)
        ) * 0.5
        total += noise_value * amplitude

        # Update max_value for normalization
        max_value += amplitude

        # Increase frequency and decrease amplitude for the next octave
        frequency *= 2
        amplitude *= persistence

    # Normalize the result to the range [0, 1]
    return total / max_value


def generate_base_map(amplitude, scale, octaves=4, persistence=0.5):
    """
    Generates a base map using Perlin noise with multiple octaves.

    Parameters:
        amplitude (float): Overall amplitude of the noise.
        scale (float): Scaling factor for the noise coordinates.
        octaves (int): Number of noise layers (octaves).
        persistence (float): Amplitude decay factor for each octave.
    """
    for y in range(NY):
        for x in range(NX):
            base_map[x, y] = (
                perlin_noise(x * scale, y * scale, octaves, persistence) * amplitude
            )


generate_base_map(0.01, noise_scale)


@numba.jit
def step_lbm(dt_step, f, f_temp, air_temp):
    feq_array = np.zeros_like(f)
    cell_vel = np.zeros((NX, NY, 2))
    cell_vavg = np.zeros((NX, NY))

    # Compute macroscopic variables and equilibrium distributions
    for y in range(NY):
        for x in range(NX):
            rho = np.sum(f[x, y])
            mom_x = np.sum(f[x, y] * ex)
            mom_y = np.sum(f[x, y] * ey)
            ux = mom_x / rho
            uy = mom_y / rho
            cell_vel[x, y] = [ux, uy]
            cell_vavg[x, y] = (abs(ux) + abs(uy)) / 2
            u_sq = ux**2 + uy**2
            for i in range(9):
                e_dot_u = ex[i] * ux + ey[i] * uy
                feq_array[x, y, i] = (
                    weights[i] * rho * (1 + 3 * e_dot_u + 4.5 * e_dot_u**2 - 1.5 * u_sq)
                )

    # Compute t_inc_average for each cell from incoming neighbors
    t_inc_average = np.zeros((NX, NY))
    for y in range(NY):
        for x in range(NX):
            sum_val = 0
            weight_sum = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx) % NX
                    ny = (y + dy) % NY
                    rx = -dx
                    ry = -dy
                    n_vel = cell_vel[nx, ny]
                    dot = n_vel[0] * rx + n_vel[1] * ry
                    if dot > 0:
                        weight_sum += dot
                        sum_val += dot * air_temp[nx, ny]
            t_inc_average[x, y] = (
                sum_val / weight_sum if weight_sum > 0 else air_temp[x, y]
            )

    # Compute t_inc
    t_inc = np.zeros((NX, NY))
    for y in range(NY):
        for x in range(NX):
            t_prior = air_temp[x, y]
            alpha = min(1, vc * cell_vavg[x, y])
            t_inc[x, y] = alpha * t_inc_average[x, y] + (1 - alpha) * t_prior

    # Compute t_avg_adj
    t_avg_adj = np.zeros((NX, NY))
    for y in range(NY):
        for x in range(NX):
            sum_val = 0
            cnt = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx) % NX
                    ny = (y + dy) % NY
                    sum_val += t_inc[nx, ny]
                    cnt += 1
            t_avg_adj[x, y] = sum_val / cnt

    # Thermal update
    S_array = np.zeros((NX, NY))
    t_final_array = np.zeros((NX, NY))
    for y in range(NY):
        for x in range(NX):
            t_ground = (
                base_map[x, y] + edit_map[x, y] + seasonal_blend * seasonal_map[x, y]
            )
            inc = t_inc[x, y]
            avg_adj = t_avg_adj[x, y]
            d_diff = D_const * (avg_adj - inc)
            t_diff = k_const * (t_ground - inc)
            S_val = t_diff * dpdt
            S_val = max(-d_max, min(d_max, S_val))
            S_array[x, y] = S_val
            t_final_array[x, y] = inc + t_diff

    # Collision step
    for y in range(NY):
        for x in range(NX):
            for i in range(9):
                f[x, y, i] = (
                    f[x, y, i]
                    - (f[x, y, i] - feq_array[x, y, i]) / tau
                    + S_array[x, y] * weights[i]
                )

    # Streaming step
    for y in range(NY):
        for x in range(NX):
            for i in range(9):
                nx = (x + ex[i]) % NX
                ny = (y + ey[i]) % NY
                f_temp[nx, ny, i] = f[x, y, i]

    f, f_temp = f_temp, f

    # Damping
    for y in range(NY):
        for x in range(NX):
            for i in range(9):
                f[x, y, i] = weights[i] + (f[x, y, i] - weights[i]) * (
                    1 - damping * dt_step
                )

    # Update air temperature
    air_temp = t_final_array

    return f, f_temp, air_temp


def get_velocity_at(gx, gy):
    i = int(gx)
    j = int(gy)
    di = gx - i
    dj = gy - j
    i = max(0, min(i, NX - 2))
    j = max(0, min(j, NY - 2))

    def cell_velocity(x, y):
        rho = np.sum(f[x, y])
        mom_x = np.sum(f[x, y] * ex)
        mom_y = np.sum(f[x, y] * ey)
        return [mom_x / rho, mom_y / rho]

    v00 = cell_velocity(i, j)
    v10 = cell_velocity(i + 1, j)
    v01 = cell_velocity(i, j + 1)
    v11 = cell_velocity(i + 1, j + 1)

    ux = (
        (1 - di) * (1 - dj) * v00[0]
        + di * (1 - dj) * v10[0]
        + (1 - di) * dj * v01[0]
        + di * dj * v11[0]
    )
    uy = (
        (1 - di) * (1 - dj) * v00[1]
        + di * (1 - dj) * v10[1]
        + (1 - di) * dj * v01[1]
        + di * dj * v11[1]
    )

    return [ux, uy]


def update_particles(dt):
    global particles, particle_spawn_accumulator
    speed_factor = 200
    for p in particles:
        p["trail"].append(
            {"x": p["x"], "y": p["y"], "t": pygame.time.get_ticks() / 1000}
        )
        while (
            len(p["trail"]) > 0
            and (pygame.time.get_ticks() / 1000 - p["trail"][0]["t"]) > tail_length
        ):
            p["trail"].pop(0)
        gx = p["x"] / CELL_SIZE
        gy = p["y"] / CELL_SIZE
        vel = get_velocity_at(gx, gy)
        p["x"] += vel[0] * speed_factor * dt
        p["y"] += vel[1] * speed_factor * dt
        p["x"] %= NX * CELL_SIZE
        p["y"] %= NY * CELL_SIZE
        p["age"] += dt
        if p["age"] > particle_lifetime:
            particles.remove(p)
    particle_spawn_accumulator += dt * spawn_rate
    while particle_spawn_accumulator >= 1:
        particles.append(
            {
                "x": random.uniform(0, NX * CELL_SIZE),
                "y": random.uniform(0, NY * CELL_SIZE),
                "age": 0,
                "trail": [],
            }
        )
        particle_spawn_accumulator -= 1


def draw_simulation():
    screen.fill((0, 0, 0))
    for p in particles:
        if len(p["trail"]) > 1:
            for i in range(len(p["trail"]) - 1):
                p1 = p["trail"][i]
                p2 = p["trail"][i + 1]
                if (
                    abs(p1["x"] - p2["x"]) > NX * CELL_SIZE / 2
                    or abs(p1["y"] - p2["y"]) > NY * CELL_SIZE / 2
                ):
                    continue
                # alpha1 = (p1['t'] - (pygame.time.get_ticks() / 1000 - tail_length)) / tail_length
                # alpha2 = (p2['t'] - (pygame.time.get_ticks() / 1000 - tail_length)) / tail_length
                # seg_alpha = (alpha1 + alpha2) / 2
                # seg_alpha = max(0, min(1, seg_alpha))
                pygame.draw.line(
                    screen,
                    (0, 255, 255, 1),
                    (int(p1["x"]), int(p1["y"])),
                    (int(p2["x"]), int(p2["y"])),
                    2,
                )
        pygame.draw.circle(screen, (0, 255, 255), (int(p["x"]), int(p["y"])), 2)


def main_loop():
    global f, f_temp, air_temp, particles, particle_spawn_accumulator
    last_time = pygame.time.get_ticks()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        now = pygame.time.get_ticks()
        dt = (now - last_time) / 1000
        last_time = now

        for _ in range(STEPS_PER_FRAME):
            f, f_temp, air_temp = step_lbm(DT, f, f_temp, air_temp)
        update_particles(dt)
        draw_simulation()
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main_loop()
