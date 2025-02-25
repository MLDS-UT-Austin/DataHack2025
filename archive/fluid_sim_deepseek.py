import sys

import noise
import numpy as np
import pygame
from pygame.locals import *

# Simulation parameters
NX, NY = 200, 100
CELL_SIZE = 4
STEPS_PER_FRAME = 5
TAU = 0.04 * 3 + 0.5
DAMPING = 0.02
NOISE_AMPLITUDE = 0.01
NOISE_SCALE = 0.05
TAIL_LENGTH = 10.0
SPAWN_RATE = 50
PARTICLE_LIFETIME = 15

# D2Q9 lattice constants
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
weights = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float32,
)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((NX * CELL_SIZE, NY * CELL_SIZE))
clock = pygame.time.Clock()

# LBM arrays
f = np.tile(weights, (NY, NX, 1)).astype(np.float32)
f_temp = np.zeros_like(f)
base_map = np.zeros((NY, NX), dtype=np.float32)
edit_map = np.zeros((NY, NX), dtype=np.float32)

# Generate initial base map using Perlin noise
for y in range(NY):
    for x in range(NX):
        base_map[y, x] = (
            noise.pnoise2(x * NOISE_SCALE, y * NOISE_SCALE, octaves=1) * NOISE_AMPLITUDE
        )

# Particle system
particles = []
sim_time = 0.0
spawn_accumulator = 0.0

# UI state
show_divergence = False
editor_mode = False
painting = False
brush_size = 5
brush_value = 0.01
brush_weight = 1.0


class Slider:
    def __init__(self, pos, size, min_val, max_val, initial_val):
        self.pos = pos
        self.size = size
        self.min = min_val
        self.max = max_val
        self.val = initial_val
        self.dragging = False

    def draw(self, screen):
        pygame.draw.rect(screen, (100, 100, 100), (self.pos, self.size))
        handle_x = int(
            (self.val - self.min) / (self.max - self.min) * self.size[0] + self.pos[0]
        )
        pygame.draw.circle(
            screen, (200, 200, 200), (handle_x, self.pos[1] + self.size[1] // 2), 8
        )

    def update(self, events):
        mouse_pos = pygame.mouse.get_pos()
        for event in events:
            if event.type == MOUSEBUTTONDOWN:
                if pygame.Rect(self.pos, self.size).collidepoint(mouse_pos):
                    self.dragging = True
            elif event.type == MOUSEMOTION and self.dragging:
                rel_x = mouse_pos[0] - self.pos[0]
                self.val = self.min + (rel_x / self.size[0]) * (self.max - self.min)
                self.val = np.clip(self.val, self.min, self.max)
            elif event.type == MOUSEBUTTONUP:
                self.dragging = False


# Create sliders (example - extend as needed)
viscosity_slider = Slider((20, 20), (100, 20), 0.01, 0.2, 0.04)


def step_lbm(dt_step, count):
    global f, f_temp
    # Collision step
    rho = np.sum(f, axis=2)
    ux = np.sum(f * ex, axis=2) / rho
    uy = np.sum(f * ey, axis=2) / rho

    # Compute equilibrium and apply forces
    for i in range(9):
        e_dot_u = ex[i] * ux + ey[i] * uy
        feq = (
            weights[i]
            * rho
            * (1 + 3 * e_dot_u + 4.5 * (e_dot_u**2) - 1.5 * (ux**2 + uy**2))
        )
        S = (base_map + edit_map) * weights[i]
        f[:, :, i] += -(f[:, :, i] - feq) / TAU + S  # FIXME

    # Streaming step
    for i in range(9):
        f_temp[:, :, i] = np.roll(f[:, :, i], ex[i], axis=1)
        f_temp[:, :, i] = np.roll(f_temp[:, :, i], ey[i], axis=0)

    # Apply damping
    f = f_temp * (1 - DAMPING * dt_step) + weights * DAMPING * dt_step
    f_temp.fill(0)

    # apply wind by shifting the base map to the left by 0.0001 cells
    if count % 50 == 0:
        temp = base_map[:, 0]
        base_map[:, :-1] = base_map[:, 1:]
        base_map[:, -1] = temp
        print(count)


def get_velocity(x, y):
    ix, iy = int(x), int(y)
    ix = np.clip(ix, 0, NX - 2)
    iy = np.clip(iy, 0, NY - 2)
    dx = x - ix
    dy = y - iy

    # Bilinear interpolation
    v00 = get_cell_velocity(ix, iy)
    v10 = get_cell_velocity(ix + 1, iy)
    v01 = get_cell_velocity(ix, iy + 1)
    v11 = get_cell_velocity(ix + 1, iy + 1)

    ux = (
        (1 - dx) * (1 - dy) * v00[0]
        + dx * (1 - dy) * v10[0]
        + (1 - dx) * dy * v01[0]
        + dx * dy * v11[0]
    )
    uy = (
        (1 - dx) * (1 - dy) * v00[1]
        + dx * (1 - dy) * v10[1]
        + (1 - dx) * dy * v01[1]
        + dx * dy * v11[1]
    )
    return (ux * 200, uy * 200)


def get_cell_velocity(x, y):
    rho = np.sum(f[y, x])
    ux = np.sum(f[y, x] * ex) / rho
    uy = np.sum(f[y, x] * ey) / rho
    return (ux, uy)


def update_particles(dt):
    global particles, sim_time, spawn_accumulator
    sim_time += dt

    # Update existing particles
    for p in particles:
        vel = get_velocity(p["x"] / CELL_SIZE, p["y"] / CELL_SIZE)
        p["x"] += vel[0] * dt
        p["y"] += vel[1] * dt
        p["age"] += dt

        # Boundary wrap
        p["x"] %= NX * CELL_SIZE
        p["y"] %= NY * CELL_SIZE

        # Update trail
        p["trail"].append((p["x"], p["y"], sim_time))
        while p["trail"] and (sim_time - p["trail"][0][2]) > TAIL_LENGTH:
            p["trail"].pop(0)

    # Remove old particles
    particles = [p for p in particles if p["age"] <= PARTICLE_LIFETIME]

    # Spawn new particles
    spawn_accumulator += dt * SPAWN_RATE
    while spawn_accumulator >= 1:
        particles.append(
            {
                "x": np.random.uniform(0, NX * CELL_SIZE),
                "y": np.random.uniform(0, NY * CELL_SIZE),
                "age": 0,
                "trail": [],
            }
        )
        spawn_accumulator -= 1


def draw():
    screen.fill((0, 0, 0))

    if editor_mode or show_divergence:
        # Draw divergence map
        div = base_map + edit_map
        div_normalized = np.clip((div + 0.02) / 0.04 * 255, 0, 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(div_normalized)
        surf = pygame.transform.scale(surf, (NX * CELL_SIZE, NY * CELL_SIZE))
        screen.blit(surf, (0, 0))
    else:
        # Draw particles with proper wrap-around handling
        screen_width = NX * CELL_SIZE
        screen_height = NY * CELL_SIZE
        for p in particles:
            if len(p["trail"]) > 1:
                for i in range(len(p["trail"]) - 1):
                    x1, y1, t1 = p["trail"][i]
                    x2, y2, t2 = p["trail"][i + 1]

                    # Skip segments that wrap around the screen
                    if (
                        abs(x1 - x2) > screen_width / 2
                        or abs(y1 - y2) > screen_height / 2
                    ):
                        continue

                    alpha = int(255 * (1 - (sim_time - t1) / TAIL_LENGTH))
                    pygame.draw.line(
                        screen,
                        (0, 255, 255, alpha),
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        2,
                    )

            # Draw current particle position
            pygame.draw.circle(
                screen,
                (0, 255, 255),
                (int(p["x"] % screen_width), int(p["y"] % screen_height)),
                2,
            )

    # Draw UI elements
    viscosity_slider.draw(screen)
    pygame.display.flip()


def handle_events():
    global editor_mode, show_divergence, painting, brush_size, brush_value, brush_weight
    events = pygame.event.get()
    for event in events:
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEBUTTONDOWN:
            if editor_mode:
                painting = True
        elif event.type == MOUSEMOTION and painting and editor_mode:
            paint(event.pos)
        elif event.type == MOUSEBUTTONUP:
            painting = False

    viscosity_slider.update(events)


def paint(pos):
    x, y = pos[0] // CELL_SIZE, pos[1] // CELL_SIZE
    radius = brush_size
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dist = np.sqrt(dx**2 + dy**2)
            if dist > radius:
                continue
            factor = (1 - dist / radius) * brush_weight
            nx, ny = x + dx, y + dy
            if 0 <= nx < NX and 0 <= ny < NY:
                current = base_map[ny, nx] + edit_map[ny, nx]
                target = current * (1 - factor) + brush_value * factor
                edit_map[ny, nx] = np.clip(target - base_map[ny, nx], -0.02, 0.02)


count = 0
while True:
    dt = clock.tick(60) / 1000.0
    handle_events()

    if not editor_mode:
        for _ in range(STEPS_PER_FRAME):
            step_lbm(dt / STEPS_PER_FRAME, count)
            count += 1
        update_particles(dt)

    draw()
    pygame.display.update()
