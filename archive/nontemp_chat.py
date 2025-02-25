import math
import random
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

# New Wind Parameters for simulating actual wind patterns
GLOBAL_WIND = np.array([-0.0001, 0.0])  # Base wind vector (adjust as desired)
WIND_SCALE = 0.1
WIND_TIME_SCALE = 0.1
WIND_AMPLITUDE = 0.00

# New Stability Parameter: lower values reduce forcing amplitude and improve stability.
STABILITY = 1.0  # Adjust between 0.0 and 1.0 as needed

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

# Land map and cities (land_map: 0=water, 1=land)
land_map = np.zeros((NY, NX), dtype=np.uint8)


cities = [
    dict(name="Sparseville", x=63, y=35),
    dict(name="Tensorburg", x=214, y=378),
    dict(name="Bayes Bay", x=160, y=262),
    dict(name="ReLU Ridge", x=413, y=23),
    dict(name="GANopolis", x=318, y=132),
    dict(name="Gradient Grove", x=468, y=158),
    dict(name="Offshore A", x=502, y=356),
    dict(name="Offshore B", x=660, y=184),
]


class PerlinNoise:
    def __init__(self):
        self.permutation = [
            151,
            160,
            137,
            91,
            90,
            15,
            131,
            13,
            201,
            95,
            96,
            53,
            194,
            233,
            7,
            225,
            140,
            36,
            103,
            30,
            69,
            142,
            8,
            99,
            37,
            240,
            21,
            10,
            23,
            190,
            6,
            148,
            247,
            120,
            234,
            75,
            0,
            26,
            197,
            62,
            94,
            252,
            219,
            203,
            117,
            35,
            11,
            32,
            57,
            177,
            33,
            88,
            237,
            149,
            56,
            87,
            174,
            20,
            125,
            136,
            171,
            168,
            68,
            175,
            74,
            165,
            71,
            134,
            139,
            48,
            27,
            166,
            77,
            146,
            158,
            231,
            83,
            111,
            229,
            122,
            60,
            211,
            133,
            230,
            220,
            105,
            92,
            41,
            55,
            46,
            245,
            40,
            244,
            102,
            143,
            54,
            65,
            25,
            63,
            161,
            1,
            216,
            80,
            73,
            209,
            76,
            132,
            187,
            208,
            89,
            18,
            169,
            200,
            196,
            135,
            130,
            116,
            188,
            159,
            86,
            164,
            100,
            109,
            198,
            173,
            186,
            3,
            64,
            52,
            217,
            226,
            250,
            124,
            123,
            5,
            202,
            38,
            147,
            118,
            126,
            255,
            82,
            85,
            212,
            207,
            206,
            59,
            227,
            47,
            16,
            58,
            17,
            182,
            189,
            28,
            42,
            223,
            183,
            170,
            213,
            119,
            248,
            152,
            2,
            44,
            154,
            163,
            70,
            221,
            153,
            101,
            155,
            167,
            43,
            172,
            9,
            129,
            22,
            39,
            253,
            19,
            98,
            108,
            110,
            79,
            113,
            224,
            232,
            178,
            185,
            112,
            104,
            218,
            246,
            97,
            228,
            251,
            34,
            242,
            193,
            238,
            210,
            144,
            12,
            191,
            179,
            162,
            241,
            81,
            51,
            145,
            235,
            249,
            14,
            239,
            107,
            49,
            192,
            214,
            31,
            181,
            199,
            106,
            157,
            184,
            84,
            204,
            176,
            115,
            121,
            50,
            45,
            127,
            4,
            150,
            254,
            138,
            236,
            205,
            93,
            222,
            114,
            67,
            29,
            24,
            72,
            243,
            141,
            128,
            195,
            78,
            66,
            215,
            61,
            156,
            180,
        ]
        self.p = self.permutation * 2

    # fmt:on

    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(self, t, a, b):
        return a + t * (b - a)

    def grad(self, hash, x, y, z):
        h = hash & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if (h == 12 or h == 14) else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def noise(self, x, y, z):
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        Z = int(math.floor(z)) & 255
        x -= math.floor(x)
        y -= math.floor(y)
        z -= math.floor(z)
        u = self.fade(x)
        v = self.fade(y)
        w = self.fade(z)
        A = self.p[X] + Y
        AA = self.p[A] + Z
        AB = self.p[A + 1] + Z
        B = self.p[X + 1] + Y
        BA = self.p[B] + Z
        BB = self.p[B + 1] + Z
        return self.lerp(
            w,
            self.lerp(
                v,
                self.lerp(
                    u,
                    self.grad(self.p[AA], x, y, z),
                    self.grad(self.p[BA], x - 1, y, z),
                ),
                self.lerp(
                    u,
                    self.grad(self.p[AB], x, y - 1, z),
                    self.grad(self.p[BB], x - 1, y - 1, z),
                ),
            ),
            self.lerp(
                v,
                self.lerp(
                    u,
                    self.grad(self.p[AA + 1], x, y, z - 1),
                    self.grad(self.p[BA + 1], x - 1, y, z - 1),
                ),
                self.lerp(
                    u,
                    self.grad(self.p[AB + 1], x, y - 1, z - 1),
                    self.grad(self.p[BB + 1], x - 1, y - 1, z - 1),
                ),
            ),
        )


perlin = PerlinNoise()


def perlin2(x, y):
    return perlin.noise(x, y, 0)


def gen_perlin(amplitude, scale, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    x_offset = rng.random() * 1000
    y_offset = rng.random() * 1000
    output = np.zeros((NY, NX), dtype=np.float32)
    for j in range(NY):
        for i in range(NX):
            output[j, i] = (
                perlin2(i * scale + x_offset, j * scale + y_offset) * amplitude
            )
    return output


base_map = gen_perlin(0.01, noise_scale)
seasonal_map = gen_perlin(0.01, 0.02)
land_map = gen_perlin(0.01, 0.02, seed=0) + gen_perlin(0.01, 0.005, seed=0)

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


SHIFT_DIST = -0.1


def shift_left(array, count):
    """Shifts a 2D array to the left by one column. Optionally wraps around."""
    last_dist = (count - 1) * SHIFT_DIST
    curr = count * SHIFT_DIST

    if curr // CELL_SIZE != last_dist // CELL_SIZE:
        array = np.roll(array, shift=-1, axis=0)
    return array


def shift_particles():
    # Update particle positions
    for p in particles:
        p["x"] -= SHIFT_DIST  # Shift particles left
        if p["x"] < 0:
            p["x"] += NX * CELL_SIZE  # Wrap around if needed


def step_lbm(dt_step, wind_field_x, wind_field_y, count):
    global f, f_temp, base_map
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
        # Force from pressure differences and interactive editing
        force_term = (base_map + edit_map) * weights[i]
        # Additional wind forcing based on global wind and evolving noise field
        wind_force = weights[i] * (wind_field_x * ex[i] + wind_field_y * ey[i])
        # Scale the forcing term by the stability parameter
        S = STABILITY * (force_term + wind_force)
        f[:, :, i] += -(f[:, :, i] - feq) / TAU + S

    # Streaming step
    for i in range(9):
        f_temp[:, :, i] = np.roll(f[:, :, i], ex[i], axis=1)
        f_temp[:, :, i] = np.roll(f_temp[:, :, i], ey[i], axis=0)

    # Apply damping
    f = f_temp * (1 - DAMPING * dt_step) + weights * DAMPING * dt_step
    f_temp.fill(0)

    # Wind
    # apply wind by shifting the base map to the left
    # f = shift_left(f, count)
    # f_temp = shift_left(f_temp, count)
    # base_map = shift_left(base_map, count)

    # shift_particles()

    # slightly change the base map
    # print(np.mean(base_map), np.max(base_map), np.min(base_map))
    # exit()
    base_map += (generate_base_map() - base_map) * 0.02
    # normalize the base map to a mean of 0, and a range of 0.04
    base_map = (
        (base_map - np.mean(base_map)) / (np.max(base_map) - np.min(base_map)) * 0.02
    )
    # base_map = generate_base_map()


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


def draw(count):
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
        offset_x = count * SHIFT_DIST
        for p in particles:
            if len(p["trail"]) > 1:
                for i in range(len(p["trail"]) - 1):
                    x1, y1, t1 = p["trail"][i]
                    x2, y2, t2 = p["trail"][i + 1]
                    # print(p["trail"])

                    trail_offset_x = -(len(p["trail"]) - i) * SHIFT_DIST
                    x1 += offset_x + trail_offset_x
                    x2 += offset_x + trail_offset_x
                    x1 %= screen_width
                    x2 %= screen_width

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
            x = p["x"] + offset_x
            pygame.draw.circle(
                screen,
                (0, 255, 255),
                (int(x % screen_width), int(p["y"] % screen_height)),
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

    # Compute wind forcing arrays for the entire grid once per frame
    ix, iy = np.meshgrid(np.arange(NX), np.arange(NY))
    vec_noise = np.vectorize(
        lambda i, j: noise.pnoise3(
            i * WIND_SCALE, j * WIND_SCALE, sim_time * WIND_TIME_SCALE
        )
    )
    wind_noise_x = vec_noise(ix, iy)
    vec_noise_y = np.vectorize(
        lambda i, j: noise.pnoise3(
            i * WIND_SCALE + 100, j * WIND_SCALE + 100, sim_time * WIND_TIME_SCALE
        )
    )
    wind_noise_y = vec_noise_y(ix, iy)
    wind_field_x = GLOBAL_WIND[0] + wind_noise_x * WIND_AMPLITUDE
    wind_field_y = GLOBAL_WIND[1] + wind_noise_y * WIND_AMPLITUDE

    if not editor_mode:
        for _ in range(STEPS_PER_FRAME):
            step_lbm(dt / STEPS_PER_FRAME, wind_field_x, wind_field_y, count)
            # count += 1
        update_particles(dt)

    draw(count)
    pygame.display.update()
    count += 1
