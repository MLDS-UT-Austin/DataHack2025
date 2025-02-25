import math
import random
import time

import numba
import numpy as np
import pygame
from numba.typed import List
from numba.types import float32, int32

# ---------------------------
# Simulation Parameters
# ---------------------------
NX = 200  # grid width (cells)
NY = 100  # grid height (cells)
cell_size = 4  # pixels per cell
steps_per_frame = 5  # LBM steps per animation frame

# LBM lattice directions and weights (D2Q9)
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
weights = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float32,
)
cs2 = 1 / 3

# Fluid simulation parameters
viscosity = 0.04
tau = viscosity * 3 + 0.5
damping = 0.02
noise_amplitude = 0.01
noise_scale = 0.05
tail_length = 10.0
spawn_rate = 100

# Thermal parameters
D_const = 0.0
k_const = 0.0007
# dpdt = 300.0
dpdt = 30000.0
vc = 1.0
d_max = 0.005

# Hurricane control (defaults from JS sliders)
hurricane_divergence_rate = 0.01
hurricane_growth_rate = 100

GLOBAL_WIND = -0.1

# ---------------------------
# Data Arrays and Initialization
# ---------------------------
# LBM distribution functions: shape (NY, NX, 9)
f = np.empty((NY, NX, 9), dtype=np.float32)
f_temp = np.empty_like(f)
# Initialize with equilibrium values (ρ = 1, u = 0)
for j in range(NY):
    for i in range(NX):
        f[j, i, :] = weights

# Ground temperature maps (base, edit, seasonal)
base_map = np.empty((NY, NX), dtype=np.float32)
edit_map = np.zeros((NY, NX), dtype=np.float32)
seasonal_map = np.zeros((NY, NX), dtype=np.float32)

# Air temperature field (one per cell)
air_temp = np.zeros((NY, NX), dtype=np.float32)

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

# Hurricane variables
hurricane_ids = List.empty_list(int32)
hurricane_x = List.empty_list(float32)
hurricane_y = List.empty_list(float32)
hurricane_radius = List.empty_list(float32)
next_hurricane_id = 1
hurricane_spawn_acc = 0.0
hurricane_records = []  # log of hurricanes (updated periodically)

# Time and seasonal effects
current_month = 0
current_year = 1955


# seasonal_effects = np.zeros(12, dtype=np.float32)
# Seasonal effects using a sine wave for smoother transition
def seasonal_sine_wave(month):
    """Generate seasonal effect using a sine wave (peaks in summer)"""
    # Shifted and scaled sine wave: peaks at month ~7 (July)
    return 0.5 * (1 + np.sin((month / 12.0 * 2 * np.pi) - np.pi / 2))


# Initial seasonal effect values
seasonal_effects = np.array(
    [seasonal_sine_wave(m) for m in range(12)], dtype=np.float32
)

# Particle system (for flow visualization)
particles = []
particle_lifetime = 40.0
particle_spawn_acc = 0.0

TRAIL_LENGTH = 5

# ---------------------------
# Perlin Noise (for base map generation)
# ---------------------------
# fmt:off
class PerlinNoise:
    def __init__(self):
        self.permutation = [151,160,137,91,90,15,
            131,13,201,95,96,53,194,233,7,225,140,36,103,30,
            69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,
            75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,
            33,88,237,149,56,87,174,20,125,136,171,168,68,175,
            74,165,71,134,139,48,27,166,77,146,158,231,83,111,
            229,122,60,211,133,230,220,105,92,41,55,46,245,40,
            244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,
            187,208,89,18,169,200,196,135,130,116,188,159,86,
            164,100,109,198,173,186,3,64,52,217,226,250,124,123,
            5,202,38,147,118,126,255,82,85,212,207,206,59,227,
            47,16,58,17,182,189,28,42,223,183,170,213,119,248,
            152,2,44,154,163,70,221,153,101,155,167,43,172,9,
            129,22,39,253,19,98,108,110,79,113,224,232,178,185,
            112,104,218,246,97,228,251,34,242,193,238,210,144,
            12,191,179,162,241,81,51,145,235,249,14,239,107,49,
            192,214,31,181,199,106,157,184,84,204,176,115,121,50,
            45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,
            243,141,128,195,78,66,215,61,156,180]
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
        return self.lerp(w,
               self.lerp(v,
                   self.lerp(u, self.grad(self.p[AA], x, y, z),
                                self.grad(self.p[BA], x - 1, y, z)),
                   self.lerp(u, self.grad(self.p[AB], x, y - 1, z),
                                self.grad(self.p[BB], x - 1, y - 1, z))
               ),
               self.lerp(v,
                   self.lerp(u, self.grad(self.p[AA + 1], x, y, z - 1),
                                self.grad(self.p[BA + 1], x - 1, y, z - 1)),
                   self.lerp(u, self.grad(self.p[AB + 1], x, y - 1, z - 1),
                                self.grad(self.p[BB + 1], x - 1, y - 1, z - 1))
               )
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
            output[j, i] = perlin2(i * scale+x_offset, j * scale+ y_offset) * amplitude
    return output

base_map = gen_perlin(0.01, noise_scale)
seasonal_map = gen_perlin(0.01, 0.02)
land_map = gen_perlin(0.01, 0.02, seed=0) + gen_perlin(0.01, 0.005, seed=0)

# center = (75, 50)
# for j in range(NY):
#     for i in range(NX):
#         dist = math.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
#         if dist < 5:
#             land_map[j, i] = 1
land_map = np.where(land_map > 0.005, 1, 0)

# ---------------------------
# Helper Functions
# ---------------------------
def spawn_particle():
    particles.append({
        "x": random.random() * NX * cell_size,
        "y": random.random() * NY * cell_size,
        "age": 0.0,
        "trail": []
    })

def get_velocity_at(gx, gy):
    # Bilinear interpolation of cell velocities
    i = int(math.floor(gx))
    j = int(math.floor(gy))
    di = gx - i
    dj = gy - j
    i = max(0, min(i, NX - 2))
    j = max(0, min(j, NY - 2))
    def cell_velocity(i, j):
        rho = np.sum(f[j, i, :])
        ux = np.sum(f[j, i, :] * ex)
        uy = np.sum(f[j, i, :] * ey)
        return ux / rho, uy / rho
    v00 = cell_velocity(i, j)
    v10 = cell_velocity(i + 1, j)
    v01 = cell_velocity(i, j + 1)
    v11 = cell_velocity(i + 1, j + 1)
    ux = (1 - di) * (1 - dj) * v00[0] + di * (1 - dj) * v10[0] + (1 - di) * dj * v01[0] + di * dj * v11[0]
    uy = (1 - di) * (1 - dj) * v00[1] + di * (1 - dj) * v10[1] + (1 - di) * dj * v01[1] + di * dj * v11[1]
    ux += GLOBAL_WIND
    return ux, uy

# ---------------------------
# Simulation Update Functions
# ---------------------------
sim_step_count = 0

@numba.jit
def step_lbm(dt_step, f, f_temp, air_temp, sim_step_count, hurricane_ids, hurricane_x, hurricane_y, hurricane_radius, base_map, edit_map, seasonal_map, current_month:int):
    # (1) Compute macroscopic variables and equilibrium distribution
    feq = np.empty_like(f)
    cell_vel = [[(0, 0) for _ in range(NX)] for _ in range(NY)]
    cell_vavg = np.zeros((NY, NX), dtype=np.float32)
    for j in range(NY):
        for i in range(NX):
            rho = np.sum(f[j, i, :])
            mom_x = np.sum(f[j, i, :] * ex)
            mom_y = np.sum(f[j, i, :] * ey)
            ux = mom_x / rho
            uy = mom_y / rho
            cell_vel[j][i] = (ux, uy)
            cell_vavg[j, i] = (abs(ux) + abs(uy)) / 2
            u_sq = ux * ux + uy * uy
            for k in range(9):
                e_dot_u = ex[k] * ux + ey[k] * uy
                feq[j, i, k] = weights[k] * rho * (1 + 3 * e_dot_u + 4.5 * e_dot_u * e_dot_u - 1.5 * u_sq)
    # (2) Compute t_inc_average from incoming neighbors
    t_inc_average = np.empty((NY, NX), dtype=np.float32)
    for j in range(NY):
        for i in range(NX):
            s_val = 0.0
            w_sum = 0.0
            for dj in [-1, 0, 1]:
                for di in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni = (i + di) % NX
                    nj = (j + dj) % NY
                    rx, ry = -di, -dj
                    nvel = cell_vel[nj][ni]
                    dot = nvel[0] * rx + nvel[1] * ry
                    if dot > 0:
                        w_sum += dot
                        s_val += dot * air_temp[nj, ni]
            t_inc_average[j, i] = s_val / w_sum if w_sum > 0 else air_temp[j, i]
    # (3) Compute t_inc via affine combination
    t_inc = np.empty((NY, NX), dtype=np.float32)
    for j in range(NY):
        for i in range(NX):
            alpha = min(1, vc * cell_vavg[j, i])
            t_inc[j, i] = alpha * t_inc_average[j, i] + (1 - alpha) * air_temp[j, i]
    # (4) Compute t_avg_adj (adjacent neighbors)
    t_avg_adj = np.empty((NY, NX), dtype=np.float32)
    for j in range(NY):
        for i in range(NX):
            total = 0.0
            count = 0
            for dj in [-1, 0, 1]:
                for di in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni = (i + di) % NX
                    nj = (j + dj) % NY
                    total += t_inc[nj, ni]
                    count += 1
            t_avg_adj[j, i] = total / count
    # (5) Thermal update: compute forcing S and update temperature
    S_array = np.empty((NY, NX), dtype=np.float32)
    t_final_array = np.empty((NY, NX), dtype=np.float32)
    for j in range(NY):
        for i in range(NX):
            # Ground temperature uses base_map, edit_map and seasonal effect
            seasonal_scale = -0.5*math.cos(current_month / 12 * math.pi * 2) + 0.5  # 0 at month 0, 1 at month 6, 0 at month 12
            t_ground = base_map[j, i] + edit_map[j, i] + seasonal_scale * seasonal_map[j, i]
            inc = t_inc[j, i]
            t_diff = k_const * (t_ground - inc)
            S_val = t_diff * dpdt
            S_val = max(-d_max, min(d_max, S_val))
            S_array[j, i] = S_val
            t_final_array[j, i] = inc + t_diff
    # (5b) Hurricane-induced divergence (adds extra forcing)
    for h_index in range(len(hurricane_ids)):
        h_x = hurricane_x[h_index]
        h_y = hurricane_y[h_index]
        h_radius = hurricane_radius[h_index]
        r = int(math.ceil(h_radius))
        ix = int(math.floor(h_x))
        iy = int(math.floor(h_y))
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                cx = (ix + di)
                cy = (iy + dj)
                if 0 <= cx < NX and 0 <= cy < NY:
                    dx = (cx + 0.5) - h_x
                    dy = (cy + 0.5) - h_y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < h_radius:
                        # if dist < h_radius / 2:
                        #     effect = hurricane_divergence_rate
                        # else:
                        #     temp = (dist -h_radius / 2)/ (h_radius/2)
                        #     effect = hurricane_divergence_rate * (1 - temp **2)
                        effect = hurricane_divergence_rate * (1 - (dist / h_radius) ** 2)
                        S_array[cy, cx] = - effect
    # (6) Collision step
    for j in range(NY):
        for i in range(NX):
            for k in range(9):
                f[j, i, k] = f[j, i, k] - (f[j, i, k] - feq[j, i, k]) / tau + S_array[j, i] * weights[k]
    # (7) Streaming step (propagate distributions)
    for j in range(NY):
        for i in range(NX):
            for k in range(9):
                ni = (i + ex[k]) % NX
                nj = (j + ey[k]) % NY
                f_temp[nj, ni, k] = f[j, i, k]
    f[:,:,:] = f_temp
    # (8) Damping step
    for j in range(NY):
        for i in range(NX):
            for k in range(9):
                f[j, i, k] = weights[k] + (f[j, i, k] - weights[k]) * (1 - damping * dt_step)
    # (9) Update air temperature field
    air_temp[:,:] = t_final_array
    sim_step_count += steps_per_frame
    
    # shift everything by the global wind
    previous_mod = int((sim_step_count-steps_per_frame)*dt_step*GLOBAL_WIND % NX)
    current_mod = int(sim_step_count*dt_step*GLOBAL_WIND % NX)
    def shift_array(arr, shift):
        return np.concatenate((arr[:, -shift:], arr[:, :-shift]), axis=1)

    # print(sim_step_count, dt_step, GLOBAL_WIND, NX)
    if previous_mod > current_mod:
        shift = np.int64(current_mod - previous_mod)
        f = shift_array(f, shift)
        f_temp = shift_array(f_temp, shift)
        air_temp = shift_array(air_temp, shift)
        base_map = shift_array(base_map, shift)
        edit_map = shift_array(edit_map, shift)
        seasonal_map = shift_array(seasonal_map, shift)

    return f, f_temp, air_temp, sim_step_count, hurricane_ids, hurricane_x, hurricane_y, hurricane_radius, base_map, edit_map, seasonal_map

def update_hurricanes(dt):
    global hurricane_spawn_acc, next_hurricane_id, hurricane_records, hurricane_ids, hurricane_x, hurricane_y, hurricane_radius
    hurricane_spawn_acc += dt * steps_per_frame
    spawn_threshold = 20  # using a default value (steps per spawn inverse)
    if len(hurricane_ids) < 1:
        idx = random.randint(0, NX * NY - 1)
        j = idx // NX
        i = idx % NX
        if air_temp[j, i] > 0 and land_map[j, i] == 0:
            hurricane_ids.append(next_hurricane_id)
            hurricane_x.append(i + 0.5)
            hurricane_y.append(j + 0.5)
            hurricane_radius.append(10.0)
            next_hurricane_id += 1
        hurricane_spawn_acc -= spawn_threshold
    # Update each hurricane's size and position
    for h_index in range(len(hurricane_ids)):
        h_id = hurricane_ids[h_index]
        h_x = hurricane_x[h_index]
        h_y = hurricane_y[h_index]
        h_radius = hurricane_radius[h_index]
        i = int(math.floor(h_x))
        j = int(math.floor(h_y))
        if 0 <= i < NX and 0 <= j < NY:
            is_water = (land_map[j, i] == 0)
            if is_water:
                h_radius += hurricane_growth_rate * air_temp[j, i]
            else:
                h_radius -= hurricane_growth_rate
            h_radius = min(20, h_radius)
            # Average velocity inside hurricane radius
            # sum_ux, sum_uy, count = 0.0, 0.0, 0
            # r = int(math.ceil(h_radius))
            # for dj in range(-r, r + 1):
            #     for di in range(-r, r + 1):
            #         cx = i + di
            #         cy = j + dj
            #         if 0 <= cx < NX and 0 <= cy < NY:
            #             if math.sqrt(di * di + dj * dj) <= h_radius:
            #                 ux, uy = get_velocity_at(cx, cy)
            #                 sum_ux += ux
            #                 sum_uy += uy
            #                 count += 1
            # if count > 0:
            avg_ux, avg_uy = get_velocity_at(h_x, h_y)
            h_x += avg_ux * dt * 10
            h_y += avg_uy * dt * 10
            # Wrap around periodic boundaries
            if h_x < 0: h_x += NX
            if h_y < 0: h_y += NY
            if h_x >= NX: h_x -= NX
            if h_y >= NY: h_y -= NY
        hurricane_x[h_index] = h_x
        hurricane_y[h_index] = h_y
        hurricane_radius[h_index] = h_radius
        if h_radius < 1.0:
            hurricane_ids.pop(h_index)
            hurricane_x.pop(h_index)
            hurricane_y.pop(h_index)
            hurricane_radius.pop(h_index)
        
    if sim_step_count % 1000 < steps_per_frame:
        for i in range(len(hurricane_ids)):
            h_id = hurricane_ids[i]
            h_x = hurricane_x[i]
            h_y = hurricane_y[i]
            h_radius = hurricane_radius[i]
            hurricane_records.append({"id": h_id, "size": h_radius, "x": h_x, "y": h_y})

def update_particles(dt):
    global particle_spawn_acc
    sim_time = time.time()
    speed_factor = 200
    for p in particles[:]:
        p["trail"].append((p["x"], p["y"], sim_time))
        p["trail"] = [pt for pt in p["trail"] if sim_time - pt[2] <= tail_length]
        gx = p["x"] / cell_size
        gy = p["y"] / cell_size
        vx, vy = get_velocity_at(gx, gy)
        p["x"] += vx * speed_factor * dt + GLOBAL_WIND * dt
        p["y"] += vy * speed_factor * dt
        # Periodic boundaries
        if p["x"] < 0: p["x"] += NX * cell_size
        if p["y"] < 0: p["y"] += NY * cell_size
        if p["x"] >= NX * cell_size: p["x"] -= NX * cell_size
        if p["y"] >= NY * cell_size: p["y"] -= NY * cell_size
        p["age"] += dt
        if p["age"] > particle_lifetime:
            particles.remove(p)
    particle_spawn_acc += dt * spawn_rate
    while particle_spawn_acc >= 1:
        spawn_particle()
        particle_spawn_acc -= 1

# ---------------------------
# Pygame Setup
# ---------------------------
pygame.init()
screen = pygame.display.set_mode((NX * cell_size, NY * cell_size), depth=32)
pygame.display.set_caption("LBM Fluid Simulation – Python Port")
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
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused

    if not paused:
        for _ in range(steps_per_frame):
            f, f_temp, air_temp, sim_step_count, hurricane_ids, hurricane_x, hurricane_y, hurricane_radius, base_map, edit_map, seasonal_map = step_lbm(dt, f, f_temp, air_temp, sim_step_count, hurricane_ids, hurricane_x, hurricane_y, hurricane_radius, base_map, edit_map, seasonal_map, int(current_month % 12))
        update_hurricanes(dt)
        update_particles(dt)
    current_month += dt * 0.2
    
    # Draw land map (blue for water, green for land)
    for j in range(NY):
        for i in range(NX):
            # Compute temperature value
            T = base_map[j, i] + edit_map[j, i] + seasonal_effects[int(current_month % 12)] * seasonal_map[j, i]
            # T =  seasonal_effects[int(current_month % 12)] * seasonal_map[j, i]
            scale = ((T + 0.02) / 0.04)
            scale = max(0, min(1, scale))


            if land_map[j, i] == 0:
                color = (0, 0, 255*scale)  # Blue for water
            else:
                color = (0, 255*scale, 0)  # Blue for water

            pygame.draw.rect(screen, color, (i * cell_size, j * cell_size, cell_size, cell_size))


    # Draw hurricanes (red circles)
    for i in range(len(hurricane_ids)):
        h_x = hurricane_x[i]
        h_y = hurricane_y[i]
        h_radius = hurricane_radius[i]
        pygame.draw.circle(screen, (255, 0, 0),
                           (int(h_x * cell_size), int(h_y * cell_size)),
                           int(h_radius * cell_size), 1)
    # Draw particle trails (green lines)
    for p in particles:
        if len(p["trail"]) > 1:
            for start in range(max(0, len(p["trail"]) - TRAIL_LENGTH), len(p["trail"]) - 1):
                x0, y0, _ = p["trail"][start]
                x1, y1, _ = p["trail"][start + 1]
                if abs(x0 - x1) < NX * cell_size / 2 and abs(y0 - y1) < NY * cell_size / 2:
                    pygame.draw.line(screen, (190, 190, 190),
                            (int(x0), int(y0)),
                            (int(x1), int(y1)), 1)
                    
    # Draw cities
    for city in cities:
        x = city["x"]
        y = city["y"]
        pygame.draw.circle(screen, (255, 255, 255), (x, y), 5)
        font = pygame.font.Font(None, 24)
        text = font.render(city["name"], True, (255, 255, 255))
        screen.blit(text, (x + 10, y - 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
