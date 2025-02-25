import math, random, time
import numpy as np
import pygame
from numba import njit

# -------------------------------
# Simulation Parameters
# -------------------------------
NX = 200        # grid width (cells)
NY = 100        # grid height (cells)
cellSize = 4    # pixels per cell
stepsPerFrame = 5

# D2Q9 lattice directions and weights:
ex = np.array([0,  1,  0, -1,  0,  1, -1, -1,  1], dtype=np.float32)
ey = np.array([0,  0,  1,  0, -1,  1,  1, -1, -1], dtype=np.float32)
weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
cs2 = 1/3

# Fluid simulation parameters:
viscosity = 0.04
tau = viscosity * 3 + 0.5
damping = 0.02
noiseAmplitude = 100
noiseScale = 0.05
tailLength = 10.0
spawnRate = 50  # particles per second
seasonalBlend = 0.0

# Thermal simulation parameters:
D_const = 0.0
k_const = 0.7
dpdt = 300.0
vc = 1.0
d_max = 0.01

# -------------------------------
# Numpy Array Initialization
# -------------------------------
# f and fTemp: shape (NY, NX, 9) with initial equilibrium (all cells at rest)
f = np.empty((NY, NX, 9), dtype=np.float32)
fTemp = np.empty((NY, NX, 9), dtype=np.float32)
for y in range(NY):
    for x in range(NX):
        for i in range(9):
            f[y, x, i] = weights[i]
            fTemp[y, x, i] = weights[i]

# Air temperature field:
airTemp = np.zeros((NY, NX), dtype=np.float32)

# Ground temperature maps:
baseMap = np.empty((NY, NX), dtype=np.float32)
editMap = np.zeros((NY, NX), dtype=np.float32)
seasonalMap = np.zeros((NY, NX), dtype=np.float32)

# -------------------------------
# Perlin Noise (for baseMap)
# -------------------------------
# (Based on Ken Perlin’s improved noise algorithm.)
_permutation = [151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
                140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
                247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
                57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
                74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
                60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
                65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,
                196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,
                217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,
                206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,
                152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,
                253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
                228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,
                235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,
                176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,
                29,24,72,243,141,128,195,78,66,215,61,156,180]
p = np.empty(512, dtype=np.int32)
for i in range(256):
    p[i] = _permutation[i]
    p[256+i] = _permutation[i]

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(t, a, b):
    return a + t * (b - a)

def grad(hash, x, y, z):
    h = hash & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if (h == 12 or h == 14) else z)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

def noise(x, y, z):
    X = int(math.floor(x)) & 255
    Y = int(math.floor(y)) & 255
    Z = int(math.floor(z)) & 255
    x -= math.floor(x)
    y -= math.floor(y)
    z -= math.floor(z)
    u = fade(x)
    v = fade(y)
    w = fade(z)
    A  = p[X] + Y
    AA = p[A] + Z
    AB = p[A + 1] + Z
    B  = p[X + 1] + Y
    BA = p[B] + Z
    BB = p[B + 1] + Z
    return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
                                  grad(p[BA], x-1, y, z)),
                           lerp(u, grad(p[AB], x, y-1, z),
                                  grad(p[BB], x-1, y-1, z))),
               lerp(v, lerp(u, grad(p[AA+1], x, y, z-1),
                            grad(p[BA+1], x-1, y, z-1)),
                    lerp(u, grad(p[AB+1], x, y-1, z-1),
                         grad(p[BB+1], x-1, y-1, z-1))))

def perlin2(x, y):
    return noise(x, y, 0)

# Generate baseMap using Perlin noise.
for y in range(NY):
    for x in range(NX):
        baseMap[y, x] = perlin2(x * noiseScale, y * noiseScale) * 0.01

# -------------------------------
# Numba-accelerated LBM & Thermal Step
# -------------------------------
@njit
def stepLBM(f, fTemp, airTemp, baseMap, editMap, seasonalMap,
            NX, NY, ex, ey, weights, tau, damping, dtStep,
            D_const, k_const, dpdt, vc, d_max, seasonalBlend):
    # Temporary arrays:
    feq = np.empty((NY, NX, 9), dtype=np.float32)
    cellVavg = np.empty((NY, NX), dtype=np.float32)
    cellVel_ux = np.empty((NY, NX), dtype=np.float32)
    cellVel_uy = np.empty((NY, NX), dtype=np.float32)
    
    # (1) Compute macroscopic variables and equilibrium distributions.
    for y in range(NY):
        for x in range(NX):
            rho = 0.0
            momX = 0.0
            momY = 0.0
            for i in range(9):
                val = f[y, x, i]
                rho += val
                momX += val * ex[i]
                momY += val * ey[i]
            ux = momX / rho
            uy = momY / rho
            cellVel_ux[y, x] = ux
            cellVel_uy[y, x] = uy
            cellVavg[y, x] = (abs(ux) + abs(uy)) / 2.0
            usq = ux * ux + uy * uy
            for i in range(9):
                edotu = ex[i] * ux + ey[i] * uy
                feq[y, x, i] = weights[i] * rho * (1 + 3 * edotu + 4.5 * edotu * edotu - 1.5 * usq)
                
    # (2) Compute t_inc_average from incoming neighbors.
    t_inc_average = np.empty((NY, NX), dtype=np.float32)
    for y in range(NY):
        for x in range(NX):
            s = 0.0
            wsum = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx + NX) % NX
                    ny = (y + dy + NY) % NY
                    rx = -dx
                    ry = -dy
                    dot = cellVel_ux[ny, nx] * rx + cellVel_uy[ny, nx] * ry
                    if dot > 0:
                        wsum += dot
                        s += dot * airTemp[ny, nx]
            if wsum > 0:
                t_inc_average[y, x] = s / wsum
            else:
                t_inc_average[y, x] = airTemp[y, x]
                
    # (3) Compute t_inc as blend of t_inc_average and previous air temperature.
    t_inc = np.empty((NY, NX), dtype=np.float32)
    for y in range(NY):
        for x in range(NX):
            t_prior = airTemp[y, x]
            alpha = vc * cellVavg[y, x]
            if alpha > 1:
                alpha = 1.0
            t_inc[y, x] = alpha * t_inc_average[y, x] + (1 - alpha) * t_prior
            
    # (4) Compute t_avg_adj: average of t_inc from adjacent neighbors.
    t_avg_adj = np.empty((NY, NX), dtype=np.float32)
    for y in range(NY):
        for x in range(NX):
            s = 0.0
            cnt = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx + NX) % NX
                    ny = (y + dy + NY) % NY
                    s += t_inc[ny, nx]
                    cnt += 1
            t_avg_adj[y, x] = s / cnt
            
    # (5) Thermal update: compute forcing S and update temperature.
    S_array = np.empty((NY, NX), dtype=np.float32)
    t_final_array = np.empty((NY, NX), dtype=np.float32)
    for y in range(NY):
        for x in range(NX):
            t_ground = baseMap[y, x] + editMap[y, x] + seasonalBlend * seasonalMap[y, x]
            inc_val = t_inc[y, x]
            avg_adj = t_avg_adj[y, x]
            # d_diff is computed but not directly used here (could be added if desired)
            t_diff = k_const * (t_ground - inc_val)
            S_val = t_diff * dpdt
            if S_val > d_max:
                S_val = d_max
            elif S_val < -d_max:
                S_val = -d_max
            S_array[y, x] = S_val
            t_final_array[y, x] = inc_val + t_diff
            
    # (6) Collision step.
    for y in range(NY):
        for x in range(NX):
            for i in range(9):
                f[y, x, i] = f[y, x, i] - (f[y, x, i] - feq[y, x, i]) / tau + S_array[y, x] * weights[i]
                
    # (7) Streaming step.
    for y in range(NY):
        for x in range(NX):
            for i in range(9):
                nx = (x + int(ex[i]) + NX) % NX
                ny = (y + int(ey[i]) + NY) % NY
                fTemp[ny, nx, i] = f[y, x, i]
    # Copy fTemp back to f.
    for y in range(NY):
        for x in range(NX):
            for i in range(9):
                f[y, x, i] = fTemp[y, x, i]
                
    # (8) Damping step.
    for y in range(NY):
        for x in range(NX):
            for i in range(9):
                f[y, x, i] = weights[i] + (f[y, x, i] - weights[i]) * (1 - damping * dtStep)
                
    # (9) Update air temperature.
    for y in range(NY):
        for x in range(NX):
            airTemp[y, x] = t_final_array[y, x]
            
    return S_array  # return forcing field for visualization if desired

# -------------------------------
# Particle System (Tracer Particles)
# -------------------------------
particleLifetime = 15.0
speedFactor = 200.0

class Particle:
    def __init__(self, width, height):
        self.x = random.uniform(0, width)
        self.y = random.uniform(0, height)
        self.age = 0.0
        self.trail = []  # list of (x, y, t)

def cell_velocity(f, i, j, ex, ey):
    rho = 0.0
    ux = 0.0
    uy = 0.0
    for k in range(9):
        val = f[j, i, k]
        rho += val
        ux += val * ex[k]
        uy += val * ey[k]
    return (ux / rho, uy / rho)

def get_velocity_at(f, gx, gy, NX, NY, ex, ey):
    i = int(math.floor(gx))
    j = int(math.floor(gy))
    di = gx - i
    dj = gy - j
    i = max(0, min(i, NX - 2))
    j = max(0, min(j, NY - 2))
    v00 = cell_velocity(f, i, j, ex, ey)
    v10 = cell_velocity(f, i + 1, j, ex, ey)
    v01 = cell_velocity(f, i, j + 1, ex, ey)
    v11 = cell_velocity(f, i + 1, j + 1, ex, ey)
    ux = (1 - di) * (1 - dj) * v00[0] + di * (1 - dj) * v10[0] + (1 - di) * dj * v01[0] + di * dj * v11[0]
    uy = (1 - di) * (1 - dj) * v00[1] + di * (1 - dj) * v10[1] + (1 - di) * dj * v01[1] + di * dj * v11[1]
    return ux, uy

particles = []
particleSpawnAccumulator = 0.0
simTime = 0.0

def update_particles(dt, f):
    global simTime, particleSpawnAccumulator, particles
    simTime += dt
    for p in particles[:]:
        p.trail.append((p.x, p.y, simTime))
        # Remove trail points older than tailLength
        while p.trail and simTime - p.trail[0][2] > tailLength:
            p.trail.pop(0)
        # Convert particle position from pixels to grid coordinates:
        gx = p.x / cellSize
        gy = p.y / cellSize
        vx, vy = get_velocity_at(f, gx, gy, NX, NY, ex, ey)
        p.x += vx * speedFactor * dt
        p.y += vy * speedFactor * dt
        # Wrap around the screen:
        if p.x < 0: p.x += NX * cellSize
        if p.y < 0: p.y += NY * cellSize
        if p.x >= NX * cellSize: p.x -= NX * cellSize
        if p.y >= NY * cellSize: p.y -= NY * cellSize
        p.age += dt
        if p.age > particleLifetime:
            particles.remove(p)
    particleSpawnAccumulator += dt * spawnRate
    while particleSpawnAccumulator >= 1:
        particles.append(Particle(NX * cellSize, NY * cellSize))
        particleSpawnAccumulator -= 1

# -------------------------------
# Pygame Setup and Main Loop
# -------------------------------
pygame.init()
screen = pygame.display.set_mode((NX * cellSize, NY * cellSize))
pygame.display.set_caption("LBM Fluid Simulation – Thermal Forcing")
clock = pygame.time.Clock()

# Toggle variables:
paused = False
showTemperature = False
showDivergence = False

# For displaying simulation steps (not strictly necessary)
simStepCount = 0
lastS = np.zeros((NY, NX), dtype=np.float32)

running = True
while running:
    dt = clock.tick(60) / 1000.0  # seconds elapsed
    # Event handling:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:  # toggle pause
                paused = not paused
            elif event.key == pygame.K_t:  # toggle temperature view
                showTemperature = not showTemperature
                if showTemperature:
                    showDivergence = False
            elif event.key == pygame.K_d:  # toggle divergence view
                showDivergence = not showDivergence
                if showDivergence:
                    showTemperature = False
            elif event.key == pygame.K_r:  # reset simulation
                for y in range(NY):
                    for x in range(NX):
                        for i in range(9):
                            f[y, x, i] = weights[i]
                airTemp.fill(0)
                particles = []
                simStepCount = 0

    # Update simulation if not paused:
    if not paused:
        dtStep = dt / stepsPerFrame
        for _ in range(stepsPerFrame):
            lastS = stepLBM(f, fTemp, airTemp, baseMap, editMap, seasonalMap,
                            NX, NY, ex, ey, weights, tau, damping, dtStep,
                            D_const, k_const, dpdt, vc, d_max, seasonalBlend)
            simStepCount += 1
        update_particles(dt, f)

    # Drawing:
    # Create a surface for the grid (each cell drawn as a rectangle)
    for y in range(NY):
        for x in range(NX):
            rect = (x * cellSize, y * cellSize, cellSize, cellSize)
            if showTemperature:
                T = airTemp[y, x]
                col = int(((T + 0.02) / 0.04) * 255)
            elif showDivergence:
                S_val = lastS[y, x]
                col = int(((S_val + 0.02) / 0.04) * 255)
            else:
                col = 0
            col = max(0, min(255, col))
            pygame.draw.rect(screen, (col, col, col), rect)
    # Draw particle trails:
    for p in particles:
        if len(p.trail) > 1:
            for i in range(len(p.trail) - 1):
                p1 = p.trail[i]
                p2 = p.trail[i + 1]
                # Skip if wrapping is an issue:
                if abs(p1[0] - p2[0]) > (NX * cellSize) / 2 or abs(p1[1] - p2[1]) > (NY * cellSize) / 2:
                    continue
                # Compute alpha based on age in the trail:
                alpha = int(255 * ((p1[2] - (simTime - tailLength)) / tailLength))
                color = (0, 255, 255, alpha)
                # Pygame doesn't support per-line alpha easily, so we draw a thin line.
                pygame.draw.line(screen, (0, 255, 255), (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])), 2)
        # Draw particle as a circle:
        pygame.draw.circle(screen, (0, 255, 255), (int(p.x), int(p.y)), 2)
    
    pygame.display.flip()

pygame.quit()
