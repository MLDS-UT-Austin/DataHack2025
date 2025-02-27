import math
import random

import numpy as np
from numba import float32, float64, int32, int64, types
from numba.experimental import jitclass
from numba.typed import List

from util import generate_perlin_noise, rescale

# Global Constants
# grid width (cells)
NX = 200
# grid height (cells)
NY = 100
# cell size (pixels)
cell_size = 4

# LBM lattice directions and weights (D2Q9)
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
weights = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float32,
)


spec = [
    ("sim_step_count", int32),
    ("current_month", float32),
    ("current_year", int32),
    ("dt", float32),
    (
        "particles",
        types.ListType(
            types.Tuple(
                (
                    float64,
                    float64,
                    float64,
                    types.ListType(types.Tuple((float64, float64))),
                )
            )
        ),
    ),
    ("particle_lifetime", float32),
    ("particle_spawn_acc", float32),
    ("global_wind", float32),
    ("viscosity", float32),
    ("tau", float32),
    ("damping", float32),
    ("noise_amplitude", float32),
    ("spawn_rate", float32),
    ("k_const", float32),
    ("dpdt", float32),
    ("vc", float32),
    ("d_max", float32),
    ("base_map", float64[:, :]),
    ("base_map_scale", float64),
    ("base_map_freq", float64),
    ("base_map_octaves", int64),
    ("base_map_alpha", float64),
    ("seasonal_map", float64[:, :]),
    ("land_map", int64[:, :]),
    ("air_temp", float32[:, :]),
    ("sim_state", float32[:, :, :]),
]


@jitclass(spec)
class LBMEngine:
    def __init__(
        self,
        land_map,
        seasonal_map,
        # base_map_scale=0.007,
        base_map_scale=0.001,
        base_map_freq=4.0,
        base_map_octaves=2,
        base_map_alpha=0.0001,
        dt=0.08,
        particle_lifetime=40.0,
        particle_spawn_acc=0.0,
        global_wind=-0.1,
        viscosity=0.04,
        damping=0.02,
        spawn_rate=100,
        k_const=0.0007,
        dpdt=30000.0,
        vc=1.0,
        d_max=0.005,
    ):

        self.sim_step_count = 0
        self.current_month = 0
        self.current_year = 0
        self.dt = dt
        trail_list = List.empty_list((0.0, 0.0))
        self.particles = List.empty_list((0.0, 0.0, 0.0, trail_list))

        self.particle_lifetime = particle_lifetime
        self.particle_spawn_acc = particle_spawn_acc
        self.global_wind = global_wind

        # Fluid simulation parameters
        self.viscosity = viscosity
        self.tau = viscosity * 3 + 0.5
        self.damping = damping
        self.spawn_rate = spawn_rate

        # Thermal parameters
        self.k_const = k_const
        self.dpdt = dpdt
        self.vc = vc
        self.d_max = d_max

        self.base_map_scale = base_map_scale
        self.base_map_freq = base_map_freq
        self.base_map_octaves = base_map_octaves
        self.base_map_alpha = base_map_alpha
        # self.base_map = (
        #     generate_perlin_noise(
        #         random.randint(0, 100000), (NY, NX), self.base_map_octaves, self.base_map_freq
        #     ).astype(np.float64)
        # )
        # self.base_map = rescale(self.base_map, -base_map_scale, base_map_scale)
        self.base_map = generate_perlin_noise(0.01, 0.05, (NY, NX))

        # code to generate seasonal_map
        self.seasonal_map = generate_perlin_noise(0.01, 0.02, (NY, NX))
        # self.seasonal_map = (
        #     generate_perlin_noise(random.randint(0, 100000), (NY, NX), 2, 1.5) * base_map_scale
        # )
        # self.seasonal_map = seasonal_map
        self.land_map = land_map

        self.air_temp = np.zeros((NY, NX), dtype=np.float32)
        self.sim_state = np.empty((NY, NX, 9), dtype=np.float32)
        # Initialize with equilibrium values (ρ = 1, u = 0)
        for j in range(NY):
            for i in range(NX):
                self.sim_state[j, i, :] = weights

    def step(self):
        for _ in range(5):
            self._step_lbm()
        self._step_particles()
        self._step_base_map()
        if self.current_month >= 12:
            self.current_year = self.current_year + 1
        self.current_month = (self.current_month + self.dt * 0.2) % 12
        # self.current_month = (self.current_month + self.dt * 0.5) % 12

    def _step_lbm(self):
        # (1) Compute macroscopic variables and equilibrium distribution
        feq = np.empty_like(self.sim_state)
        cell_vel = [[(0, 0) for _ in range(NX)] for _ in range(NY)]
        cell_vavg = np.zeros((NY, NX), dtype=np.float32)
        for j in range(NY):
            for i in range(NX):
                rho = np.sum(self.sim_state[j, i, :])
                mom_x = np.sum(self.sim_state[j, i, :] * ex)
                mom_y = np.sum(self.sim_state[j, i, :] * ey)
                ux = mom_x / rho
                uy = mom_y / rho
                cell_vel[j][i] = (ux, uy)
                cell_vavg[j, i] = (abs(ux) + abs(uy)) / 2
                u_sq = ux * ux + uy * uy
                for k in range(9):
                    e_dot_u = ex[k] * ux + ey[k] * uy
                    feq[j, i, k] = (
                        weights[k]
                        * rho
                        * (1 + 3 * e_dot_u + 4.5 * e_dot_u * e_dot_u - 1.5 * u_sq)
                    )
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
                            s_val += dot * self.air_temp[nj, ni]
                t_inc_average[j, i] = (
                    s_val / w_sum if w_sum > 0 else self.air_temp[j, i]
                )
        # (3) Compute t_inc via affine combination
        t_inc = np.empty((NY, NX), dtype=np.float32)
        for j in range(NY):
            for i in range(NX):
                alpha = min(1, self.vc * cell_vavg[j, i])
                t_inc[j, i] = (
                    alpha * t_inc_average[j, i] + (1 - alpha) * self.air_temp[j, i]
                )
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
                # Ground temperature uses base_map and seasonal effect
                seasonal_scale = (
                    -0.5 * math.cos(self.current_month / 12 * math.pi * 2) + 0.5
                )  # 0 at month 0, 1 at month 6, 0 at month 12
                t_ground = (
                    self.base_map[j, i] + seasonal_scale * self.seasonal_map[j, i]
                )
                inc = t_inc[j, i]
                t_diff = self.k_const * (t_ground - inc)
                S_val = t_diff * self.dpdt
                S_val = max(-self.d_max, min(self.d_max, S_val))
                S_array[j, i] = S_val
                t_final_array[j, i] = inc + t_diff

        # (6) Collision step
        for j in range(NY):
            for i in range(NX):
                for k in range(9):
                    self.sim_state[j, i, k] = (
                        self.sim_state[j, i, k]
                        - (self.sim_state[j, i, k] - feq[j, i, k]) / self.tau
                        + S_array[j, i] * weights[k]
                    )
        # (7) Streaming step (propagate distributions)
        f_temp = np.empty_like(self.sim_state)
        for j in range(NY):
            for i in range(NX):
                for k in range(9):
                    ni = (i + ex[k]) % NX
                    nj = (j + ey[k]) % NY
                    f_temp[nj, ni, k] = self.sim_state[j, i, k]
        self.sim_state[:, :, :] = f_temp
        # (8) Damping step
        for j in range(NY):
            for i in range(NX):
                for k in range(9):
                    self.sim_state[j, i, k] = weights[k] + (
                        self.sim_state[j, i, k] - weights[k]
                    ) * (1 - self.damping * self.dt)
        # (9) Update air temperature field
        self.air_temp[:, :] = t_final_array
        self.sim_step_count += 1

        # shift everything by the global wind
        previous_mod = int((self.sim_step_count - 1) * 5 * self.dt * self.global_wind % NX)
        current_mod = int(self.sim_step_count * 5 * self.dt * self.global_wind % NX)

        def shift_array(arr, shift):
            return np.concatenate((arr[:, -shift:], arr[:, :-shift]), axis=1)

        if previous_mod > current_mod:
            shift = np.int64(current_mod - previous_mod)
            self.sim_state = shift_array(self.sim_state, shift)
            self.air_temp = shift_array(self.air_temp, shift)
            self.base_map = shift_array(self.base_map, shift)
            self.seasonal_map = shift_array(self.seasonal_map, shift)

    def _step_particles(self):
        speed_factor = 200
        i = 0
        for _ in range(len(self.particles)):
            p = self.particles[i]
            x, y, dt, trail = p[0], p[1], p[2], p[3]
            trail.append((p[0], p[1]))
            gx = x / cell_size
            gy = y / cell_size
            vx, vy = self.get_velocity_at(gx, gy)
            x += vx * speed_factor * self.dt
            y += vy * speed_factor * self.dt
            # Periodic boundaries
            if x < 0:
                x += NX * cell_size
            if y < 0:
                y += NY * cell_size
            if x >= NX * cell_size:
                x -= NX * cell_size
            if y >= NY * cell_size:
                y -= NY * cell_size
            dt += self.dt
            if dt > self.particle_lifetime:
                del self.particles[i]
            else:
                self.particles[i] = (x, y, dt, trail)
                i+=1
        self.particle_spawn_acc += self.dt * self.spawn_rate
        while self.particle_spawn_acc >= 1:
            self.spawn_particle()
            self.particle_spawn_acc -= 1

    def _step_base_map(self):
        return
        new_noise = (
            generate_perlin_noise(
                random.randint(0, 100000), (NY, NX), self.base_map_octaves, self.base_map_freq
            ).astype(np.float64)
        )
        self.base_map = self.base_map * (1 - self.base_map_alpha) + new_noise * self.base_map_alpha
        self.base_map = rescale(self.base_map, -self.base_map_scale, self.base_map_scale)

    def get_velocity_at(self, gx, gy):
        # Bilinear interpolation of cell velocities
        i = int(math.floor(gx))
        j = int(math.floor(gy))
        di = gx - i
        dj = gy - j
        i = max(0, min(i, NX - 2))
        j = max(0, min(j, NY - 2))

        def cell_velocity(i, j):
            rho = np.sum(self.sim_state[j, i, :])
            ux = np.sum(self.sim_state[j, i, :] * ex)
            uy = np.sum(self.sim_state[j, i, :] * ey)
            return ux / rho, uy / rho

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
        ux += self.global_wind
        return ux, uy

    def get_velocity_field(self):
        x_vel = np.zeros((NY, NX), dtype=np.float32)
        y_vel = np.zeros((NY, NX), dtype=np.float32)

        def cell_velocity(i, j):
            rho = np.sum(self.sim_state[j, i, :])
            ux = np.sum(self.sim_state[j, i, :] * ex)
            uy = np.sum(self.sim_state[j, i, :] * ey)
            return ux / rho, uy / rho

        for j in range(NY):
            for i in range(NX):
                x_vel[j, i], y_vel[j, i] = cell_velocity(i, j)
        return x_vel, y_vel

    def spawn_particle(self):
        self.particles.append(
            (
                random.random() * NX * cell_size,
                random.random() * NY * cell_size,
                0.0,
                List.empty_list((0.0, 0.0)),
            )
        )
