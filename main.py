import math
import pickle

import numba
import numpy as np
import pygame
from identify_hurricanes import identify_hurricanes
from numba.typed import List

from engine import NX, NY, LBMEngine, cell_size, engine_spec
clock = pygame.time.Clock()


import csv  # add at the top if not already imported

# Global logging lists:
hurricane_log = []   # each record: [sim_step, hurricane_id, center_x, center_y, avg_velocity, max_velocity]
station_log = []     # each record: [sim_step, city_name, pressure, air_temp, ground_temp, vel_x, vel_y]

# Global hurricane tracking (to keep the same id across timesteps)
tracked_hurricanes = {}  # mapping: hurricane_id -> (last_center_y, last_center_x)
hurricane_id_counter = 0


CITIES = [
    dict(name="Sparseville", x=63, y=35),
    dict(name="Tensorburg", x=214, y=378),
    dict(name="Bayes Bay", x=160, y=262),
    dict(name="ReLU Ridge", x=413, y=23),
    dict(name="GANopolis", x=318, y=132),
    dict(name="Gradient Grove", x=468, y=158),
    dict(name="Offshore A", x=502, y=356),
    dict(name="Offshore B", x=660, y=184),
]

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
base_year = 1955


class Visualizer:
    def __init__(self, sim):
        self.engine = sim
        pygame.init()
        self.screen = pygame.display.set_mode(
            (NX * cell_size, NY * cell_size), depth=32
        )
        pygame.display.set_caption("LBM Fluid Simulation – Python Port")
        self.trail_length = 5

    def step(self):
        self.engine.step()
        self.draw()
        self.log_data() # log every step (log_data() itself only acts every 10 steps)

    def run(self):
        clock = pygame.time.Clock()
        # Set auto-save interval in milliseconds (e.g., 60000ms = 60 seconds)
        save_interval = 60000  
        last_save_time = pygame.time.get_ticks()

        while True:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:  # manual save on 's' key
                        save_logs_to_csv()
                        print("Manual save triggered.")

            # Automatic save after the specified interval
            current_time = pygame.time.get_ticks()
            if current_time - last_save_time >= save_interval:
                save_logs_to_csv()
                last_save_time = current_time
                print("Automatic save triggered.")

            # Run simulation step and drawing
            self.step()

            # Limit frame rate (60 FPS should be plenty)
            clock.tick(60)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return self.engine.__getattribute__(name)

    def draw(self):
        # Draw land map (blue for water, green for land)
        seasonal_scale = (
            -0.5 * math.cos(self.current_month / 12 * math.pi * 2) + 0.5
        )  # 0 at month 0, 1 at month 6, 0 at month 12
        t_ground = self.base_map + seasonal_scale * self.seasonal_map

        for j in range(NY):
            for i in range(NX):
                # Compute temperature value
                T = t_ground[j, i]
                # T =  seasonal_effects[int(current_month % 12)] * seasonal_map[j, i]
                scale = (T + 0.02) / 0.04
                scale = max(0, min(1, scale))

                if self.land_map[j, i] == 0:
                    color = (0, 0, 255 * scale)  # Blue for water
                else:
                    color = (0, 255 * scale, 0)  # Blue for water

                pygame.draw.rect(
                    self.screen,
                    color,
                    (i * cell_size, j * cell_size, cell_size, cell_size),
                )

        h_centers, h_sizes, h_indicator = self.identify_hurricanes(0.02)

        # draw h_indicator grid
        # for j in range(NY):
        #     for i in range(NX):
        #         scale = (h_indicator[j, i] + 0.02) / 0.04
        #         scale = max(0, min(1, scale))
        #         color = (100 * scale, 100 * scale, 100 * scale)
        #         pygame.draw.rect(
        #             self.screen,
        #             color,
        #             (i * cell_size, j * cell_size, cell_size, cell_size),
        #         )

        for (h_y, h_x), h_radius in zip(h_centers, h_sizes):
            pygame.draw.circle(
                self.screen,
                (0, 255, 0),
                (int(h_x * cell_size), int(h_y * cell_size)),
                int(h_radius * cell_size),
                1,
            )

        # Draw particle trails (white lines)
        for p in self.particles:
            if len(p[3]) > 1:
                for start in range(
                    max(0, len(p[3]) - self.trail_length), len(p[3]) - 1
                ):
                    x0, y0 = p[3][start]
                    x1, y1 = p[3][start + 1]
                    if (
                        abs(x0 - x1) < NX * cell_size / 2
                        and abs(y0 - y1) < NY * cell_size / 2
                    ):
                        pygame.draw.line(
                            self.screen,
                            (190, 190, 190),
                            (int(x0), int(y0)),
                            (int(x1), int(y1)),
                            1,
                        )

        # Draw particles (white dots)
        # for p in self.particles:
        #     x, y = p[0], p[1]
        #     pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), 2)

        # Draw cities
        for city in CITIES:
            x = city["x"]
            y = city["y"]
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 5)
            font = pygame.font.Font(None, 24)
            text = font.render(city["name"], True, (255, 255, 255))
            self.screen.blit(text, (x + 10, y - 10))

        yearMonthStr = (
            f"{MONTHS[int(self.current_month)]}, {base_year + self.current_year}"
        )
        font = pygame.font.Font(None, 24)
        text = font.render(yearMonthStr, True, (255, 255, 255))
        self.screen.blit(text, (0, 0))

        pygame.display.flip()

    def save_state(self, filename):
        def recursive_convert(obj):
            if isinstance(obj, numba.typed.typedlist.List):
                return [recursive_convert(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(recursive_convert(x) for x in obj)
            return obj

        attributes = [x[0] for x in engine_spec]
        state = {attr: recursive_convert(getattr(self, attr)) for attr in attributes}

        with open(filename, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, filename):
        with open(filename, "rb") as f:
            state = pickle.load(f)

        for _ in range(len(self.particles)):
            del self.particles[0]

        for p in state["particles"]:
            from numba import types

            trail = List.empty_list(types.UniTuple(types.float64, 2))
            for t in p[3]:
                trail.append(t)
            self.particles.append((p[0], p[1], p[2], trail))

        del state["particles"]

        for attr, value in state.items():
            setattr(self.engine, attr, value)

    def log_data(self):
        # Only log every 10 simulation steps.
        if self.engine.sim_step_count % 10 != 0:
            return

        global hurricane_log, station_log, tracked_hurricanes, hurricane_id_counter
        step = self.engine.sim_step_count

        ### Log Hurricanes ###
        # Get hurricane centers and sizes from the engine.
        h_centers, h_sizes, _ = self.engine.identify_hurricanes()
        # Get the velocity field.
        x_vel, y_vel = self.engine.get_velocity_field()
        # Compute the magnitude field.
        mag = np.sqrt(x_vel**2 + y_vel**2)

        for center, size in zip(h_centers, h_sizes):
            # center is (row, col) => (y, x) in grid coordinates
            y_center, x_center = center
            # Use the hurricane "size" (rounded) as an approximate radius (in grid cells)
            radius = int(size)
            # Determine a bounding box (make sure we stay within bounds)
            y_min = max(0, y_center - radius)
            y_max = min(mag.shape[0], y_center + radius + 1)
            x_min = max(0, x_center - radius)
            x_max = min(mag.shape[1], x_center + radius + 1)
            region = mag[y_min:y_max, x_min:x_max]
            avg_vel = np.mean(region) if region.size > 0 else 0
            max_vel = np.max(region) if region.size > 0 else 0

            # Track the hurricane to assign a persistent id.
            assigned_id = None
            for hid, last_center in tracked_hurricanes.items():
                # Compute distance in grid cells.
                dist = np.sqrt((last_center[0] - y_center)**2 + (last_center[1] - x_center)**2)
                if dist < 3:  # threshold distance; adjust as needed
                    assigned_id = hid
                    break
            if assigned_id is None:
                hurricane_id_counter += 1
                assigned_id = hurricane_id_counter
            # Update the tracking dictionary with the current center.
            tracked_hurricanes[assigned_id] = (y_center, x_center)
            # Append the record: [step, hurricane_id, center_x, center_y, avg_velocity, max_velocity]
            hurricane_log.append([step, assigned_id, x_center, y_center, avg_vel, max_vel])

        ### Log Stations (Cities) ###
        for city in CITIES:
            # City coordinates are given in pixels; convert to grid indices.
            grid_x = int(city["x"] // cell_size)
            grid_y = int(city["y"] // cell_size)
            # Pressure is approximated as the density (sum over the 9 distributions)
            pressure = np.sum(self.engine.sim_state[grid_y, grid_x, :])
            air_temp = self.engine.air_temp[grid_y, grid_x]
            # Compute ground temperature similar to your draw() method:
            seasonal_scale = -0.5 * math.cos(self.engine.current_month / 12 * math.pi * 2) + 0.5
            ground_temp = self.engine.base_map[grid_y, grid_x] + seasonal_scale * self.engine.seasonal_map[grid_y, grid_x]
            # Get velocity at the station:
            vel_x, vel_y = self.engine.get_velocity_at(grid_x, grid_y)
            # Append the record: [step, city_name, pressure, air_temp, ground_temp, vel_x, vel_y]
            station_log.append([step, city["name"], pressure, air_temp, ground_temp, vel_x, vel_y])
   



def save_logs_to_csv():
    global hurricane_log, station_log
    with open("hurricanes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "hurricane_id", "center_x", "center_y", "avg_velocity", "max_velocity"])
        writer.writerows(hurricane_log)
    with open("stations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "city", "pressure", "air_temp", "ground_temp", "velocity_x", "velocity_y"])
        writer.writerows(station_log)
    print("Logs saved to hurricanes.csv and stations.csv")



# list(Tuple(float64, float64, float64, reflected list(UniTuple(float64 x 2))<iv=None>))
# ListType[Tuple(float64, float64, float64, ListType[UniTuple(float64 x 2)])]:


# Load land map outside of the jitclass
land_map = np.load("land_map.npy")
# base_map = np.load("base_map.npy")
seasonal_map = np.load("seasonal_map.npy")

engine = LBMEngine(land_map, seasonal_map)

sim = Visualizer(engine)
# sim.load_state("state.pkl")
sim.engine.viscosity = 0.08
sim.engine.tau = sim.engine.viscosity * 3 + 0.5



sim = Visualizer(engine)
sim.engine.viscosity = 0.08
sim.engine.tau = sim.engine.viscosity * 3 + 0.5
sim.run()

#while True:
#    for _ in range(100):
#        sim.step()
    # sim.save_state("state.pkl")


