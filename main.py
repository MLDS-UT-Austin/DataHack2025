import math
import pickle

import numba
import numpy as np
import pygame
from identify_hurricanes import identify_hurricanes
from numba.typed import List

from engine import NX, NY, LBMEngine, cell_size, engine_spec

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

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            self.step()

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

while True:
    for _ in range(100):
        sim.step()
    # sim.save_state("state.pkl")

# class DualLogger:
#     def __init__(self, data_log_file="logs/data.log", output_log_file="data/output.log"):
#         self.data_logger = self._setup_logger("data_logger", data_log_file, "%(asctime)s - %(levelname)s - %(message)s")
#         self.output_logger = self._setup_logger("output_logger", output_log_file, "%(message)s")
#     def _setup_logger(self, name, log_file, format):
#         logger = logging.getLogger(name)
#         logger.setLevel(logging.INFO)
#         # Create file handler
#         file_handler = logging.FileHandler(log_file)
#         file_handler.setLevel(logging.INFO)
#         # Create formatter and add it to the handlers
# formatter = logging.Formatter()
#         file_handler.setFormatter(formatter)

#         # Add the handlers to the logger
#         logger.addHandler(file_handler)

#         return logger

#     def log_data(self, message):
#         """
#         Logs a data message.

#         Args:
#             message (str): The message to log.
#         """
#         self.data_logger.info(message)

#     def log_output(self, message):
#         """
#         Logs an output message.

#         Args:
#             message (str): The message to log.
#         """
#         self.output_logger.info(message)

#     def close_handlers(self):
#         """Closes all handlers from both loggers"""
#         for handler in self.data_logger.handlers[:]:
#             handler.close()
#             self.data_logger.removeHandler(handler)
#         for handler in self.output_logger.handlers[:]:
#             handler.close()
#             self.output_logger.removeHandler(handler)
