import math
import pickle

import numpy as np
import pygame

from engine import NX, NY, LBMEngine, cell_size
from identify_hurricanes import identify_hurricanes

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


class Visualizer:
    def __init__(self, sim):
        self.sim = sim
        pygame.init()
        self.screen = pygame.display.set_mode(
            (NX * cell_size, NY * cell_size), depth=32
        )
        pygame.display.set_caption("LBM Fluid Simulation – Python Port")
        self.trail_length = 5

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            self.sim.step()
            self.draw()

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return self.sim.__getattribute__(name)

    def draw(self):

        # Draw land map (blue for water, green for land)
        if self.current_month % 12 == 0:
            self.current_year = self.current_year + 1
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

        x_vel, y_vel = self.get_velocity_field()
        h_centers, h_sizes, h_indicator = identify_hurricanes(
            x_vel, y_vel, threshold=0.01
        )

        # h_indicator /= 0.01

        # # draw h_indicator grid
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
            h_radius *= 200
            # print(h_x, h_y, h_radius)
            pygame.draw.circle(
                self.screen,
                (0, 255, 0),
                (int(h_x * cell_size), int(h_y * cell_size)),
                int(h_radius * cell_size),
                1,
            )

        # # Draw particle trails (green lines)
        # for p in self.particles:
        #     if len(p[3]) > 1:
        #         for start in range(
        #             max(0, len(p[3]) - self.trail_length), len(p[3]) - 1
        #         ):
        #             x0, y0 = p[3][start]
        #             x1, y1 = p[3][start + 1]
        #             if (
        #                 abs(x0 - x1) < NX * cell_size / 2
        #                 and abs(y0 - y1) < NY * cell_size / 2
        #             ):
        #                 pygame.draw.line(
        #                     self.screen,
        #                     (190, 190, 190),
        #                     (int(x0), int(y0)),
        #                     (int(x1), int(y1)),
        #                     1,
        #                 )

        # Draw particles (white dots)
        for p in self.particles:
            x, y = p[0], p[1]
            pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), 2)

        # Draw cities
        for city in CITIES:
            x = city["x"]
            y = city["y"]
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 5)
            font = pygame.font.Font(None, 24)
            text = font.render(city["name"], True, (255, 255, 255))
            self.screen.blit(text, (x + 10, y - 10))

        pygame.display.flip()


# Load land map outside of the jitclass
land_map = np.load("land_map.npy")
# base_map = np.load("base_map.npy")
seasonal_map = np.load("seasonal_map.npy")

engine = LBMEngine(land_map, seasonal_map)

sim = Visualizer(engine)
sim.run()

# current_base_map = engine.base_map

# # np.save("seasonal_map.npy", current_base_map)

# import matplotlib.pyplot as plt

# # Plot the base_map and current_base_map
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(seasonal_map, cmap='coolwarm')
# plt.colorbar(label='Temperature')
# plt.title('seasonal_map')

# plt.subplot(1, 2, 2)
# plt.imshow(current_base_map, cmap='coolwarm')
# plt.colorbar(label='Temperature')
# plt.title('Current Base Map')

# plt.tight_layout()
# plt.savefig('temperature_maps.png')
# plt.show()


# Initialize the visualizer with the engine
# sim = Visualizer(engine)
# sim.run()


# sim = Visualizer(LBMEngine(land_map))
# sim.run()
