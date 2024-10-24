from random import randint, choice

import numpy as np
import pygame

class PlantEnv:
    def __init__(self):
        self.water = 5
        self.health = 5
        self.moisture_level = 5

    def reset(self):
        self.water = 5
        self.health = 5
        self.moisture_level = 5

    def step(self, action):
        # Store the previous health to calculate the difference
        previous_health = self.health

        # Water control only
        self.moisture_level = min(self.moisture_level + action[0], 10)
        self.water += self.moisture_level * 0.1

        # Health decrease for too much or too little water
        if self.water > 7:  # Too much water
            self.health = max(self.health - 0.5, 0)
        elif self.water < 4:  # Too little water
            self.health = max(self.health - 0.5, 0)
        else:
            self.health = max(self.health + 0.5, 0)

        # Updates in the environment after each update
        self.water -= 0.5
        self.moisture_level -= 0.5

        # Calculate the reward based on the change in health
        reward = self.health - previous_health
        return self.get_state(), reward

    def render(self, screen, x, y):
        if self.health > 5:
            color = (0, 255, 0)
        elif self.health > 3:
            color = (255, 150, 0)
        else:
            color = (255, 0, 0)
        pygame.draw.circle(screen, color, (x, y), int(25))

    def get_state(self):
        return np.array([
            self.moisture_level,
            self.water
        ])