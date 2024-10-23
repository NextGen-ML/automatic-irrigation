from random import randint, choice
from env import WEATHER_CONDITIONS
import pygame

class Plant:
    def __init__(self):
        self.water = 5
        self.health = 5
        self.moisture_level = 5  # New factor
        self.weather = choice(WEATHER_CONDITIONS)  # Random initial weather

    def update(self, action):
        # Water control only
        self.water = min(self.water + action[0], 10)

        # Health decrease for too much or too little water
        if self.water > 7:  # Too much water
            self.health = max(self.health - 0.5, 0)
        elif self.water < 3:  # Too little water
            self.health = max(self.health - 0.5, 0)
        else:
            self.health = max(self.health + 0.5, 0)

        # Weather influence on health
        if self.weather == 'Sunny':
            self.water = min(self.water - 0.1, 10)
        elif self.weather == 'Rainy':
            self.water = min(self.water + 0.5, 10)
        elif self.weather == 'Cloudy':
            self.health = max(self.health - 1, 0)

    def render(self, screen, x, y):
        if self.health > 5:
            color = (0, 255, 0)
        elif self.health > 3:
            color = (255, 150, 0)
        else:
            color = (255, 0, 0)
        pygame.draw.circle(screen, color, (x, y), int(25))

    def update_weather(self):
        """Randomly change the weather every hour (simulation)."""
        self.weather = choice(WEATHER_CONDITIONS)