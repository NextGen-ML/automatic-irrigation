from random import randint, choice
import numpy as np
import pygame
import random

WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Rainy']

# Function to introduce a 10% margin of error for weather predictions
def introduce_error(weather_conditions, error_rate=0.1):
    modified_conditions = []
    for condition in weather_conditions:
        # Introduce a 10% chance to change the condition
        if random.random() < error_rate:
            # Select a different condition randomly
            new_condition = random.choice([c for c in WEATHER_CONDITIONS if c != condition])
            modified_conditions.append(new_condition)
        else:
            modified_conditions.append(condition)
    return modified_conditions

class PlantEnv:
    def __init__(self):
        self.water = 5
        self.health = 5
        self.moisture_level = 5
        self.water_usage = 0  # Track water usage
        self.current_weather = random.sample(WEATHER_CONDITIONS, 3)  # Start with three random weather conditions
        self.predicted_weather = introduce_error(self.current_weather)  # Array with 10% error margin

    def reset(self):
        self.water = 5
        self.health = 5
        self.moisture_level = 5
        self.water_usage = 0  # Reset water usage each episode
        self.current_weather = random.sample(WEATHER_CONDITIONS, 3)  # Reset weather conditions
        self.predicted_weather = introduce_error(self.current_weather)  # Reset predicted weather

    def step(self, action):

        previous_health = self.health

        # Update moisture level and track water usage
        self.moisture_level = min(self.moisture_level + action[0], 10)
        self.water_usage = action[0]
        self.water += self.moisture_level * 0.1

        self.moisture_level -= self.moisture_level * 0.15

        weather_effect = self.get_weather_effect()
        if weather_effect == 'Sunny':
            self.water = max(self.water - 1, 0)
        elif weather_effect == 'Cloudy':
            self.health = max(self.health - 0.5, 0)
        elif weather_effect == 'Rainy':
            self.health = max(self.health + 0.5, 0)

        # Health decrease for too much or too little water
        if self.water > 7:  # Too much water
            self.health = max(self.health - 0.5, 0)
        elif self.water < 3:  # Too little water
            self.health = max(self.health - 0.5, 0)
        elif 4.5 < self.water < 5.5:
            self.health = max(self.health + 1, 0)
        else:
            self.health = max(self.health + 0.5, 0)

        # Calculate the reward based on the change in health
        reward = (self.health - previous_health) - (self.water_usage * 0.2)

        # Update weather conditions after each step
        self.update_weather_conditions()

        return self.get_state(), reward, self.water_usage

    def update_weather_conditions(self):
        # Remove the first weather condition and append a new random weather condition
        self.current_weather.pop(0)  # Remove the first condition
        new_weather = random.choice(WEATHER_CONDITIONS)  # Get a new random weather condition
        self.current_weather.append(new_weather)  # Append the new weather condition

        # Update predicted weather based on the new current weather
        self.predicted_weather = introduce_error(self.current_weather)  # Update predicted weather

    def render(self, screen, x, y):
        # Render the plant with color based on health
        if self.health > 5:
            color = (0, 255, 0)
        elif self.health > 3:
            color = (255, 150, 0)
        else:
            color = (255, 0, 0)
        pygame.draw.circle(screen, color, (x, y), int(25))

    def get_state(self):
        # Include moisture level, water, and arrays for current and predicted weather
        print("p:", self.predicted_weather)
        state = np.array([
            self.moisture_level,
            self.water,
            *self.encode_weather(self.predicted_weather)
        ])
        return state

    def encode_weather(self, weather_conditions):
        # Encode weather conditions into numerical values
        return [1 if condition == 'Sunny' else 0 for condition in weather_conditions] + \
               [1 if condition == 'Cloudy' else 0 for condition in weather_conditions] + \
               [1 if condition == 'Rainy' else 0 for condition in weather_conditions]

    def get_weather_effect(self):
        # For simplicity, take the first weather condition as the current weather effect
        print("c:", self.current_weather)
        print("-")
        return self.current_weather[0]

# Example usage
env = PlantEnv()
state, reward, water_usage = env.step([1])  # Action can be [1] for watering