import pygame
import pandas as pd
import matplotlib.pyplot as plt
from random import randint, choice
import numpy as np
import time

MAX_PLANTS = 5

# Weather conditions that can influence plant health
WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Rainy']

def initialize_plot():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    ax1.set_xlabel('Interval')
    ax1.set_ylabel('Plant Health')
    ax1.set_title('Plant Health Over Time')
    ax1.grid(True)

    ax2.set_xlabel('Interval')
    ax2.set_ylabel('Plant Growth')
    ax2.set_title('Plant Growth Over Time')
    ax2.grid(True)

    ax3.set_xlabel('Interval')
    ax3.set_ylabel('Rewards')
    ax3.set_title('Rewards Over Time')
    ax3.grid(True)

    plt.tight_layout()
    return fig, ax1, ax2, ax3

def update_plot(health_records, growth_records, reward_records, fig, ax1, ax2, ax3):
    ax1.plot(range(1, len(health_records) + 1), health_records, label='Health', marker='o')
    ax2.plot(range(1, len(growth_records) + 1), growth_records, label='Growth', marker='x')
    ax3.plot(range(1, len(reward_records) + 1), reward_records, label='Rewards', marker='s')

    for ax in (ax1, ax2, ax3):
        ax.relim()
        ax.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()

def save_and_plot_data(health_records, growth_records, reward_records):
    data = pd.DataFrame({
        'Interval': range(1, len(health_records) + 1),
        'Health': health_records,
        'Growth': growth_records,
        'Rewards': reward_records
    })
    data.to_csv('garden_simulation_data.csv', index=False)

def update_parameters(config, action):
    """Update garden conditions based on the agent's actions."""
    config['watering'] += action[0]  # Control only watering

class Plant:
    def __init__(self):
        self.water = 5
        self.sunlight = 5
        self.fertilizer = 0
        self.health = 5
        self.growth = 0
        self.moisture_level = 5  # New factor
        self.nitrogen_level = 5   # New factor
        self.chlorophyll_index = 5 # New factor
        self.weather = choice(WEATHER_CONDITIONS)  # Random initial weather

    def update(self, action):
        # Water control only
        self.water = min(self.water + action[0], 10)

        # Health decrease for too much or too little water
        if self.water > 7 and self.nitrogen_level > 1:  # Too much water
            self.health = max(self.health - 0.5, 0)
            self.chlorophyll_index = max(self.chlorophyll_index - 0.5, 0)
        elif self.water < 3 or self.nitrogen_level < 1:  # Too little water
            self.health = max(self.health - 0.5, 0)
            self.chlorophyll_index = max(self.chlorophyll_index - 0.5, 0)

        # Update growth based on moisture, nitrogen, and weather
        if self.water > 3 and self.nitrogen_level > 1 and self.chlorophyll_index > 3:
            self.growth = min(self.growth + 0.5, 10)
        else:
            self.health = max(self.health - 0.5, 0)
            self.chlorophyll_index = max(self.chlorophyll_index - 0.5, 0)

        # Weather influence on health
        if self.weather == 'Sunny':
            self.water = min(self.water - 0.1, 10)
            self.chlorophyll_index += 0.5
        elif self.weather == 'Rainy':
            self.water = min(self.water + 0.5, 10)
        elif self.weather == 'Cloudy':
            self.health = max(self.health - 1, 0)

        self.chlorophyll_index -= 0.1
        self.water -= 0.1

    def render(self, screen, x, y):
        if self.health > 5:
            color = (0, 255, 0)
        elif self.health > 3:
            color = (255, 150, 0)
        else:
            color = (255, 0, 0)
        pygame.draw.circle(screen, color, (x, y), int(self.health * 5))

    def update_weather(self):
        """Randomly change the weather every hour (simulation)."""
        self.weather = choice(WEATHER_CONDITIONS)

def run_simulation(config, agent):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Garden Simulation')
    clock = pygame.time.Clock()
    running = True
    start_time = pygame.time.get_ticks()

    fig, ax1, ax2, ax3 = initialize_plot()

    plants = [Plant() for _ in range(MAX_PLANTS)]
    interval_count = 0
    total_reward = 0
    health_records = []
    growth_records = []
    reward_records = []

    while running:
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - start_time

        if elapsed_time > 10000:  # Run simulation for 10 seconds (equivalent to 10 hours)
            running = False

        state = np.array([plant.health for plant in plants] +
                         [plant.growth for plant in plants] +
                         [plant.moisture_level for plant in plants] +
                         [plant.nitrogen_level for plant in plants] +
                         [plant.chlorophyll_index for plant in plants])
        action = agent.select_action(state)
        update_parameters(config, action)

        screen.fill((255, 255, 255))
        for i, plant in enumerate(plants):
            plant.update(action)
            plant.update_weather()  # Update weather condition
            plant.render(screen, 100 + i * 100, 300)

        total_health = sum(plant.health for plant in plants) / MAX_PLANTS
        total_growth = sum(plant.growth for plant in plants) / MAX_PLANTS
        reward = total_health + total_growth
        total_reward += reward

        health_records.append(total_health)
        growth_records.append(total_growth)
        reward_records.append(reward)

        update_plot(health_records, growth_records, reward_records, fig, ax1, ax2, ax3)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()
        clock.tick(30)  # 30 FPS
        interval_count += 1

    pygame.quit()
    save_and_plot_data(health_records, growth_records, reward_records)

class DummyAgent:
    def select_action(self, state):
        return [randint(-2, 2)]  # Control only watering

config = {'watering': 0}
agent = DummyAgent()

run_simulation(config, agent)