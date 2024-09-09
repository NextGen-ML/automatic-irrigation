import pygame
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import time
from collections import defaultdict

MAX_PLANTS = 10


def is_close_to(val1, val2, tolerance):
    """Check if two values are within a certain distance."""
    return abs(val1 - val2) < tolerance


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
    watering, fertilizing, pruning = action
    config['watering'] += watering
    config['fertilizing'] += fertilizing
    config['pruning'] += pruning


class Plant:
    def __init__(self):
        self.water = 5
        self.sunlight = 5
        self.fertilizer = 0
        self.health = 5
        self.growth = 0

    def update(self, action):

        self.water = min(self.water + action[0], 10)
        self.fertilizer = min(self.fertilizer + action[1], 10)
        self.health = min(self.health + action[2], 10)

        if self.water > 3 and self.fertilizer > 1:
            self.growth = min(self.growth + 0.5, 10)
        else:
            self.health = max(self.health - 0.5, 0)

    def render(self, screen, x, y):
        color = (0, 255, 0) if self.health > 5 else (255, 0, 0)
        pygame.draw.circle(screen, color, (x, y), int(self.health * 5))


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

        if elapsed_time > 45000:
            running = False

        state = np.array([plant.health for plant in plants] + [plant.growth for plant in plants])
        action = agent.select_action(state)
        update_parameters(config, action)

        screen.fill((255, 255, 255))
        for i, plant in enumerate(plants):
            plant.update(action)
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
        clock.tick(30)
        interval_count += 1

    pygame.quit()
    save_and_plot_data(health_records, growth_records, reward_records)


class DummyAgent:
    def select_action(self, state):
        return [randint(-1, 1) for _ in range(3)]


config = {'watering': 0, 'fertilizing': 0, 'pruning': 0}
agent = DummyAgent()

run_simulation(config, agent)