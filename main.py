import pygame
import pandas as pd
import matplotlib.pyplot as plt
from random import randint, choice
import numpy as np
import time
from plant import Plant
from agent import DummyAgent
from env import WEATHER_CONDITIONS

MAX_PLANTS = 5

def initialize_plot():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.set_xlabel('Interval')
    ax1.set_ylabel('Plant Health')
    ax1.set_title('Plant Health Over Time')
    ax1.grid(True)

    ax2.set_xlabel('Interval')
    ax2.set_ylabel('Rewards')
    ax2.set_title('Rewards Over Time')
    ax2.grid(True)

    plt.tight_layout()
    return fig, ax1, ax2

def update_plot(health_records, reward_records, fig, ax1, ax2):
    ax1.plot(range(1, len(health_records) + 1), health_records, label='Health', marker='o')
    ax2.plot(range(1, len(reward_records) + 1), reward_records, label='Rewards', marker='s')

    for ax in (ax1, ax2):
        ax.relim()
        ax.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()


def save_and_plot_data(health_records, reward_records):
    data = pd.DataFrame({
        'Interval': range(1, len(health_records) + 1),
        'Health': health_records,
        'Rewards': reward_records
    })
    data.to_csv('garden_simulation_data.csv', index=False)

def update_parameters(config, action):
    """Update garden conditions based on the agent's actions."""
    config['watering'] += action[0]  # Control only watering


def run_simulation(config, agent):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Garden Simulation')
    clock = pygame.time.Clock()
    running = True
    start_time = pygame.time.get_ticks()

    fig, ax1, ax2 = initialize_plot()

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
                         [plant.moisture_level for plant in plants])
        action = agent.select_action(state)
        update_parameters(config, action)

        screen.fill((255, 255, 255))
        for i, plant in enumerate(plants):
            plant.update(action)
            plant.update_weather()  # Update weather condition
            plant.render(screen, 100 + i * 100, 300)

        total_health = sum(plant.health for plant in plants) / MAX_PLANTS
        reward = total_health
        total_reward += reward

        health_records.append(total_health)
        reward_records.append(reward)

        update_plot(health_records, reward_records, fig, ax1, ax2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()
        clock.tick(30)  # 30 FPS
        interval_count += 1

    pygame.quit()
    save_and_plot_data(health_records, reward_records)


config = {'watering': 0}
agent = DummyAgent()

run_simulation(config, agent)