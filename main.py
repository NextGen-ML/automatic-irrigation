import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from plant import PlantEnv
from agent import Agent
import random

def train_agent(episodes, max_steps=25, seed=42):
    # Clear previous training results
    open('training_results.csv', 'w').close()
    open('weather_actions.csv', 'w').close()

    # Write headers for the CSV files
    training_results_header = pd.DataFrame(columns=['Episode', 'Reward', 'Average_Reward', 'Epsilon', 'Loss', 'Water_Usage'])
    weather_actions_header = pd.DataFrame(columns=['Episode', 'Step', 'Weather_Prediction_Day1', 'Weather_Prediction_Day2', 'Weather_Prediction_Day3', 'Action'])
    training_results_header.to_csv('training_results.csv', index=False)
    weather_actions_header.to_csv('weather_actions.csv', index=False)

    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Plant DQN Training')
    clock = pygame.time.Clock()

    # Initialize plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))
    plt.ion()

    # Initialize tracking metrics
    episode_rewards = []
    average_rewards = []
    epsilon_history = []
    loss_history = []
    water_usage_history = []

    # Initialize environment and agent
    env = PlantEnv()
    state_size = len(env.get_state())
    action_size = 5
    agent = Agent(state_size=state_size, action_size=action_size)

    # Define a mapping for weather conditions
    weather_mapping = {
        "Sunny": -1,
        "Cloudy": -0.5,
        "Rainy": 0.5
    }

    for episode in range(episodes):
        env.reset()
        episode_reward = 0
        episode_water_usage = 0
        losses = []
        weather_actions_data = []  # Reset per episode

        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            state = env.get_state()
            action = agent.act(state)
            next_state, reward, water_used = env.step(action)
            episode_water_usage += water_used
            agent.remember(state, action, reward, next_state)

            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    losses.append(loss)

            episode_reward += reward

            # Collect weather prediction and action for saving
            weather_actions_data.append({
                "Episode": episode,
                "Step": step,
                "Weather_Prediction_Day1": weather_mapping.get(env.predicted_weather[0], None) if len(env.predicted_weather) > 0 else None,
                "Weather_Prediction_Day2": weather_mapping.get(env.predicted_weather[1], None) if len(env.predicted_weather) > 1 else None,
                "Weather_Prediction_Day3": weather_mapping.get(env.predicted_weather[2], None) if len(env.predicted_weather) > 2 else None,
                "Action": action[0]
            })

            screen.fill((255, 255, 255))
            env.render(screen, 400, 300)

            font = pygame.font.Font(None, 36)
            text = font.render(f'Episode: {episode + 1}', True, (0, 0, 0))
            screen.blit(text, (10, 10))
            text = font.render(f'Step: {step + 1}', True, (0, 0, 0))
            screen.blit(text, (10, 50))
            text = font.render(f'Epsilon: {agent.epsilon:.3f}', True, (0, 0, 0))
            screen.blit(text, (10, 90))

            pygame.display.flip()
            clock.tick(30)

        episode_rewards.append(episode_reward)
        average_rewards.append(np.mean(episode_rewards[-100:]))
        epsilon_history.append(agent.epsilon)
        water_usage_history.append(episode_water_usage)
        loss_history.append(np.mean(losses) if losses else 0)

        # Save training metrics after each episode
        training_data = pd.DataFrame([{
            'Episode': episode,
            'Reward': episode_reward,
            'Average_Reward': np.mean(episode_rewards[-100:]),
            'Epsilon': agent.epsilon,
            'Loss': np.mean(losses) if losses else 0,
            'Water_Usage': episode_water_usage
        }])
        training_data.to_csv('training_results.csv', mode='a', header=False, index=False)

        # Save weather predictions and actions data after each episode
        weather_actions_df = pd.DataFrame(weather_actions_data)
        weather_actions_df.to_csv('weather_actions.csv', mode='a', header=False, index=False)

        if episode % 10 == 0:
            ax1.clear()
            ax1.plot(episode_rewards, label='Episode Reward')
            ax1.plot(average_rewards, label='100-Episode Average')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Training Rewards')
            ax1.legend()
            ax1.grid(True)

            ax2.clear()
            ax2.plot(epsilon_history)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Epsilon')
            ax2.set_title('Exploration Rate')
            ax2.grid(True)

            ax3.clear()
            ax3.plot(loss_history)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Loss')
            ax3.set_title('Training Loss')
            ax3.grid(True)

            ax4.clear()
            ax4.plot(water_usage_history)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Water Usage')
            ax4.set_title('Water Usage per Episode')
            ax4.grid(True)

            plt.tight_layout()
            plt.pause(0.01)

    pygame.quit()
    return agent, pd.DataFrame(episode_rewards, columns=['Rewards'])


def test_agent(agent, episodes=3):
    """Test the trained agent"""
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Plant DQN Testing')
    clock = pygame.time.Clock()

    env = PlantEnv()
    test_rewards = []

    for episode in range(episodes):
        env.reset()
        episode_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            state = env.get_state()

            # Use greedy action selection during testing
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.policy_net(state_tensor)
                action_idx = q_values.argmax().item()
                action = np.array([agent.action_map[action_idx]])

            next_state, reward, water_usage = env.step(action)
            episode_reward += reward

            # Render
            screen.fill((255, 255, 255))
            env.render(screen, 400, 300)

            font = pygame.font.Font(None, 36)
            text = font.render(f'Test Episode: {episode + 1}', True, (0, 0, 0))
            screen.blit(text, (10, 10))
            text = font.render(f'Reward: {episode_reward:.2f}', True, (0, 0, 0))
            screen.blit(text, (10, 50))

            pygame.display.flip()
            clock.tick(30)

            if env.health <= 0:
                done = True

        test_rewards.append(episode_reward)
        # print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")

    pygame.quit()
    return np.mean(test_rewards)


if __name__ == "__main__":
    # Train the agent
    trained_agent, training_data = train_agent(episodes=750)

    # Test the trained agent
    # average_test_reward = test_agent(trained_agent)
    # print(f"\nAverage Test Reward: {average_test_reward:.2f}")