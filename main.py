import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from plant import PlantEnv
from agent import Agent
import random

def train_agent(episodes=1000, max_steps=25, seed=42):
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
    plt.ion()  # Interactive mode for live updates

    # Initialize tracking metrics
    episode_rewards = []
    average_rewards = []
    epsilon_history = []
    loss_history = []
    water_usage_history = []  # New list to store water usage

    # Initialize environment and agent
    env = PlantEnv()
    state_size = len(env.get_state())
    action_size = 5
    agent = Agent(state_size=state_size, action_size=action_size)

    for episode in range(episodes):
        env.reset()
        episode_reward = 0
        episode_water_usage = 0  # Track water use per episode
        losses = []

        for step in range(max_steps):
            # Handle PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Get current state
            state = env.get_state()

            # Select action
            action = agent.act(state)

            # Take action in environment and track water use
            next_state, reward, water_used = env.step(action)  # Ensure env.step() returns water_used
            episode_water_usage += water_used  # Add water use for this step

            # Store experience
            agent.remember(state, action, reward, next_state)

            # Learn from experience
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    losses.append(loss)

            episode_reward += reward

            # Render environment
            screen.fill((255, 255, 255))
            env.render(screen, 400, 300)

            # Display training info on screen
            font = pygame.font.Font(None, 36)
            text = font.render(f'Episode: {episode + 1}', True, (0, 0, 0))
            screen.blit(text, (10, 10))
            text = font.render(f'Step: {step + 1}', True, (0, 0, 0))
            screen.blit(text, (10, 50))
            text = font.render(f'Epsilon: {agent.epsilon:.3f}', True, (0, 0, 0))
            screen.blit(text, (10, 90))

            pygame.display.flip()
            clock.tick(30)  # 30 FPS

        # Record metrics
        episode_rewards.append(episode_reward)
        average_rewards.append(np.mean(episode_rewards[-100:]))
        epsilon_history.append(agent.epsilon)
        water_usage_history.append(episode_water_usage)  # Record total water usage for the episode
        if losses:
            loss_history.append(np.mean(losses))
        else:
            loss_history.append(0)

        # Update plots every 10 episodes
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

    # Save training data
    training_data = pd.DataFrame({
        'Episode': range(episodes),
        'Reward': episode_rewards,
        'Average_Reward': average_rewards,
        'Epsilon': epsilon_history,
        'Loss': loss_history,
        'Water_Usage': water_usage_history  # Include water usage in saved data
    })
    training_data.to_csv('training_results.csv', index=False)

    pygame.quit()
    return agent, training_data


def test_agent(agent, episodes=10):
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

            next_state, reward = env.step(action)
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
    trained_agent, training_data = train_agent(episodes=1000)

    # Test the trained agent
    average_test_reward = test_agent(trained_agent)
    # print(f"\nAverage Test Reward: {average_test_reward:.2f}")