"""
Script to modify the DQN agent to focus more on landing rather than hovering
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
import time
import random

# Import our existing agent class
from train import DQNAgent

class LandingFocusedAgent(DQNAgent):
    """Modified DQN agent with enhanced reward shaping to focus on landing."""
    
    def __init__(self, state_size, action_size, **kwargs):
        # Initialize the parent DQN agent
        super().__init__(state_size, action_size, **kwargs)
        
        # Additional parameters for landing focus
        self.landing_focus = True
        self.step_penalty = -0.05  # Small penalty for each step to encourage faster landing
        self.altitude_penalty = -0.1  # Penalty for hovering
        self.landing_zone_radius = 0.2  # Consider this radius around center as landing zone
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience with modified rewards to encourage landing."""
        # Extract state components
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = state
        
        # Original reward
        original_reward = reward
        modified_reward = reward
        
        # Apply reward shaping to encourage landing
        if self.landing_focus:
            # Add step penalty to discourage hovering
            modified_reward += self.step_penalty
            
            # If in landing zone but not landing, apply altitude penalty
            if abs(x) < self.landing_zone_radius and y < 1.0 and y > 0.1:
                modified_reward += self.altitude_penalty * y  # More penalty the higher it hovers
            
            # If close to ground with low velocity and good angle, give bonus
            if y < 0.2 and abs(vy) < 0.3 and abs(angle) < 0.2:
                modified_reward += 0.5  # Landing guidance bonus
                
            # If legs are touching but not landing completely, give strong incentive
            if (left_leg == 1 or right_leg == 1) and not done:
                modified_reward += 1.0
            
            # Give larger rewards for successful landings
            if done and original_reward > 0:
                modified_reward = original_reward * 1.5
        
        # Store the modified experience
        super().remember(state, action, modified_reward, next_state, done)

def train_landing_focused_agent(num_episodes=1000, render=False):
    """Train an agent with enhanced landing focus."""
    print("Training Landing-Focused Agent...")
    
    # Create the environment
    render_mode = "human" if render else None
    env = gym.make('LunarLander-v3', render_mode=render_mode)
    
    # Get state and action space dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create the landing focused agent
    agent = LandingFocusedAgent(
        state_size=state_size,
        action_size=action_size,
        memory_size=100000,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        learning_rate=0.0005,
        batch_size=64,
        update_target_freq=20
    )
    
    # Try to load the best model to continue training
    try:
        model_path = "results/current_run/models/best_model.weights.h5"
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"Loaded existing model from {model_path}")
            # Increase exploration slightly for retraining
            agent.epsilon = max(0.3, agent.epsilon)
    except Exception as e:
        print(f"Could not load model: {e}")
    
    # Training loop
    best_reward = -float('inf')
    total_rewards = []
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=42 + episode)
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 650:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience with modified reward
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            # Update state and tracking variables
            state = next_state
            total_reward += reward
            steps += 1
            
            if render and episode % 20 == 0:
                time.sleep(0.01)
        
        # Track rewards
        total_rewards.append(total_reward)
        
        # Print progress
        avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
        print(f"Episode {episode}/{num_episodes} - Reward: {total_reward:.2f}, Avg: {avg_reward:.2f}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}")
        
        # Save if better than previous best
        if total_reward > best_reward:
            best_reward = total_reward
            model_dir = "results/landing_focused/models"
            os.makedirs(model_dir, exist_ok=True)
            agent.save(os.path.join(model_dir, "best_model"))
            print(f"New best model saved with reward: {best_reward:.2f}")
        
        # Evaluate every 100 episodes
        if episode % 100 == 0:
            evaluate_agent(agent, env, 3)
    
    # Save final model
    model_dir = "results/landing_focused/models"
    os.makedirs(model_dir, exist_ok=True)
    agent.save(os.path.join(model_dir, "final_model"))
    print("Training completed")
    
    return agent

def evaluate_agent(agent, env, num_episodes=5):
    """Evaluate the agent on the environment."""
    rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            
            if terminated and reward > 0:
                success_count += 1
        
        rewards.append(total_reward)
    
    avg_reward = np.mean(rewards)
    success_rate = success_count / num_episodes
    
    print(f"Evaluation - Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")
    
    return avg_reward, success_rate

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a landing-focused DQN agent')
    
    parser.add_argument('--episodes', type=int, default=500,
                      help='Number of episodes for training')
    parser.add_argument('--render', action='store_true',
                      help='Render the environment during training')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_landing_focused_agent(num_episodes=args.episodes, render=args.render)