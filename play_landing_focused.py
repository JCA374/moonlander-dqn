"""
Script to visualize the landing-focused agent.
"""
import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
import gymnasium as gym

# Import the landing-focused agent class
from landing_fix import LandingFocusedAgent

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Play LunarLander with landing-focused agent')
    
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to a specific model to load')
    parser.add_argument('--episodes', type=int, default=5,
                      help='Number of episodes to play')
    parser.add_argument('--max-steps', type=int, default=650,
                      help='Maximum steps per episode')
    parser.add_argument('--delay', type=float, default=0.0,
                      help='Delay between steps in seconds for visualization')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    return parser.parse_args()

def find_best_model(model_dir="results/landing_focused/models"):
    """Find the best landing-focused model."""
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found")
        return None
    
    best_model_path = os.path.join(model_dir, "best_model.weights.h5")
    if os.path.exists(best_model_path):
        print(f"Found landing-focused model: {best_model_path}")
        return best_model_path
    
    # Fall back to find any model
    model_files = []
    for file in os.listdir(model_dir):
        if file.endswith('.weights.h5') or file.endswith('.h5'):
            model_path = os.path.join(model_dir, file)
            model_files.append(model_path)
    
    if model_files:
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        print(f"Using most recent model: {model_files[0]}")
        return model_files[0]
    
    print(f"No models found in {model_dir}")
    return None

def play():
    """Play LunarLander with the landing-focused agent."""
    args = parse_args()
    
    # Create environment with rendering
    env = gym.make('LunarLander-v3', render_mode='human')
    
    # Get state and action space dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print("\n=== MoonLander Visualization (Landing-Focused) ===")
    print(f"State space: {state_size} dimensions")
    print(f"Action space: {action_size} actions (discrete)")
    
    # Create agent
    agent = LandingFocusedAgent(
        state_size=state_size,
        action_size=action_size,
        memory_size=10000,
        gamma=0.99,
        epsilon=0.0,  # No exploration for playing
        epsilon_min=0.0,
        epsilon_decay=0.995,
        learning_rate=0.0005,
        batch_size=64,
        update_target_freq=20
    )
    
    # Load model
    model_path = args.model_path
    if model_path is None:
        print("\nSearching for landing-focused model...")
        model_path = find_best_model()
    
    if model_path is None:
        model_path = find_best_model("results/current_run/models")
        print("Using standard model instead...")
    
    if model_path is None:
        print("\nNo model found. Please train a model first.")
        return
    
    try:
        agent.load(model_path)
        print(f"\nSuccessfully loaded model from {model_path}")
    except Exception as e:
        print(f"\nFailed to load model: {e}")
        return
    
    print(f"\nPlaying {args.episodes} episodes with delay={args.delay}s")
    print("-----------------------------------")
    
    # Statistics
    total_rewards = []
    success_count = 0
    hover_count = 0
    
    for episode in range(1, args.episodes + 1):
        print(f"Episode {episode}/{args.episodes}")
        state, _ = env.reset(seed=args.seed + episode)
        total_reward = 0
        steps = 0
        done = False
        
        # Track status for hovering detection
        hovering_steps = 0
        near_ground = False
        
        while not done and steps < args.max_steps:
            # Select action
            action = agent.act(state, evaluate=True)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Extract state components for analysis
            x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state
            
            # Check for hovering behavior
            if abs(x) < 0.2 and y < 0.5 and y > 0.05 and abs(vy) < 0.2:
                hovering_steps += 1
                near_ground = True
            else:
                hovering_steps = 0
            
            # Add visualization delay
            if args.delay > 0:
                time.sleep(args.delay)
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            steps += 1
            
            # Print status periodically
            if steps % 100 == 0 or done:
                print(f"Step {steps}: Score: {total_reward:.2f} | Position: ({x:.2f}, {y:.2f}), Speed: ({vx:.2f}, {vy:.2f}), Angle: {angle:.2f}")
        
        # Determine outcome
        if hovering_steps > 100 and near_ground:
            outcome = "HOVERING - Agent hovering above landing pad"
            hover_count += 1
        elif steps >= args.max_steps:
            outcome = "TIMEOUT - Reached maximum steps"
        elif terminated and reward > 0:
            outcome = "SUCCESS - Good landing!"
            success_count += 1
        elif reward <= -100:
            outcome = "CRASH"
        else:
            outcome = "Failed to land properly"
        
        print(f"\nEpisode {episode} finished after {steps} steps")
        print(f"Final Score: {total_reward:.2f}")
        print(f"Outcome: {outcome}")
        print("-" * 40)
        
        total_rewards.append(total_reward)
    
    # Print summary
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / args.episodes
    hover_rate = hover_count / args.episodes
    
    print("\n======== RESULTS SUMMARY ========")
    print(f"Average Score: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f} ({success_count}/{args.episodes})")
    print(f"Hover Rate: {hover_rate:.2f} ({hover_count}/{args.episodes})")
    print(f"Best Score: {max(total_rewards):.2f}")
    print(f"Worst Score: {min(total_rewards):.2f}")
    print("=================================")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    play()