"""
Script to test and analyze rewards in the LunarLander-v3 environment from OpenAI Gymnasium.
This script helps determine if the reward structure is sufficiently encouraging desired behavior.
"""
import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Test LunarLander-v3 rewards')
    
    parser.add_argument('--episodes', type=int, default=100,
                      help='Number of episodes to analyze (default: 100)')
    parser.add_argument('--render', action='store_true',
                      help='Render the environment during testing')
    parser.add_argument('--max-steps', type=int, default=650,
                      help='Maximum steps per episode (default: 650)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    return parser.parse_args()

def analyze_rewards():
    """Run tests to analyze the reward structure in LunarLander-v3."""
    args = parse_args()
    
    # Create the OpenAI Gymnasium LunarLander-v3 environment
    render_mode = "human" if args.render else None
    env = gym.make('LunarLander-v3', render_mode=render_mode)
    
    # Get state space dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print("\n=== LunarLander-v3 Reward Analysis ===")
    print(f"State space: {state_size} dimensions")
    print(f"Action space: {action_size} actions (discrete)")
    print(f"Testing over {args.episodes} episodes with max {args.max_steps} steps each")
    print("-----------------------------------")
    
    # Statistics tracking
    total_rewards = []
    step_counts = []
    success_count = 0
    crash_count = 0
    timeout_count = 0
    
    # Track reward breakdown by event
    reward_sources = defaultdict(list)
    reward_deltas = []
    
    # Event tracking
    landing_pad_rewards = []
    successful_landing_rewards = []
    crash_landing_rewards = []
    leg_contact_rewards = []
    movement_rewards = []
    
    # Track states and rewards for analysis
    altitude_vs_reward = []
    velocity_vs_reward = []
    angle_vs_reward = []
    
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        total_reward = 0
        step_count = 0
        done = False
        
        prev_reward = 0
        prev_state = state
        
        episode_altitude_rewards = []
        episode_velocity_rewards = []
        episode_angle_rewards = []
        
        leg_contact_left = False
        leg_contact_right = False
        
        print(f"Episode {episode}/{args.episodes}")
        
        while not done and step_count < args.max_steps:
            # Take random action
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Calculate reward delta
            reward_delta = reward - prev_reward
            reward_deltas.append(reward_delta)
            
            # Extract state components
            x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state
            
            # Analyze reward sources
            if reward > 0:
                # Analyze what might have triggered positive reward
                if y < prev_state[1]:  # Moving downward
                    movement_rewards.append(reward)
                
                # Check for leg contacts
                if left_leg == 1 and not leg_contact_left:
                    leg_contact_left = True
                    leg_contact_rewards.append(reward)
                
                if right_leg == 1 and not leg_contact_right:
                    leg_contact_right = True
                    leg_contact_rewards.append(reward)
                
                # Landing pad detection (approximation)
                if abs(x) < 0.2 and y < 0.1:
                    landing_pad_rewards.append(reward)
            
            # Track correlations
            episode_altitude_rewards.append((y, reward))
            episode_velocity_rewards.append((np.sqrt(vx**2 + vy**2), reward))
            episode_angle_rewards.append((abs(angle), reward))
            
            # Update state and tracking variables
            prev_state = next_state
            prev_reward = reward
            state = next_state
            total_reward += reward
            step_count += 1
            
            if args.render and step_count % 10 == 0:
                time.sleep(0.01)  # Small delay for rendering
        
        # Record outcome
        if terminated and total_reward > 0:
            outcome = "SUCCESS"
            success_count += 1
            successful_landing_rewards.append(total_reward)
        elif step_count >= args.max_steps:
            outcome = "TIMEOUT"
            timeout_count += 1
        else:
            outcome = "CRASH"
            crash_count += 1
            crash_landing_rewards.append(total_reward)
        
        # Track all episode stats
        total_rewards.append(total_reward)
        step_counts.append(step_count)
        
        # Accumulate state-reward correlations
        altitude_vs_reward.extend(episode_altitude_rewards)
        velocity_vs_reward.extend(episode_velocity_rewards)
        angle_vs_reward.extend(episode_angle_rewards)
        
        print(f"  Reward: {total_reward:.2f}, Steps: {step_count}, Outcome: {outcome}")
    
    # Close environment
    env.close()
    
    # Analyze and print statistics
    print("\n=== Reward Analysis Results ===")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Median reward: {np.median(total_rewards):.2f}")
    print(f"Min/Max rewards: {min(total_rewards):.2f} / {max(total_rewards):.2f}")
    print(f"Average steps: {np.mean(step_counts):.2f}")
    
    # Outcome statistics
    print("\n=== Outcomes ===")
    print(f"Success rate: {success_count/args.episodes:.2%} ({success_count}/{args.episodes})")
    print(f"Crash rate: {crash_count/args.episodes:.2%} ({crash_count}/{args.episodes})")
    print(f"Timeout rate: {timeout_count/args.episodes:.2%} ({timeout_count}/{args.episodes})")
    
    # Reward component analysis
    print("\n=== Reward Component Analysis ===")
    if successful_landing_rewards:
        print(f"Successful landing rewards: {np.mean(successful_landing_rewards):.2f} ± {np.std(successful_landing_rewards):.2f}")
    if crash_landing_rewards:
        print(f"Crash landing rewards: {np.mean(crash_landing_rewards):.2f} ± {np.std(crash_landing_rewards):.2f}")
    if leg_contact_rewards:
        print(f"Leg contact rewards: {np.mean(leg_contact_rewards):.2f} ± {np.std(leg_contact_rewards):.2f}")
    if landing_pad_rewards:
        print(f"Landing pad rewards: {np.mean(landing_pad_rewards):.2f} ± {np.std(landing_pad_rewards):.2f}")
    if movement_rewards:
        print(f"Movement rewards: {np.mean(movement_rewards):.2f} ± {np.std(movement_rewards):.2f}")
    
    # Calculate reward sufficiency metrics
    print("\n=== Reward Sufficiency Analysis ===")
    
    # Calculate if rewards for successful landing are significantly higher
    if successful_landing_rewards and crash_landing_rewards:
        success_mean = np.mean(successful_landing_rewards)
        crash_mean = np.mean(crash_landing_rewards)
        reward_ratio = success_mean / abs(crash_mean) if crash_mean < 0 else 0
        print(f"Success vs Crash reward ratio: {reward_ratio:.2f}")
        
        if reward_ratio > 1.5:
            print("✓ Success rewards are SUFFICIENTLY higher than crash penalties")
        else:
            print("✗ Success rewards may NOT be sufficiently higher than crash penalties")
    
    # Analyze reward volatility
    reward_volatility = np.std(reward_deltas)
    print(f"Reward volatility: {reward_volatility:.2f}")
    if reward_volatility < 10:
        print("✓ Reward volatility is LOW (stable learning environment)")
    else:
        print("✗ Reward volatility is HIGH (may lead to unstable learning)")
    
    # Evaluate learning signal clarity based on correlations
    altitude_corr = np.corrcoef([a[0] for a in altitude_vs_reward], [a[1] for a in altitude_vs_reward])[0, 1]
    velocity_corr = np.corrcoef([v[0] for v in velocity_vs_reward], [v[1] for v in velocity_vs_reward])[0, 1]
    angle_corr = np.corrcoef([a[0] for a in angle_vs_reward], [a[1] for a in angle_vs_reward])[0, 1]
    
    print(f"Altitude-reward correlation: {altitude_corr:.2f}")
    print(f"Velocity-reward correlation: {velocity_corr:.2f}")
    print(f"Angle-reward correlation: {angle_corr:.2f}")
    
    # Evaluate if success rewards are distinct and learnable
    success_reward_separation = 0
    if successful_landing_rewards and total_rewards:
        success_mean = np.mean(successful_landing_rewards)
        all_mean = np.mean(total_rewards)
        success_reward_separation = (success_mean - all_mean) / np.std(total_rewards) if np.std(total_rewards) > 0 else 0
    
    print(f"Success reward separation: {success_reward_separation:.2f}")
    if success_reward_separation > 1.0:
        print("✓ Success rewards are DISTINCT (good learning signal)")
    else:
        print("✗ Success rewards may NOT be distinct enough")
    
    # Final assessment
    print("\n=== Final Assessment ===")
    if success_count > 0 and (success_reward_separation > 1.0 or reward_ratio > 1.5):
        print("✅ Rewards appear SUFFICIENT for effective learning")
        print("   The reward structure clearly distinguishes success from failure")
        print("   and provides adequate signals for reinforcement learning.")
    else:
        print("❌ Rewards may be INSUFFICIENT for effective learning")
        print("   Consider modifying the reward structure to create clearer signals.")
    
    # Generate plots
    plot_reward_analysis(total_rewards, altitude_vs_reward, velocity_vs_reward, angle_vs_reward)

def plot_reward_analysis(total_rewards, altitude_vs_reward, velocity_vs_reward, angle_vs_reward):
    """Generate plots for reward analysis."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot reward distribution
    axs[0, 0].hist(total_rewards, bins=20)
    axs[0, 0].set_title('Reward Distribution')
    axs[0, 0].set_xlabel('Total Reward')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].axvline(np.mean(total_rewards), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(total_rewards):.2f}')
    axs[0, 0].legend()
    
    # Plot altitude vs reward
    altitude = [a[0] for a in altitude_vs_reward]
    alt_rewards = [a[1] for a in altitude_vs_reward]
    axs[0, 1].scatter(altitude, alt_rewards, alpha=0.5)
    axs[0, 1].set_title('Altitude vs Reward')
    axs[0, 1].set_xlabel('Altitude')
    axs[0, 1].set_ylabel('Reward')
    
    # Plot velocity vs reward
    velocity = [v[0] for v in velocity_vs_reward]
    vel_rewards = [v[1] for v in velocity_vs_reward]
    axs[1, 0].scatter(velocity, vel_rewards, alpha=0.5)
    axs[1, 0].set_title('Velocity vs Reward')
    axs[1, 0].set_xlabel('Velocity')
    axs[1, 0].set_ylabel('Reward')
    
    # Plot angle vs reward
    angle = [a[0] for a in angle_vs_reward]
    angle_rewards = [a[1] for a in angle_vs_reward]
    axs[1, 1].scatter(angle, angle_rewards, alpha=0.5)
    axs[1, 1].set_title('Angle vs Reward')
    axs[1, 1].set_xlabel('Absolute Angle')
    axs[1, 1].set_ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('reward_analysis.png')
    plt.close()
    print("Reward analysis plots saved to 'reward_analysis.png'")

if __name__ == "__main__":
    analyze_rewards()