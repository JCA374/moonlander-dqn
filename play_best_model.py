"""
Script to visualize a trained model playing the LunarLander-v3 environment from OpenAI Gymnasium.
"""
import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# Optimize TensorFlow for maximum performance
try:
    # Set number of threads based on available CPU cores
    tf.config.threading.set_intra_op_parallelism_threads(4)  # Adjust based on your CPU
    tf.config.threading.set_inter_op_parallelism_threads(2)  # Adjust based on your CPU
    
    # Enable JIT compilation for faster execution
    tf.config.optimizer.set_jit(True)  # Enable XLA compilation
except:
    print("CPU optimization failed, using default settings")
import gymnasium as gym

class SimpleDQNAgent:
    """Simplified DQN Agent for loading and visualizing trained models."""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
    
    def _build_model(self):
        """Build a neural network for Q-function approximation matching the training model."""
        model = Sequential()
        
        # Match the exact architecture from train.py
        # First layer - use the same approach with build() instead of input_shape
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.build((None, self.state_size))
            
        # Single hidden layer - exact same as in train.py
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        
        # Output layer - linear activation for Q-values
        model.add(Dense(self.action_size, activation='linear'))
        
        # Use the same optimizer as in train.py
        optimizer = Adam(
            learning_rate=0.0005,  # Same as in train.py
            epsilon=1e-5
        )
        
        # Use Huber loss like in train.py
        model.compile(loss='huber', optimizer=optimizer)
        return model
    
    def act(self, state):
        """Choose the best action for the given state."""
        # Direct numpy prediction instead of TensorFlow for speed
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def load(self, filepath):
        """Load the model weights."""
        if not filepath.endswith('.weights.h5') and not os.path.exists(filepath):
            if os.path.exists(filepath + '.weights.h5'):
                filepath = filepath + '.weights.h5'
            else:
                filepath = filepath + '.h5'  # For backward compatibility
        self.model.load_weights(filepath)

def find_best_model(results_dir="results"):
    """Find the best model from previous runs."""
    print(f"Searching for models in {results_dir}")
    
    # First check the current_run directory to match train.py's updated structure
    current_run_dir = os.path.join(results_dir, 'current_run', 'models')
    if os.path.exists(current_run_dir):
        print(f"Checking current run directory: {current_run_dir}")
        
        # First, look specifically for files named 'best_model'
        best_model_candidates = []
        for ext in ['.weights.h5', '.h5']:
            best_model_path = os.path.join(current_run_dir, f'best_model{ext}')
            if os.path.exists(best_model_path):
                print(f"Found best model file: {best_model_path}")
                best_model_candidates.append(best_model_path)
        
        if best_model_candidates:
            # Sort by modification time (most recent first)
            best_model_candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            selected_model = best_model_candidates[0]
            print(f"Selected best model: {selected_model}")
            return selected_model
        
        # If no best_model files, check all model files
        model_files = []
        for file in os.listdir(current_run_dir):
            if file.endswith('.weights.h5') or file.endswith('.h5'):
                model_path = os.path.join(current_run_dir, file)
                print(f"Found model file: {model_path}")
                model_files.append(model_path)
        
        if model_files:
            # Sort by modification time (most recent first)
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            selected_model = model_files[0]
            print(f"Selected recent model: {selected_model}")
            return selected_model
    
    # Fall back to searching the entire results directory
    if not os.path.exists(results_dir):
        print(f"No results directory found at {results_dir}")
        return None
    
    # Look for best_model files first
    best_model_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if (file.startswith('best_model') and 
                (file.endswith('.weights.h5') or file.endswith('.h5'))):
                model_path = os.path.join(root, file)
                print(f"Found best model file: {model_path}")
                best_model_files.append(model_path)
    
    if best_model_files:
        # Sort by modification time (most recent first)
        best_model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        selected_model = best_model_files[0]
        print(f"Selected best model: {selected_model}")
        return selected_model
    
    # Fall back to any model file if no best_model files found
    model_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.weights.h5') or file.endswith('.h5'):
                model_path = os.path.join(root, file)
                print(f"Found model file: {model_path}")
                model_files.append(model_path)
    
    if model_files:
        # Sort by modification time (most recent first)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        selected_model = model_files[0]
        print(f"Selected recent model: {selected_model}")
        return selected_model
    
    print("No model files found.")
    return None

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Play LunarLander with trained model')
    
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to a specific model to load (if not provided, finds the most recent model)')
    parser.add_argument('--episodes', type=int, default=5,
                      help='Number of episodes to play')
    parser.add_argument('--max-steps', type=int, default=650,
                      help='Maximum steps per episode (default: 650)')
    parser.add_argument('--delay', type=float, default=0.0,
                      help='Delay between steps in seconds for visualization (0.0 for max speed)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    return parser.parse_args()

def play_model():
    """Visualize a trained model playing LunarLander-v3."""
    args = parse_args()
    
    # Create the OpenAI Gymnasium LunarLander-v3 environment with rendering
    env = gym.make('LunarLander-v3', render_mode='human')
    
    # Get state and action space dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print("\n=== MoonLander Visualization ===")
    print(f"State space: {state_size} dimensions")
    print(f"Action space: {action_size} actions (discrete)")
    
    # Create agent
    agent = SimpleDQNAgent(state_size, action_size)
    
    # Load model
    model_path = args.model_path
    if model_path is None:
        print("\nSearching for best trained model...")
        model_path = find_best_model()
    
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
    
    # Play episodes
    total_rewards = []
    success_count = 0
    
    for episode in range(1, args.episodes + 1):
        print(f"Episode {episode}/{args.episodes}")
        state, _ = env.reset(seed=args.seed + episode)
        total_reward = 0
        steps = 0
        done = False
        
        # Action mapping for display
        action_map = {
            0: "No engines",
            1: "Left engine (clockwise)",
            2: "Main engine",
            3: "Right engine (counter-clockwise)"
        }
        
        # Print initial status
        print(f"Episode {episode} started - watching agent play...")
        print(f"Current total score: 0.00")
        
        # Initialize step tracking variables
        steps = 0
        last_print_step = 0
        
        step_count = 0
        max_steps = args.max_steps
        
        while not done and step_count < max_steps:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step_count += 1
            
            # Enforce max steps limit
            if step_count >= max_steps:
                print(f"Reached maximum steps limit ({max_steps})")
                done = True
            
            # Optional delay for visualization - can be set to 0 for fastest playback
            if args.delay > 0:
                time.sleep(args.delay)
            
            # Give the CPU a small break even if delay is 0
            elif step_count % 10 == 0:
                time.sleep(0.001)
                
            steps = step_count  # Update steps to match step_count for display purposes
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            
            # Only print status every 100 steps or when something important happens
            if (steps % 100 == 0 and steps > last_print_step) or done:
                x, y, vx, vy, angle, _, _, _ = next_state
                print(f"Step {steps}: Score: {total_reward:.2f} | Position: ({x:.2f}, {y:.2f}), Speed: ({vx:.2f}, {vy:.2f}), Angle: {angle:.2f}")
                last_print_step = steps
            
            if done:
                if step_count >= max_steps:
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
    
    print("\n======== RESULTS SUMMARY ========")
    print(f"Average Score: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f} ({success_count}/{args.episodes})")
    print(f"Best Score: {max(total_rewards):.2f}")
    print(f"Worst Score: {min(total_rewards):.2f}")
    print("=================================")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    play_model()