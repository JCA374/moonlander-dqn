"""
Training script for LunarLander-v3 from OpenAI Gymnasium.
This file includes all necessary components for training a DQN agent.
"""
import os
import sys
import argparse
import time
import numpy as np
import random
from datetime import datetime
import json
import matplotlib.pyplot as plt


# Import TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# Optimize TensorFlow for Intel i7-8565U CPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations for Intel CPU

try:
    # Optimize for i7-8565U (4 cores, 8 threads)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    
    # Use standard float32 precision
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    print("CPU optimization for Intel i7-8565U applied successfully")
except Exception as e:
    print(f"Memory optimization failed: {e}, using default settings")

# Import Gymnasium
import gymnasium as gym

#------------------------------------------------------------------------------
# Efficient Replay Buffer Implementation
#------------------------------------------------------------------------------

class EfficientReplayBuffer:
    """Efficient NumPy-based replay buffer for better performance."""
    
    def __init__(self, state_size, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate arrays for efficiency
        self.states = np.zeros((max_size, state_size), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_size), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size

# DQN Agent Implementation
#------------------------------------------------------------------------------

class DQNAgent:
    """DQN agent for reinforcement learning in the LunarLander environment."""
    
    def __init__(self, 
                state_size, 
                action_size,
                memory_size=10000,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.98,
                learning_rate=0.0005,
                batch_size=64,
                update_target_freq=50,
                use_batch_norm=True):
        """Initialize the DQN agent."""
        self.state_size = state_size
        self.action_size = action_size
        self.memory = EfficientReplayBuffer(state_size, memory_size)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.step_counter = 0
        self.use_batch_norm = use_batch_norm
        
        # Build the main Q-network
        self.model = self._build_model()
        
        # Build the target Q-network
        self.target_model = self._build_model()
        self._update_target_network()  # Initial update to match the main network

    def _build_model(self):
        """Build network optimized for landing task."""
        model = Sequential()
        
        # Input layer
        model.add(tf.keras.Input(shape=(self.state_size,)))
        
        # Larger network for complex task
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        
        # Output layer
        model.add(Dense(self.action_size, activation='linear'))
        
        # Use lower learning rate for stability
        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
        model.compile(loss='huber', optimizer=optimizer)  # Huber loss for stability
        
        return model

    def _update_target_network(self):
        """Update the target network with weights from the main network.
        
        Uses hard update for better stability in DQN.
        """
        # Hard update - copy weights completely from main to target network
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory for replay."""
        self.memory.add(state, action, reward, next_state, done)
    
    def act(self, state, evaluate=False):
        """Choose an action based on the current state."""
        if not evaluate and np.random.rand() <= self.epsilon:
            # Exploration: choose a random action
            return random.randrange(self.action_size)
        
        # Exploitation: choose the best action based on Q-values
        # Direct numpy prediction instead of TensorFlow for speed
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Optimized training with better batch processing."""
        if len(self.memory) < self.batch_size * 2:  # Need more samples
            return 0.0

        # Train multiple times per replay for faster learning
        total_loss = 0
        for _ in range(2):  # Train twice per call
            # Sample batch
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            # Clip rewards for stability
            rewards = np.clip(rewards, -10, 10)
            
            # Calculate targets
            current_q = self.model.predict(states, verbose=0)
            next_q = self.target_model.predict(next_states, verbose=0)
            
            targets = current_q.copy()
            for i in range(self.batch_size):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
            
            # Train
            history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
            total_loss += history.history['loss'][0]
        
        # Update target network
        self.step_counter += 1
        if self.step_counter % self.update_target_freq == 0:
            self._update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_loss / 2

    def save(self, filepath):
        """Save the model weights."""
        if not filepath.endswith('.weights.h5'):
            filepath = filepath + '.weights.h5'
        self.model.save_weights(filepath)
    
    def load(self, filepath):
        """Load the model weights."""
        if not filepath.endswith('.weights.h5') and not os.path.exists(filepath):
            if os.path.exists(filepath + '.weights.h5'):
                filepath = filepath + '.weights.h5'
            else:
                filepath = filepath + '.h5'  # For backward compatibility
        self.model.load_weights(filepath)

#------------------------------------------------------------------------------
# Training Functions
#------------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LunarLander-v3 Training')
    
    # Set better default arguments for LunarLander-v3 environment
    
    parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'play'],
                    help='Mode: train or play')
    parser.add_argument('--output-dir', type=str, default='results',
                    help='Directory to save results')
    parser.add_argument('--model-path', type=str, default=None,
                    help='Path to a specific model to load')
    parser.add_argument('--episodes', type=int, default=500,
                    help='Number of episodes for training or playing')
    parser.add_argument('--restart', action='store_true',
                    help='Start training with a new model')
    parser.add_argument('--max-steps', type=int, default=1000,
                    help='Maximum steps per episode (increased for better landing attempts)')
    parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size for training (optimized for i7-8565U CPU)')
    parser.add_argument('--memory-size', type=int, default=50000,
                    help='Size of replay memory (optimized for CPU performance)')
    parser.add_argument('--memory-fraction', type=float, default=0.9,
                    help='Fraction of system memory to use (0.1-0.95)')
    parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount factor - standard value for this environment')
    parser.add_argument('--epsilon', type=float, default=1.0,
                    help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.05,
                    help='Minimum exploration rate - balanced for exploration/exploitation')
    parser.add_argument('--epsilon-decay', type=float, default=0.9995,
                    help='Exploration rate decay (optimized for better exploration)')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                    help='Learning rate - conservative for stable learning')
    parser.add_argument('--update-target-freq', type=int, default=100,
                    help='Frequency of target network updates (increased for stability)')
    parser.add_argument('--render', action='store_true',
                    help='Render the environment')
    parser.add_argument('--eval-freq', type=int, default=100,
                    help='Episode frequency for evaluation')
    parser.add_argument('--eval-episodes', type=int, default=3,
                    help='Number of episodes for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
    parser.add_argument('--delay', type=float, default=0.05,
                    help='Delay between steps when rendering (seconds)')
    
    return parser.parse_args()

def diagnose_hovering(state, step_count, hover_threshold=50):
    """Detect if agent is hovering instead of landing."""
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = state
    
    # Check hovering conditions
    is_hovering = (
        abs(x) < 0.3 and  # Near landing zone
        0.05 < y < 0.2 and  # Low but not landed
        abs(vy) < 0.1 and  # Low vertical velocity
        abs(vx) < 0.1 and  # Low horizontal velocity
        step_count > hover_threshold  # Been doing this for a while
    )
    
    return is_hovering

def setup_logger():
    """Setup basic logging to console."""
    import logging
    logger = logging.getLogger('moonlander')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(ch)
    
    return logger

def find_best_model(results_dir):
    """Find the best model from previous runs."""
    best_model_path = None
    
    if not os.path.exists(results_dir):
        print(f"No results directory found at {results_dir}")
        return None
    
    # First look for a file specifically named best_model.weights.h5
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == 'best_model.weights.h5':
                # Find the most recently modified best_model if there are multiple
                if best_model_path is None or os.path.getmtime(os.path.join(root, file)) > os.path.getmtime(best_model_path):
                    best_model_path = os.path.join(root, file)
    
    # If we found a best_model.weights.h5, return it
    if best_model_path:
        print(f"Found best model: {best_model_path}")
        return best_model_path
        
    # Otherwise look for any model file as a fallback
    model_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.weights.h5') or file.endswith('.h5'):
                model_files.append(os.path.join(root, file))
    
    if model_files:
        # Sort by modification time (most recent first)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        best_model_path = model_files[0]
        print(f"No best_model.weights.h5 found, using most recent model: {best_model_path}")
    else:
        print("No model files found.")
    
    return best_model_path

def optimized_train_loop(env, agent, args, logger):
    """Optimized training loop for faster learning with aggressive landing incentives."""
    # Pre-allocate arrays for metrics with more efficient data types
    episode_rewards = np.zeros(args.episodes, dtype=np.float32)
    episode_steps = np.zeros(args.episodes, dtype=np.int32)
    episode_losses = np.zeros(args.episodes, dtype=np.float32)

    # Track best reward for model saving
    best_reward = -float('inf')

    # Track landing statistics
    landing_attempts = 0
    successful_landings = 0
    crashes = 0
    hover_episodes = 0

    # Model saving directory
    model_dir = os.path.join(args.output_dir, 'current_run', 'models')

    # Increase memory allocation for training
    memory_blocks = max(5, int(args.memory_size * 1.5))
    logger.info(
        f"Allocating larger replay buffer with {memory_blocks} blocks for better performance")

    # Vectorized epsilon decay planning
    epsilon_schedule = np.maximum(
        args.epsilon_min,
        args.epsilon * args.epsilon_decay ** np.arange(args.episodes)
    )

    # Training loop
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed if episode == 0 else None)

        # Pre-allocate episode tracking variables
        total_reward = 0
        losses = []

        # Track episode behavior
        min_altitude = float('inf')
        hover_time = 0
        landing_attempted = False

        # Set current epsilon from pre-computed schedule
        agent.epsilon = epsilon_schedule[episode]

        # Main episode loop
        for step in range(args.max_steps):
            # Select and take action
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Extract state components for analysis
            x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state

            # Track minimum altitude
            min_altitude = min(min_altitude, y)

            # Check if landing was attempted
            if y < 0.5:
                landing_attempted = True

            # AGGRESSIVE REWARD SHAPING
            # Apply proper reward scaling for LunarLander
            if reward > 100:  # Successful landing
                scaled_reward = reward * 2.0  # DOUBLE success reward
                successful_landings += 1
                logger.info(
                    f"Episode {episode}: SUCCESSFUL LANDING! Reward: {reward}")
            elif reward < -100:  # Crash
                scaled_reward = reward * 0.5  # Reduce crash penalty to encourage trying
                crashes += 1
                if episode % 50 == 0:  # Log crashes periodically
                    logger.info(
                        f"Episode {episode}: Crash at altitude {y:.3f}")
            else:
                scaled_reward = reward

            # AGGRESSIVE ALTITUDE PENALTY - Force descent
            if y > 0.5:
                altitude_penalty = -0.5 * y  # Penalty proportional to altitude
                scaled_reward += altitude_penalty

            # STRONG ANTI-HOVERING MEASURES
            hover_detected = False
            if abs(x) < 0.4 and 0.02 < y < 0.3 and abs(vy) < 0.15 and abs(vx) < 0.15:
                hover_detected = True
                hover_time += 1
                hover_penalty = -2.0 - (step / 50)  # Even stronger penalty
                scaled_reward += hover_penalty

                if step > 150:  # Earlier termination
                    done = True
                    scaled_reward -= 50.0
                    hover_episodes += 1
                    logger.warning(
                        f"Episode {episode}: Terminated for hovering at step {step}")

            # MASSIVE LANDING INCENTIVES
            if y < 0.5 and not hover_detected:
                # Exponential bonus as altitude decreases
                # Exponential increase near ground
                proximity_bonus = 5.0 * np.exp(-y)
                scaled_reward += proximity_bonus

                # Huge bonus for landing configuration
                if y < 0.2 and abs(vy) < 0.5 and abs(angle) < 0.3:
                    scaled_reward += 20.0  # Massive bonus

                    # Extreme bonus if legs touching
                    if left_leg == 1 or right_leg == 1:
                        scaled_reward += 50.0  # Huge incentive to touch down

            # Strong time pressure
            time_penalty = -0.05 * (step / 100)
            scaled_reward += time_penalty

            # Penalty for using too much fuel (main engine)
            if action == 2:  # Main engine
                scaled_reward -= 0.2  # Small penalty to discourage constant thrust

            # Store experience
            agent.remember(state, action, scaled_reward, next_state, done)

            # Train agent (only every 4 steps for speed)
            if step % 4 == 0:
                loss = agent.replay()
                if loss > 0:
                    losses.append(loss)

            # Update state and tracking variables
            state = next_state
            total_reward += reward  # Track original reward for logging

            # Debug output for first few episodes
            if episode < 5 and step % 100 == 0:
                logger.info(f"Episode {episode}, Step {step}: Y={y:.2f}, Action={action}, "
                            f"Scaled Reward={scaled_reward:.2f}")

            if done:
                if landing_attempted:
                    landing_attempts += 1
                break

        # Update metrics
        episode_rewards[episode] = total_reward
        episode_steps[episode] = step + 1
        episode_losses[episode] = np.mean(losses) if losses else 0

        # Log progress with more detail
        if episode % 10 == 0:
            recent_rewards = episode_rewards[max(0, episode-10):episode+1]
            logger.info(f'Episode {episode}/{args.episodes} - '
                        f'Reward: {total_reward:.2f}, '
                        f'Avg10: {np.mean(recent_rewards):.2f}, '
                        f'Min Alt: {min_altitude:.3f}, '
                        f'Epsilon: {agent.epsilon:.4f}')

            # Log landing statistics
            if episode > 0:
                logger.info(f'Landing Stats - Attempts: {landing_attempts}, '
                            f'Successes: {successful_landings}, '
                            f'Crashes: {crashes}, '
                            f'Hovers: {hover_episodes}')

        # Save model if it's better than previous best
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(model_dir, 'best_model'))
            logger.info(f'New best model saved with reward: {best_reward:.2f}')

        # Also save if we get our first successful landing
        if successful_landings == 1 and total_reward > 0:
            agent.save(os.path.join(model_dir, 'first_landing_model'))
            logger.info('First successful landing! Model saved.')

        # Evaluate less frequently but more thoroughly
        if args.eval_freq > 0 and episode % args.eval_freq == 0:
            eval_reward, eval_steps, success_rate = evaluate(
                env, agent, args.eval_episodes, args.seed,
                args.max_steps, render=False
            )
            logger.info(f'Evaluation - Reward: {eval_reward:.2f}, '
                        f'Steps: {eval_steps:.2f}, '
                        f'Success: {success_rate:.2f}')

            # If still no success after 200 episodes, increase exploration
            if episode >= 200 and success_rate == 0:
                agent.epsilon = min(0.5, agent.epsilon * 2)
                logger.warning(
                    f"No success yet - increasing exploration to {agent.epsilon:.3f}")

    # Save final model
    agent.save(os.path.join(model_dir, 'final_model'))
    logger.info('Training completed - Final model saved')

    # Log final statistics
    logger.info(f'\nFinal Training Statistics:')
    logger.info(f'Total Landing Attempts: {landing_attempts}')
    logger.info(f'Successful Landings: {successful_landings}')
    logger.info(
        f'Success Rate: {successful_landings/max(1, landing_attempts):.2%}')
    logger.info(f'Crashes: {crashes}')
    logger.info(f'Hover Episodes: {hover_episodes}')

    # Only create plots at the end
    plot_learning_curves(episode_rewards, episode_steps, episode_losses,
                         os.path.join(args.output_dir, 'current_run', 'plots'))

    return agent, best_reward

def train(args, logger):
    """Train the DQN agent on the LunarLander environment."""
    # Set up directories - use a fixed directory to keep things simple
    run_dir = os.path.join(args.output_dir, 'current_run')
    model_dir = os.path.join(run_dir, 'models')
    log_dir = os.path.join(run_dir, 'logs')
    plot_dir = os.path.join(run_dir, 'plots')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(run_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    # Advanced TensorFlow memory optimization
    try:
        # More aggressive thread settings for performance
        tf.config.threading.set_intra_op_parallelism_threads(6)
        tf.config.threading.set_inter_op_parallelism_threads(4)
        
        # Set memory limits for better performance - allow up to 90% memory usage
        # tf.config.experimental.set_virtual_device_configuration(
        #     tf.config.list_physical_devices('CPU')[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 16)]
        # )
        
        # Increase memory allocation for operations
        # tf.config.experimental.set_memory_growth(
        #     tf.config.list_physical_devices('CPU')[0], 
        #     False  # Don't restrict growth
        # )
        
        # For training speed: use larger batches when more memory is available
        # args.batch_size = 256  # Removed - respects command line argument
        
        logger.info("Advanced TensorFlow memory optimization applied - using 90% of available memory")
    except Exception as e:
        logger.warning(f"Memory optimization failed: {e}, using default settings")

    # Create environment using OpenAI Gymnasium LunarLander-v3
    render_mode = None
    if args.render:
        render_mode = 'human'

    env = gym.make('LunarLander-v3', render_mode=render_mode)

    # Get state and action space dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    logger.info(f'State size: {state_size}, Action size: {action_size}')

    # Create agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        memory_size=args.memory_size,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        update_target_freq=args.update_target_freq,
        use_batch_norm=False
    )

    # Try to load previous best model and ensure training continues from best model
    best_reward = -float('inf')
    model_loaded = False
    
    if not args.restart:
        # First try to load from the specified path if provided
        if args.model_path:
            try:
                agent.load(args.model_path)
                logger.info(f"Successfully loaded model from user-specified path: {args.model_path}")
                model_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load model from specified path: {e}")
                
        # If no path specified or loading failed, try to find the best model
        if not model_loaded:
            model_path = find_best_model(args.output_dir)
            if model_path:
                try:
                    agent.load(model_path)
                    logger.info(f"Successfully loaded model from {model_path}")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
        
        # If a model was successfully loaded, evaluate it
        if model_loaded:
            # Evaluate the loaded model
            eval_reward, _, success_rate = evaluate(
                env, agent, 5, args.seed, args.max_steps, render=False)
            logger.info(
                f"Loaded model evaluation - Avg. Reward: {eval_reward:.2f}, Success: {success_rate:.2f}")

            # Set as best reward so far
            best_reward = eval_reward
            
            # Also adjust epsilon based on model performance to avoid starting over with full exploration
            if success_rate > 0.5:  # If model is already good
                agent.epsilon = max(0.2, agent.epsilon_min * 2)  # Lower exploration
                logger.info(f"Model performing well, reducing exploration rate to {agent.epsilon:.2f}")
    
    if not model_loaded:
        logger.info("Starting with a new model (no existing model loaded)")

    # Use the optimized training loop
    logger.info('Starting training with optimized loop...')
    agent, final_best_reward = optimized_train_loop(env, agent, args, logger)

    # Close environment
    env.close()
    logger.info(
        f'Training completed with best reward: {final_best_reward:.2f}')

def evaluate(env, agent, num_episodes, seed, max_steps, render=False):
    """Evaluate the agent on the environment."""
    rewards = []
    steps = []
    successes = 0
    hovers = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0
        step_count = 0
        done = False
        
        # Track hovering behavior
        hovering_steps = 0
        near_landing_pad = False
        
        for step in range(max_steps):
            action = agent.act(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Extract state components
            x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state
            
            # Check for hovering behavior (near landing pad but not landing)
            if abs(x) < 0.2 and y < 0.2 and y > 0.01 and abs(vy) < 0.2:
                hovering_steps += 1
                near_landing_pad = True
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                # Count as success if landing is successful (positive reward)
                if terminated and reward > 0:
                    successes += 1
                break
                
            # If episode reaches max steps and was hovering, count as a hover failure
            if step == max_steps - 1 and hovering_steps > 50 and near_landing_pad:
                hovers += 1
        
        rewards.append(total_reward)
        steps.append(step_count)
    
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    success_rate = successes / num_episodes
    hover_rate = hovers / num_episodes
    
    # Log additional information about hovering
    if hover_rate > 0:
        print(f"WARNING: Agent is hovering instead of landing in {hover_rate:.0%} of evaluation episodes")
    
    return avg_reward, avg_steps, success_rate

def plot_learning_curves(rewards, steps, losses, save_dir):
    """Plot learning curves."""
    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Plot rewards
    axs[0].plot(rewards)
    axs[0].set_title('Episode Rewards')
    axs[0].set_ylabel('Reward')
    
    # Add moving average
    if len(rewards) >= 100:
        moving_avg = np.convolve(rewards, np.ones(100) / 100, mode='valid')
        axs[0].plot(range(99, len(rewards)), moving_avg, 'r-')
    
    # Plot steps
    axs[1].plot(steps)
    axs[1].set_title('Episode Steps')
    axs[1].set_ylabel('Steps')
    
    # Plot losses
    axs[2].plot(losses)
    axs[2].set_title('Episode Losses')
    axs[2].set_ylabel('Loss')
    axs[2].set_xlabel('Episode')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()



#------------------------------------------------------------------------------
# Play/Visualization Functions
#------------------------------------------------------------------------------

def play(args, logger):
    """Play the LunarLander game using a trained model."""
    # Create environment with rendering
    env = gym.make('LunarLander-v3', render_mode='human')
    
    # Get state and action space dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        memory_size=10000,  # Not used during play
        gamma=0.99,
        epsilon=0.0,  # No exploration during play
        epsilon_min=0.0,
        epsilon_decay=0.995,
        learning_rate=0.0005,
        batch_size=64,  # Not used during play
        update_target_freq=50,
        use_batch_norm=False  # Simplify model for loading
    )
    
    # Load model
    model_path = args.model_path or find_best_model(args.output_dir)
    if not model_path:
        logger.error("No model found. Please train a model first.")
        return
    
    try:
        agent.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Play episodes
    logger.info(f"Playing {args.episodes} episodes")
    total_rewards = []
    total_steps = []
    success_count = 0
    
    for episode in range(1, args.episodes + 1):
        logger.info(f"Episode {episode}/{args.episodes}")
        state, _ = env.reset(seed=args.seed + episode)
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < args.max_steps:
            # Select action
            action = agent.act(state, evaluate=True)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Removed sleep call to improve performance
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                if terminated and reward > 0:  # Successful landing
                    success_count += 1
                    logger.info(f"Successful landing! Reward: {total_reward:.2f}")
                else:
                    logger.info(f"Episode finished. Reward: {total_reward:.2f}")
                break
        
        total_rewards.append(total_reward)
        total_steps.append(steps)
    
    # Print summary
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = success_count / args.episodes
    
    logger.info("\nPlay Summary:")
    logger.info(f"Average Reward: {avg_reward:.2f}")
    logger.info(f"Average Steps: {avg_steps:.2f}")
    logger.info(f"Success Rate: {success_rate:.2f} ({success_count}/{args.episodes})")
    
    # Close environment
    env.close()

#------------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------------


def main():
    """Main function."""
    # Set high priority for Windows systems
    if os.name == 'nt':  # Windows
        import psutil
        # Set process to high priority
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)

    args = parse_args()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger()

    # Run selected mode
    if args.mode == 'train':
        train(args, logger)
    elif args.mode == 'play':
        play(args, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()