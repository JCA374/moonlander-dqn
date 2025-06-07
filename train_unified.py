"""
UNIFIED Training script for LunarLander-v3 - Complete Optimization Package
Combines:
- Fixed DQN architecture and success detection from train_fixed.py
- Advanced TensorFlow optimizations for CPU performance
- Enhanced training loop with performance monitoring
- Configurable optimization features via command line arguments

Key Features:
1. Fixed success detection (proper landing detection)
2. Optimized reward shaping for faster learning
3. Advanced CPU optimizations for Intel i7-8565U
4. Mixed precision and XLA compilation support
5. Performance monitoring and detailed logging
6. Configurable network enhancements (batch norm, dropout)
7. Auto-saving at regular intervals
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

# Import TensorFlow with optimization settings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# Environment
import gymnasium as gym

# Optional performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available - performance monitoring disabled")


def apply_cpu_optimizations(args):
    """Apply CPU optimizations based on command line arguments."""
    print("🚀 Applying Intel i7-8565U optimizations...")

    # Set environment variables for optimal performance
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

    try:
        # CPU threading optimization for your i7-8565U (4 cores, 8 threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(
            max(1, args.cpu_threads // 3))

        # XLA compilation if requested
        if args.enable_xla:
            tf.config.optimizer.set_jit(True)
            print("✓ XLA compilation enabled")

        # Mixed precision if requested
        if args.mixed_precision:
            # Use mixed_float16 for better performance on CPU
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"✓ Mixed precision enabled: {policy.name}")
        else:
            # Ensure float32 for stability
            policy = tf.keras.mixed_precision.Policy('float32')
            tf.keras.mixed_precision.set_global_policy(policy)

        print("✓ CPU optimizations applied successfully")
        print(
            f"✓ Using {tf.config.threading.get_intra_op_parallelism_threads()} threads for operations")
        print(
            f"✓ Using {tf.config.threading.get_inter_op_parallelism_threads()} threads for coordination")

        return True

    except Exception as e:
        print(f"⚠ Advanced optimization failed: {e}, using defaults")
        # Fallback to basic optimization
        try:
            tf.config.threading.set_intra_op_parallelism_threads(4)
            tf.config.threading.set_inter_op_parallelism_threads(2)
            print("✓ Fallback CPU optimization applied")
            return False
        except Exception as e2:
            print(f"✗ Even fallback optimization failed: {e2}")
            return False


class EfficientReplayBuffer:
    """Efficient NumPy-based replay buffer with optional enhancements."""

    def __init__(self, state_size, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Use efficient data types
        self.states = np.zeros((max_size, state_size), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_size), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.size < batch_size:
            return None

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


class DQNAgent:
    """Enhanced DQN agent with optimization options."""

    def __init__(self,
                 state_size,
                 action_size,
                 memory_size=50000,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 learning_rate=0.0005,
                 batch_size=64,
                 update_target_freq=20,
                 # Enhancement parameters
                 use_batch_norm=False,
                 dropout_rate=0.0,
                 gradient_clip=1.0):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.step_counter = 0

        # Enhancement parameters
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.gradient_clip = gradient_clip

        # Create memory buffer
        self.memory = EfficientReplayBuffer(state_size, memory_size)

        # Build networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_network()

        print(
            f"✓ DQN Agent created with {self.model.count_params():,} parameters")
        if use_batch_norm:
            print("✓ Batch normalization enabled")
        if dropout_rate > 0:
            print(f"✓ Dropout enabled: {dropout_rate}")
        if gradient_clip > 0:
            print(f"✓ Gradient clipping enabled: {gradient_clip}")

    def _build_model(self):
        """Enhanced model architecture with optional optimizations."""
        model = Sequential()

        # Input layer
        model.add(tf.keras.Input(shape=(self.state_size,)))

        # First hidden layer (64 neurons - proven architecture)
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        if self.use_batch_norm:
            model.add(BatchNormalization())
        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate))

        # Second hidden layer (64 neurons - proven architecture)
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        if self.use_batch_norm:
            model.add(BatchNormalization())
        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate))

        # Output layer
        model.add(Dense(self.action_size, activation='linear'))

        # Enhanced optimizer with optional gradient clipping
        optimizer_kwargs = {
            'learning_rate': self.learning_rate,
            'epsilon': 1e-5
        }
        if self.gradient_clip > 0:
            optimizer_kwargs['clipnorm'] = self.gradient_clip

        optimizer = Adam(**optimizer_kwargs)

        # Compile with optional JIT compilation
        try:
            model.compile(loss='huber', optimizer=optimizer, jit_compile=True)
        except:
            model.compile(loss='huber', optimizer=optimizer)

        return model

    def _update_target_network(self):
        """Hard update target network."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, evaluate=False):
        """Choose action with epsilon-greedy policy."""
        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size * 2:
            return 0.0

        # Sample batch
        batch_data = self.memory.sample(self.batch_size)
        if batch_data is None:
            return 0.0

        states, actions, rewards, next_states, dones = batch_data

        # Compute Q-values
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)

        # Bellman equation
        targets = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + \
                    self.gamma * np.max(next_q[i])

        # Train
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        # Update target network
        self.step_counter += 1
        if self.step_counter % self.update_target_freq == 0:
            self._update_target_network()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss

    def save(self, filepath):
        if not filepath.endswith('.weights.h5'):
            filepath = filepath + '.weights.h5'
        self.model.save_weights(filepath)

    def load(self, filepath):
        if not filepath.endswith('.weights.h5') and not os.path.exists(filepath):
            if os.path.exists(filepath + '.weights.h5'):
                filepath = filepath + '.weights.h5'
            else:
                filepath = filepath + '.h5'
        self.model.load_weights(filepath)


def parse_args():
    """Enhanced argument parser with optimization options."""
    parser = argparse.ArgumentParser(
        description='Unified Optimized LunarLander-v3 Training')

    # Core training arguments
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'play'])
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--memory-size', type=int, default=50000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon-min', type=float, default=0.01)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--update-target-freq', type=int, default=20)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--eval-freq', type=int, default=50)
    parser.add_argument('--eval-episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)

    # Performance optimization arguments
    parser.add_argument('--cpu-threads', type=int, default=6,
                        help='Number of CPU threads to use (default: 6 for i7-8565U)')
    parser.add_argument('--enable-xla', action='store_true',
                        help='Enable XLA compilation for faster execution')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training for better performance')
    parser.add_argument('--batch-norm', action='store_true',
                        help='Add batch normalization to network layers')

    # Model saving and monitoring options
    parser.add_argument('--save-interval', type=int, default=25,
                        help='Save model every N episodes (default: 25)')
    parser.add_argument('--performance-monitor', action='store_true',
                        help='Enable detailed performance monitoring and logging')

    # Advanced training options
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping threshold (default: 1.0, 0 to disable)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate for regularization (default: 0.0)')

    return parser.parse_args()


def setup_logger():
    """Setup logging."""
    import logging
    logger = logging.getLogger('moonlander_unified')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def find_best_model(results_dir):
    """Find best model from previous runs."""
    if not os.path.exists(results_dir):
        return None

    best_model_path = None
    best_time = 0

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == 'best_model.weights.h5':
                file_path = os.path.join(root, file)
                file_time = os.path.getmtime(file_path)
                if file_time > best_time:
                    best_time = file_time
                    best_model_path = file_path

    if best_model_path:
        print(f"Found best model: {best_model_path}")
        return best_model_path

    # Fallback to any model
    model_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.weights.h5') or file.endswith('.h5'):
                model_files.append(os.path.join(root, file))

    if model_files:
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return model_files[0]

    return None


def create_enhanced_agent(state_size, action_size, args):
    """Create DQN agent with enhancement options."""
    return DQNAgent(
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
        # Enhancement parameters
        use_batch_norm=args.batch_norm,
        dropout_rate=args.dropout,
        gradient_clip=args.gradient_clip
    )


def apply_advanced_reward_shaping(next_state, reward, step, args):
    """Apply intelligent reward shaping to encourage successful landings."""
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state

    original_reward = reward
    scaled_reward = reward

    # 1. Amplify significant rewards
    if reward > 50:  # Good progress or landing
        scaled_reward = reward * 1.5
    elif reward < -50:  # Significant crash
        scaled_reward = reward * 0.8  # Reduce to encourage exploration

    # 2. Landing encouragement - progressive rewards
    if y < 0.5:  # Close to ground
        # Encourage being centered over landing pad
        if abs(x) < 0.5:
            center_bonus = 0.5 * (0.5 - abs(x))  # Up to +0.25
            scaled_reward += center_bonus

            # Encourage controlled descent
            if y < 0.2 and abs(vy) < 1.0:
                scaled_reward += 1.0  # Controlled approach bonus

                # Leg contact rewards
                if left_leg or right_leg:
                    scaled_reward += 2.0  # Single leg contact

                    # Both legs touching - pre-landing state
                    if left_leg and right_leg and abs(vx) < 0.5:
                        scaled_reward += 5.0  # Strong landing encouragement

    # 3. Anti-hovering mechanism (gentle)
    if 0.05 < y < 0.3 and abs(vy) < 0.1 and step > 300:
        scaled_reward -= 0.2  # Small penalty for hovering

    # 4. Time pressure in final phase
    if step > 500:
        scaled_reward -= 0.05  # Gentle time pressure

    # 5. Timeout prevention
    if step >= args.max_steps - 1:
        scaled_reward -= 50.0  # Strong timeout penalty

    return scaled_reward, original_reward


def optimized_train_loop(env, agent, args, logger, episode_saves_dir):
    """Enhanced training loop with performance monitoring and optimizations."""

    # Performance tracking
    start_time = time.time()

    # Training metrics
    episode_rewards = np.zeros(args.episodes, dtype=np.float32)
    episode_steps = np.zeros(args.episodes, dtype=np.int32)
    episode_losses = np.zeros(args.episodes, dtype=np.float32)
    training_times = np.zeros(args.episodes, dtype=np.float32)

    # Performance monitoring
    cpu_usage = []
    memory_usage = []

    best_reward = -float('inf')

    # Training statistics
    successful_landings = 0
    crashes = 0
    timeouts = 0

    model_dir = os.path.join(args.output_dir, 'current_run', 'models')

    for episode in range(args.episodes):
        episode_start = time.time()

        # Monitor system resources every 10 episodes
        if args.performance_monitor and PSUTIL_AVAILABLE and episode % 10 == 0:
            cpu_usage.append(psutil.cpu_percent())
            memory_usage.append(psutil.virtual_memory().percent)

        state, _ = env.reset(seed=args.seed if episode == 0 else None)

        total_reward = 0
        losses = []

        for step in range(args.max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Apply advanced reward shaping
            scaled_reward, original_reward = apply_advanced_reward_shaping(
                next_state, reward, step, args)

            # Store experience with scaled reward
            agent.remember(state, action, scaled_reward, next_state, done)

            # Train agent every 4 steps
            if step % 4 == 0:
                loss = agent.replay()
                if loss > 0:
                    losses.append(loss)

            state = next_state
            total_reward += scaled_reward

            if done:
                # Handle timeout override
                if step >= args.max_steps - 1:
                    if scaled_reward > 0:
                        scaled_reward = -20.0
                        total_reward = total_reward - original_reward + scaled_reward

                # Enhanced success detection
                x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state

                landed_successfully = False
                if terminated:  # Natural episode end
                    # Multiple success criteria
                    if (left_leg and right_leg and
                        abs(vx) < 0.5 and abs(vy) < 0.5 and
                            abs(x) < 0.5):
                        landed_successfully = True
                    elif total_reward > 200:  # High reward threshold
                        landed_successfully = True

                if landed_successfully:
                    successful_landings += 1
                    logger.info(f"Episode {episode}: ✅ SUCCESSFUL LANDING! "
                                f"Total reward: {total_reward:.2f}")
                elif step >= args.max_steps - 1:
                    timeouts += 1
                elif terminated and total_reward < -50:
                    crashes += 1

                break

        # Update metrics
        episode_time = time.time() - episode_start
        episode_rewards[episode] = total_reward
        episode_steps[episode] = step + 1
        episode_losses[episode] = np.mean(losses) if losses else 0
        training_times[episode] = episode_time

        # Progress logging
        if episode % 10 == 0:
            recent_rewards = episode_rewards[max(0, episode-10):episode+1]
            avg_time = np.mean(training_times[max(0, episode-10):episode+1])

            logger.info(f'Episode {episode}/{args.episodes} - '
                        f'Reward: {total_reward:.2f}, '
                        f'Avg10: {np.mean(recent_rewards):.2f}, '
                        f'Epsilon: {agent.epsilon:.4f}, '
                        f'Time: {avg_time:.2f}s')

            # Statistics
            if episode > 0:
                success_rate = successful_landings / (episode + 1)
                logger.info(f'Stats - Successes: {successful_landings}, '
                            f'Crashes: {crashes}, '
                            f'Timeouts: {timeouts}, '
                            f'Success Rate: {success_rate:.2%}')

        # Performance stats
        if (args.performance_monitor and PSUTIL_AVAILABLE and
                episode % 50 == 0 and episode > 0):
            total_time = time.time() - start_time
            episodes_per_hour = (episode + 1) / (total_time / 3600)
            logger.info(f'Performance - Episodes/hour: {episodes_per_hour:.1f}, '
                        f'CPU: {np.mean(cpu_usage[-5:]):.1f}%, '
                        f'Memory: {np.mean(memory_usage[-5:]):.1f}%')

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(model_dir, 'best_model'))
            logger.info(
                f'🏆 New best model saved with reward: {best_reward:.2f}')

        # Auto-save at intervals
        if (episode + 1) % args.save_interval == 0:
            episode_model_path = os.path.join(
                episode_saves_dir, f'model_episode_{episode + 1}')
            agent.save(episode_model_path)
            logger.info(f'💾 Auto-saved model at episode {episode + 1}')

        # Evaluation
        if args.eval_freq > 0 and episode % args.eval_freq == 0 and episode > 0:
            eval_reward, eval_steps, success_rate = evaluate(
                env, agent, args.eval_episodes, args.seed, args.max_steps)
            logger.info(f'📊 Evaluation - Reward: {eval_reward:.2f}, '
                        f'Steps: {eval_steps:.2f}, '
                        f'Success: {success_rate:.2%}')

    # Save final model
    agent.save(os.path.join(model_dir, 'final_model'))
    logger.info('🎯 Training completed - Final model saved')

    # Final statistics
    success_rate = successful_landings / args.episodes
    total_time = time.time() - start_time
    logger.info(f'\n🎉 FINAL TRAINING STATISTICS:')
    logger.info(
        f'Successful Landings: {successful_landings}/{args.episodes} ({success_rate:.2%})')
    logger.info(f'Crashes: {crashes}, Timeouts: {timeouts}')
    logger.info(f'Best Reward: {best_reward:.2f}')
    logger.info(f'Total Training Time: {total_time/3600:.2f} hours')

    if args.performance_monitor and PSUTIL_AVAILABLE:
        logger.info(f'Average CPU Usage: {np.mean(cpu_usage):.1f}%')
        logger.info(f'Average Memory Usage: {np.mean(memory_usage):.1f}%')

    return agent, best_reward


def evaluate(env, agent, num_episodes, seed, max_steps):
    """Evaluate the agent's performance."""
    rewards = []
    steps = []
    successes = 0

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0
        step_count = 0

        for step in range(max_steps):
            action = agent.act(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward
            step_count += 1

            if done:
                if terminated and reward > 0:
                    successes += 1
                break

        rewards.append(total_reward)
        steps.append(step_count)

    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    success_rate = successes / num_episodes

    return avg_reward, avg_steps, success_rate


def train(args, logger):
    """Main training function with unified optimizations."""
    # Setup directories
    run_dir = os.path.join(args.output_dir, 'current_run')
    model_dir = os.path.join(run_dir, 'models')
    episode_saves_dir = os.path.join(run_dir, 'episode_saves')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(episode_saves_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(run_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    # Create environment
    render_mode = 'human' if args.render else None
    env = gym.make('LunarLander-v3', render_mode=render_mode)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    logger.info(f'🚀 Environment: LunarLander-v3')
    logger.info(f'State size: {state_size}, Action size: {action_size}')

    # Create enhanced agent
    agent = create_enhanced_agent(state_size, action_size, args)

    # Try to load existing model
    if not args.restart:
        model_path = args.model_path or find_best_model(args.output_dir)
        if model_path:
            try:
                agent.load(model_path)
                logger.info(f"✅ Loaded model from {model_path}")

                # Evaluate loaded model
                eval_reward, _, success_rate = evaluate(
                    env, agent, 5, args.seed, args.max_steps)
                logger.info(f"📊 Loaded model evaluation - "
                            f"Reward: {eval_reward:.2f}, Success: {success_rate:.2%}")

            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")

    # Start training
    logger.info('🎯 Starting unified optimized training loop...')
    agent, final_best_reward = optimized_train_loop(
        env, agent, args, logger, episode_saves_dir)

    env.close()
    logger.info(
        f'🏁 Training completed with best reward: {final_best_reward:.2f}')


def main():
    """Main function with unified optimizations."""
    args = parse_args()

    # Apply CPU optimizations based on arguments
    optimization_success = apply_cpu_optimizations(args)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger()

    # Performance monitoring setup
    if args.performance_monitor:
        if PSUTIL_AVAILABLE:
            logger.info("✅ Performance monitoring enabled")
        else:
            logger.warning(
                "⚠️ Performance monitoring requested but psutil not available")

    # Log configuration
    logger.info("🔧 UNIFIED TRAINING CONFIGURATION:")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"CPU Threads: {args.cpu_threads}")
    logger.info(f"Mixed Precision: {args.mixed_precision}")
    logger.info(f"XLA Compilation: {args.enable_xla}")
    logger.info(f"Batch Normalization: {args.batch_norm}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Gradient Clipping: {args.gradient_clip}")

    if args.mode == 'train':
        train(args, logger)
    else:
        logger.error("Play mode not implemented in this unified version")
        logger.info("Use play_best_model.py for playing with trained models")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
