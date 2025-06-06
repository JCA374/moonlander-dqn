"""
OPTIMIZED Training script for LunarLander-v3 - Intel i7-8565U Specific
Key Optimizations:
1. CPU threading optimized for 4 cores, 8 threads
2. Memory buffer sized for 32GB RAM
3. Batch size optimized for CPU performance
4. TensorFlow configured for Intel optimizations
5. Faster training loop with better utilization
"""
import gymnasium as gym
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
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# === INTEL i7-8565U SPECIFIC OPTIMIZATIONS ===


def optimize_for_i7_8565u():
    """Apply CPU optimizations specifically for Intel i7-8565U."""
    print("🚀 Applying Intel i7-8565U optimizations...")

    # CPU Threading - Optimized for 4 cores, 8 threads
    tf.config.threading.set_intra_op_parallelism_threads(
        6)  # 6 of 8 threads for operations
    tf.config.threading.set_inter_op_parallelism_threads(
        2)  # 2 threads for coordination

    # Intel-specific optimizations
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Enable Intel oneDNN optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '6'        # OpenMP threads
    os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'

    # Memory optimizations for 32GB RAM
    # Even for CPU, helps memory management
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    # Enable XLA JIT compilation for faster execution
    tf.config.optimizer.set_jit(True)

    # Use float32 precision for better CPU performance
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)

    print("✓ CPU optimizations applied successfully")
    print(
        f"✓ Using {tf.config.threading.get_intra_op_parallelism_threads()} threads for operations")
    print(
        f"✓ Using {tf.config.threading.get_inter_op_parallelism_threads()} threads for coordination")


# Apply optimizations immediately
optimize_for_i7_8565u()


class OptimizedReplayBuffer:
    """Optimized replay buffer for 32GB RAM system."""

    def __init__(self, state_size, max_size=200000):  # Increased from 50k to 200k
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Pre-allocate with optimal data types for CPU
        self.states = np.zeros((max_size, state_size), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_size), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)

        print(
            f"✓ Replay buffer initialized: {max_size:,} experiences ({max_size * 44 / 1024 / 1024:.1f} MB)")

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
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


class OptimizedDQNAgent:
    """DQN Agent optimized for Intel i7-8565U performance."""

    def __init__(self,
                 state_size,
                 action_size,
                 memory_size=200000,     # Increased for 32GB RAM
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.02,       # Higher minimum for more exploration
                 epsilon_decay=0.998,    # Slower decay
                 learning_rate=0.001,    # Slightly higher
                 batch_size=128,         # Optimized for CPU
                 update_target_freq=10,  # More frequent updates
                 train_freq=2):          # Train every 2 steps instead of 4

        self.state_size = state_size
        self.action_size = action_size
        self.memory = OptimizedReplayBuffer(state_size, memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.train_freq = train_freq
        self.step_counter = 0

        # Build optimized networks
        self.model = self._build_optimized_model()
        self.target_model = self._build_optimized_model()
        self._update_target_network()

        print(f"✓ Optimized DQN Agent created")
        print(f"  Memory: {memory_size:,} experiences")
        print(f"  Batch size: {batch_size}")
        print(f"  Train frequency: every {train_freq} steps")
        print(f"  Target update: every {update_target_freq} episodes")

    def _build_optimized_model(self):
        """Build network optimized for CPU performance."""
        model = Sequential([
            tf.keras.Input(shape=(self.state_size,)),

            # Larger first layer for better feature extraction
            Dense(128, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),  # Helps with training stability

            # Second hidden layer
            Dense(128, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),

            # Optional dropout for regularization (comment out if too slow)
            # Dropout(0.1),

            # Output layer
            Dense(self.action_size, activation='linear',
                  kernel_initializer='he_uniform')
        ])

        # Optimized optimizer with gradient clipping
        optimizer = Adam(
            learning_rate=self.learning_rate,
            epsilon=1e-7,
            clipnorm=1.0,  # Gradient clipping for stability
            amsgrad=True   # Better convergence properties
        )

        model.compile(
            loss='huber',
            optimizer=optimizer,
            metrics=['mae']
        )

        # Print model summary
        total_params = model.count_params()
        print(f"  Network parameters: {total_params:,}")

        return model

    def _update_target_network(self):
        """Soft update target network for better stability."""
        # Use soft updates instead of hard copies
        tau = 0.1  # Soft update parameter

        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(main_weights)):
            target_weights[i] = tau * main_weights[i] + \
                (1 - tau) * target_weights[i]

        self.target_model.set_weights(target_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, evaluate=False):
        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Optimized training loop."""
        if len(self.memory) < self.batch_size * 4:  # Wait for more experiences
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size)

        # Compute Q-values in batches for efficiency
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)

        # Vectorized Bellman equation
        targets = current_q.copy()
        batch_indices = np.arange(self.batch_size)

        # Handle terminal states
        terminal_mask = dones
        non_terminal_mask = ~dones

        # Update terminal states
        targets[batch_indices[terminal_mask],
                actions[terminal_mask]] = rewards[terminal_mask]

        # Update non-terminal states
        if np.any(non_terminal_mask):
            targets[batch_indices[non_terminal_mask], actions[non_terminal_mask]] = \
                rewards[non_terminal_mask] + self.gamma * \
                np.max(next_q[non_terminal_mask], axis=1)

        # Train with multiple steps for better CPU utilization
        history = self.model.fit(
            states, targets,
            epochs=2,  # Multiple epochs per batch
            verbose=0,
            batch_size=min(64, self.batch_size)  # Smaller sub-batches
        )

        loss = np.mean(history.history['loss'])

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
    """Parse command line arguments with optimized defaults."""
    parser = argparse.ArgumentParser(
        description='Optimized LunarLander-v3 Training for Intel i7-8565U')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'play'])
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=1000)  # Increased
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--max-steps', type=int, default=500)  # Optimized
    parser.add_argument('--batch-size', type=int,
                        default=128)  # Optimized for CPU
    parser.add_argument('--memory-size', type=int,
                        default=200000)  # Increased for 32GB RAM
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon-min', type=float, default=0.02)  # Optimized
    parser.add_argument('--epsilon-decay', type=float,
                        default=0.998)  # Optimized
    parser.add_argument('--learning-rate', type=float,
                        default=0.001)  # Optimized
    parser.add_argument('--update-target-freq', type=int,
                        default=10)  # More frequent
    parser.add_argument('--train-freq', type=int,
                        default=2)  # Train more often
    parser.add_argument('--render', action='store_true')
    # More frequent evaluation
    parser.add_argument('--eval-freq', type=int, default=25)
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def setup_logger():
    import logging
    logger = logging.getLogger('moonlander_optimized')
    logger.setLevel(logging.INFO)

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
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == 'best_model.weights.h5':
                if best_model_path is None or os.path.getmtime(os.path.join(root, file)) > os.path.getmtime(best_model_path):
                    best_model_path = os.path.join(root, file)

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


def optimized_train_loop(env, agent, args, logger, episode_saves_dir):
    """Optimized training loop for Intel i7-8565U."""

    episode_rewards = np.zeros(args.episodes, dtype=np.float32)
    episode_steps = np.zeros(args.episodes, dtype=np.int32)
    episode_losses = np.zeros(args.episodes, dtype=np.float32)

    best_reward = -float('inf')
    successful_landings = 0
    crashes = 0
    timeouts = 0

    model_dir = os.path.join(args.output_dir, 'current_run', 'models')

    # Performance monitoring
    start_time = time.time()
    steps_processed = 0

    for episode in range(args.episodes):
        episode_start = time.time()
        state, _ = env.reset(seed=args.seed if episode == 0 else None)

        total_reward = 0
        losses = []

        for step in range(args.max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Enhanced reward shaping
            x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state
            original_reward = reward
            scaled_reward = reward

            # Success amplification
            if reward > 50:
                scaled_reward = reward * 1.5
            elif reward < -50:
                scaled_reward = reward * 0.8

            # Landing encouragement
            if y < 0.5 and abs(x) < 0.5:
                scaled_reward += 0.5 * (0.5 - abs(x))

                if y < 0.2 and abs(vy) < 1.0:
                    scaled_reward += 1.0

                    if left_leg or right_leg:
                        scaled_reward += 2.0

                        if left_leg and right_leg and abs(vx) < 0.5:
                            scaled_reward += 5.0

            # Anti-hovering
            if 0.05 < y < 0.3 and abs(vy) < 0.1 and step > 300:
                scaled_reward -= 0.2

            # Time pressure
            if step > 400:
                scaled_reward -= 0.05

            # Timeout penalty
            if step >= args.max_steps - 1:
                scaled_reward -= 50.0

            # Store experience
            agent.remember(state, action, scaled_reward, next_state, done)

            # Optimized training frequency
            if step % args.train_freq == 0:
                loss = agent.replay()
                if loss > 0:
                    losses.append(loss)

            state = next_state
            total_reward += scaled_reward
            steps_processed += 1

            if done:
                # Success detection
                landed_successfully = False

                if terminated:
                    if left_leg and right_leg and abs(vx) < 0.5 and abs(vy) < 0.5 and abs(x) < 0.5:
                        landed_successfully = True
                    elif total_reward > 200:
                        landed_successfully = True

                if landed_successfully:
                    successful_landings += 1
                    logger.info(
                        f"Episode {episode}: ✓ SUCCESSFUL LANDING! Reward: {total_reward:.2f}")
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

        # Performance metrics
        if episode > 0 and episode % 10 == 0:
            recent_rewards = episode_rewards[max(0, episode-10):episode+1]
            elapsed_time = time.time() - start_time
            steps_per_second = steps_processed / elapsed_time
            episodes_per_hour = (episode + 1) / (elapsed_time / 3600)

            logger.info(f'Episode {episode}/{args.episodes} - '
                        f'Reward: {total_reward:.2f}, '
                        f'Avg10: {np.mean(recent_rewards):.2f}, '
                        f'Epsilon: {agent.epsilon:.4f}, '
                        f'Time: {episode_time:.2f}s')

            logger.info(f'Performance - Steps/sec: {steps_per_second:.1f}, '
                        f'Episodes/hour: {episodes_per_hour:.1f}')

            success_rate = successful_landings / (episode + 1)
            logger.info(f'Stats - Successes: {successful_landings}, '
                        f'Crashes: {crashes}, '
                        f'Timeouts: {timeouts}, '
                        f'Success Rate: {success_rate:.2%}')

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(model_dir, 'best_model'))
            logger.info(f'New best model saved with reward: {best_reward:.2f}')

        # Auto-save every 25 episodes
        if (episode + 1) % 25 == 0:
            episode_model_path = os.path.join(
                episode_saves_dir, f'model_episode_{episode + 1}')
            agent.save(episode_model_path)

        # Evaluation
        if args.eval_freq > 0 and episode % args.eval_freq == 0 and episode > 0:
            eval_reward, eval_steps, success_rate = evaluate(
                env, agent, args.eval_episodes, args.seed, args.max_steps)
            logger.info(
                f'Evaluation - Reward: {eval_reward:.2f}, Steps: {eval_steps:.2f}, Success: {success_rate:.2%}')

    # Final statistics
    total_time = time.time() - start_time
    final_success_rate = successful_landings / args.episodes

    logger.info(f'\n🎯 TRAINING COMPLETED!')
    logger.info(f'Total time: {total_time/3600:.2f} hours')
    logger.info(
        f'Final success rate: {final_success_rate:.2%} ({successful_landings}/{args.episodes})')
    logger.info(f'Average steps/second: {steps_processed/total_time:.1f}')

    agent.save(os.path.join(model_dir, 'final_model'))
    return agent, best_reward


def evaluate(env, agent, num_episodes, seed, max_steps):
    """Evaluation function."""
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

    return np.mean(rewards), np.mean(steps), successes / num_episodes


def train(args, logger):
    """Main training function."""
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

    logger.info(f'Environment: LunarLander-v3')
    logger.info(f'State size: {state_size}, Action size: {action_size}')

    # Create optimized agent
    agent = OptimizedDQNAgent(
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
        train_freq=args.train_freq
    )

    # Try to load existing model
    if not args.restart:
        model_path = args.model_path or find_best_model(args.output_dir)
        if model_path:
            try:
                agent.load(model_path)
                logger.info(f"Loaded model from {model_path}")

                eval_reward, _, success_rate = evaluate(
                    env, agent, 5, args.seed, args.max_steps)
                logger.info(
                    f"Loaded model evaluation - Reward: {eval_reward:.2f}, Success: {success_rate:.2%}")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")

    # Start training
    logger.info('🚀 Starting optimized training for Intel i7-8565U...')
    agent, final_best_reward = optimized_train_loop(
        env, agent, args, logger, episode_saves_dir)

    env.close()
    logger.info(
        f'Training completed with best reward: {final_best_reward:.2f}')


def main():
    args = parse_args()

    print("=" * 60)
    print("🚀 OPTIMIZED LUNARLANDER TRAINING - Intel i7-8565U")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Memory size: {args.memory_size:,}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Training frequency: every {args.train_freq} steps")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger()

    if args.mode == 'train':
        train(args, logger)
    else:
        logger.error("Play mode not implemented in this optimized version")


if __name__ == "__main__":
    main()
