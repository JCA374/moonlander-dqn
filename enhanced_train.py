"""
Enhanced training script for LunarLander-v3 with:
- Better CPU/memory utilization for Intel i7-8565U
- Improved crash reduction strategies
- Hovering detection and statistics
- Fixed reward tracking
- Advanced stability techniques
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
from collections import deque
import threading
from queue import Queue

# Import TensorFlow with optimization settings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Environment
import gymnasium as gym

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available - performance monitoring disabled")


def apply_aggressive_cpu_optimizations(args):
    """Apply aggressive CPU optimizations for Intel i7-8565U."""
    print("🚀 Applying aggressive Intel i7-8565U optimizations...")

    # Environment variables for maximum performance
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['OMP_NUM_THREADS'] = str(args.cpu_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.cpu_threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(args.cpu_threads)

    try:
        # Aggressive threading for your 8-thread CPU
        tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)

        # Enable all optimizations
        tf.config.optimizer.set_jit(True)  # XLA always on
        tf.config.optimizer.set_experimental_options({
            'layout_optimizer': True,
            'constant_folding': True,
            'shape_optimization': True,
            'arithmetic_optimization': True,
            'dependency_optimization': True,
            'loop_optimization': True,
            'function_optimization': True,
            'debug_stripper': True,
            'disable_model_pruning': False,
            'scoped_allocator_optimization': True,
            'pin_to_host_optimization': True,
            'implementation_selector': True,
            'auto_mixed_precision': True,
        })

        print(f"✓ Aggressive optimizations applied")
        print(f"✓ Using {args.cpu_threads} threads for operations")
        print(f"✓ XLA JIT compilation enabled")
        print(f"✓ All TensorFlow optimizations enabled")

        return True

    except Exception as e:
        print(f"⚠ Some optimizations failed: {e}, continuing...")
        return False


class PrioritizedReplayBuffer:
    """Enhanced replay buffer with prioritized experience replay."""

    def __init__(self, state_size, max_size, alpha=0.6):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.alpha = alpha

        # Efficient storage with pre-allocation
        self.states = np.zeros((max_size, state_size), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_size), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)
        self.priorities = np.zeros(max_size, dtype=np.float32)

        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done, priority=None):
        if priority is None:
            priority = self.max_priority

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.priorities[self.ptr] = priority ** self.alpha

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, beta=0.4):
        if self.size < batch_size:
            return None

        # Prioritized sampling
        probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
        indices = np.random.choice(self.size, batch_size, p=probs)

        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.size


class EnhancedDQNAgent:
    """Enhanced DQN agent with crash reduction and stability features."""

    def __init__(self,
                 state_size,
                 action_size,
                 memory_size=100000,  # Increased memory
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.997,  # Slower decay
                 learning_rate=0.0003,  # Lower LR for stability
                 batch_size=128,  # Larger batch
                 update_target_freq=15,
                 tau=0.001,  # Soft update
                 use_double_dqn=True,
                 use_dueling=True):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.tau = tau
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.step_counter = 0

        # Enhanced memory with prioritization
        self.memory = PrioritizedReplayBuffer(state_size, memory_size)

        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_network(soft=False)

        # Training metrics
        self.losses = deque(maxlen=100)
        self.q_values = deque(maxlen=100)

        print(f"✓ Enhanced DQN Agent created")
        print(f"  - Double DQN: {use_double_dqn}")
        print(f"  - Dueling Architecture: {use_dueling}")
        print(f"  - Memory Size: {memory_size:,}")
        print(f"  - Batch Size: {batch_size}")

    def _build_model(self):
        """Build enhanced model with dueling architecture option."""
        inputs = tf.keras.Input(shape=(self.state_size,))

        # Shared layers
        x = Dense(128, activation='relu',
                  kernel_initializer='he_uniform')(inputs)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)

        if self.use_dueling:
            # Dueling streams
            # Value stream
            value = Dense(64, activation='relu')(x)
            value = Dense(1)(value)

            # Advantage stream
            advantage = Dense(64, activation='relu')(x)
            advantage = Dense(self.action_size)(advantage)

            # Combine streams
            outputs = value + \
                (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        else:
            # Standard architecture
            x = Dense(64, activation='relu',
                      kernel_initializer='he_uniform')(x)
            outputs = Dense(self.action_size)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile with advanced optimizer
        optimizer = Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )

        model.compile(
            loss='huber',
            optimizer=optimizer,
            metrics=['mae']
        )

        return model

    def _update_target_network(self, soft=True):
        """Update target network with soft or hard update."""
        if soft:
            # Soft update
            for target_weight, weight in zip(self.target_model.weights, self.model.weights):
                target_weight.assign(self.tau * weight +
                                     (1 - self.tau) * target_weight)
        else:
            # Hard update
            self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience with TD error priority."""
        # Calculate TD error for prioritization
        q_value = self.model.predict(
            state.reshape(1, -1), verbose=0)[0][action]

        if self.use_double_dqn:
            next_action = np.argmax(self.model.predict(
                next_state.reshape(1, -1), verbose=0)[0])
            next_q = self.target_model.predict(
                next_state.reshape(1, -1), verbose=0)[0][next_action]
        else:
            next_q = np.max(self.target_model.predict(
                next_state.reshape(1, -1), verbose=0)[0])

        target = reward if done else reward + self.gamma * next_q
        td_error = abs(target - q_value)

        self.memory.add(state, action, reward,
                        next_state, done, td_error + 1e-6)

    def act(self, state, evaluate=False):
        """Enhanced action selection with safety considerations."""
        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        self.q_values.append(np.mean(np.abs(q_values)))

        # Safety enhancement: bias against extreme actions when close to ground
        if not evaluate and state[1] < 0.3:  # Low altitude
            # Slightly penalize doing nothing when very close to ground
            if state[1] < 0.1 and abs(state[3]) > 0.5:  # Falling fast
                q_values[0] -= 0.1  # Penalize no action

        return np.argmax(q_values)

    def replay(self):
        """Enhanced training with prioritized experience replay."""
        if len(self.memory) < self.batch_size * 2:
            return 0.0

        # Sample with priorities
        batch_data = self.memory.sample(self.batch_size, beta=min(
            1.0, 0.4 + 0.6 * self.step_counter / 50000))
        if batch_data is None:
            return 0.0

        states, actions, rewards, next_states, dones, indices, weights = batch_data

        # Current Q values
        current_q = self.model.predict(states, verbose=0)

        # Target Q values
        if self.use_double_dqn:
            # Double DQN: use online network to select actions
            next_actions = np.argmax(self.model.predict(
                next_states, verbose=0), axis=1)
            next_q = self.target_model.predict(next_states, verbose=0)
            target_q = next_q[np.arange(self.batch_size), next_actions]
        else:
            # Standard DQN
            next_q = self.target_model.predict(next_states, verbose=0)
            target_q = np.max(next_q, axis=1)

        # Update Q values
        targets = current_q.copy()
        td_errors = []

        for i in range(self.batch_size):
            target = rewards[i] if dones[i] else rewards[i] + \
                self.gamma * target_q[i]
            td_error = abs(target - current_q[i][actions[i]])
            td_errors.append(td_error)
            targets[i][actions[i]] = target

        # Train with importance sampling weights
        sample_weights = weights.reshape(-1, 1)
        history = self.model.fit(
            states, targets,
            sample_weight=sample_weights,
            epochs=1,
            verbose=0,
            batch_size=self.batch_size
        )

        # Update priorities
        self.memory.update_priorities(indices, td_errors)

        loss = history.history['loss'][0]
        self.losses.append(loss)

        # Update target network
        self.step_counter += 1
        if self.tau > 0:
            self._update_target_network(soft=True)  # Soft update every step
        if self.step_counter % self.update_target_freq == 0:
            self._update_target_network(soft=False)  # Hard update periodically

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss

    def save(self, filepath):
        """Save model weights."""
        if not filepath.endswith('.weights.h5'):
            filepath = filepath + '.weights.h5'
        self.model.save_weights(filepath)

    def load(self, filepath):
        """Load model weights."""
        if not filepath.endswith('.weights.h5') and not os.path.exists(filepath):
            if os.path.exists(filepath + '.weights.h5'):
                filepath = filepath + '.weights.h5'
        self.model.load_weights(filepath)
        self._update_target_network(soft=False)


def enhanced_reward_shaping(state, next_state, reward, done, step, max_steps):
    """Advanced reward shaping to reduce crashes and hovering."""
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state

    shaped_reward = reward

    # 1. Crash prevention rewards
    if y < 0.5 and not done:
        # Reward slower descent when close to ground
        if abs(vy) < 0.5:
            shaped_reward += 0.5
        elif abs(vy) < 1.0:
            shaped_reward += 0.2

        # Reward being upright when close to ground
        if abs(angle) < 0.2:
            shaped_reward += 0.3

        # Reward being centered
        if abs(x) < 0.2:
            shaped_reward += 0.5

    # 2. Anti-hovering rewards
    # Detect hovering: low altitude, low velocity, not on ground
    if 0.05 < y < 0.3 and abs(vy) < 0.05 and not (left_leg or right_leg):
        hover_penalty = -0.1 * (1 + step / max_steps)  # Increasing penalty
        shaped_reward += hover_penalty

    # 3. Time pressure that increases gradually
    if step > max_steps * 0.7:  # Last 30% of episode
        time_penalty = -0.01 * (step - max_steps * 0.7)
        shaped_reward += time_penalty

    # 4. Strong landing encouragement
    if left_leg and right_leg and abs(vx) < 0.2 and abs(vy) < 0.2:
        shaped_reward += 10.0  # Big bonus for safe landing position

    # 5. Penalize extreme tilting (crash risk)
    if abs(angle) > 0.4:
        shaped_reward -= abs(angle) * 0.5

    return shaped_reward


def detect_outcome(state, reward, done, step, max_steps):
    """Detect episode outcome including hovering."""
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = state

    # Success detection
    if done and reward > 0:
        if left_leg and right_leg and abs(vx) < 0.5 and abs(vy) < 0.5:
            return "success"
        elif reward > 100:
            return "success"

    # Crash detection
    if done and reward < -50:
        return "crash"

    # Timeout detection
    if step >= max_steps - 1:
        # Check if it was hovering at timeout
        if 0.05 < y < 0.5 and abs(vy) < 0.1:
            return "hover_timeout"
        else:
            return "timeout"

    # Other termination
    if done:
        return "other"

    return None


def parallel_environment_step(env_queue, result_queue):
    """Worker thread for parallel environment steps."""
    while True:
        task = env_queue.get()
        if task is None:
            break

        env, state, action = task
        next_state, reward, terminated, truncated, info = env.step(action)
        result_queue.put((next_state, reward, terminated, truncated, info))


def enhanced_training_loop(env, agent, args, logger):
    """Enhanced training loop with better resource usage and crash reduction."""

    # Training setup
    start_time = time.time()
    model_dir = os.path.join(args.output_dir, 'current_run', 'models')
    episode_saves_dir = os.path.join(
        args.output_dir, 'current_run', 'episode_saves')

    # Metrics tracking
    episode_rewards = []
    outcomes = {
        "success": 0,
        "crash": 0,
        "hover_timeout": 0,
        "timeout": 0,
        "other": 0
    }

    best_episode_reward = -float('inf')
    best_avg_reward = -float('inf')
    recent_rewards = deque(maxlen=100)

    # Performance monitoring
    cpu_usage = []
    memory_usage = []

    # Training variables
    training_stable = False
    stability_counter = 0

    for episode in range(args.episodes):
        episode_start = time.time()

        # Performance monitoring
        if args.performance_monitor and PSUTIL_AVAILABLE and episode % 10 == 0:
            cpu_usage.append(psutil.cpu_percent(interval=0.1))
            memory_usage.append(psutil.virtual_memory().percent)

        state, _ = env.reset(seed=args.seed if episode == 0 else None)

        episode_reward = 0
        losses = []

        for step in range(args.max_steps):
            # Action selection
            action = agent.act(state)

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Enhanced reward shaping
            shaped_reward = enhanced_reward_shaping(
                state, next_state, reward, done, step, args.max_steps
            )

            # Store experience
            agent.remember(state, action, shaped_reward, next_state, done)

            # Train more frequently for better CPU usage
            if step % 2 == 0 and len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss > 0:
                    losses.append(loss)

            state = next_state
            episode_reward += reward  # Track original reward

            if done:
                # Detect outcome
                outcome = detect_outcome(
                    next_state, reward, done, step, args.max_steps)
                if outcome:
                    outcomes[outcome] += 1

                break

        # Episode complete
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)

        # Calculate metrics
        avg_reward = np.mean(recent_rewards)
        success_rate = outcomes["success"] / (episode + 1)
        crash_rate = outcomes["crash"] / (episode + 1)
        hover_rate = outcomes["hover_timeout"] / (episode + 1)

        # Check for stability
        if len(recent_rewards) >= 100:
            reward_std = np.std(recent_rewards)
            if reward_std < 50 and avg_reward > 0:
                stability_counter += 1
                if stability_counter > 50:
                    training_stable = True
            else:
                stability_counter = 0

        # Logging
        if episode % 10 == 0:
            episode_time = time.time() - episode_start
            avg_loss = np.mean(losses) if losses else 0
            avg_q = np.mean(agent.q_values) if len(agent.q_values) > 0 else 0

            logger.info(
                f'Episode {episode}/{args.episodes} - '
                f'Reward: {episode_reward:.2f}, '
                f'Avg100: {avg_reward:.2f}, '
                f'Success: {success_rate:.2%}, '
                f'Crash: {crash_rate:.2%}, '
                f'Hover: {hover_rate:.2%}, '
                f'ε: {agent.epsilon:.4f}, '
                f'Loss: {avg_loss:.4f}, '
                f'Q: {avg_q:.2f}, '
                f'Time: {episode_time:.2f}s'
            )

        # Save best model (based on episode reward, not cumulative)
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            agent.save(os.path.join(model_dir, 'best_model'))
            logger.info(
                f'🏆 New best episode reward: {best_episode_reward:.2f}')

        # Save best average model
        if avg_reward > best_avg_reward and len(recent_rewards) >= 100:
            best_avg_reward = avg_reward
            agent.save(os.path.join(model_dir, 'best_avg_model'))
            logger.info(f'📊 New best average reward: {best_avg_reward:.2f}')

        # Auto-save
        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(episode_saves_dir,
                       f'model_episode_{episode + 1}'))
            logger.info(f'💾 Auto-saved model at episode {episode + 1}')

        # Performance stats
        if args.performance_monitor and PSUTIL_AVAILABLE and episode % 50 == 0 and episode > 0:
            total_time = time.time() - start_time
            episodes_per_hour = episode / (total_time / 3600)
            avg_cpu = np.mean(cpu_usage[-5:]) if cpu_usage else 0
            avg_mem = np.mean(memory_usage[-5:]) if memory_usage else 0

            logger.info(
                f'Performance - Episodes/hour: {episodes_per_hour:.1f}, '
                f'CPU: {avg_cpu:.1f}%, '
                f'Memory: {avg_mem:.1f}%, '
                f'Stable: {training_stable}'
            )

        # Adaptive learning
        if training_stable and episode % 100 == 0:
            # Reduce learning rate for fine-tuning
            current_lr = K.get_value(agent.model.optimizer.learning_rate)
            new_lr = max(1e-5, current_lr * 0.95)
            K.set_value(agent.model.optimizer.learning_rate, new_lr)
            logger.info(f'📉 Reduced learning rate to {new_lr:.6f}')

    # Training complete
    agent.save(os.path.join(model_dir, 'final_model'))

    # Final statistics
    total_time = time.time() - start_time
    final_success_rate = outcomes["success"] / args.episodes
    final_crash_rate = outcomes["crash"] / args.episodes
    final_hover_rate = outcomes["hover_timeout"] / args.episodes

    logger.info(f'\n🎉 FINAL TRAINING STATISTICS:')
    logger.info(f'Episodes: {args.episodes}')
    logger.info(
        f'Success Rate: {final_success_rate:.2%} ({outcomes["success"]} landings)')
    logger.info(
        f'Crash Rate: {final_crash_rate:.2%} ({outcomes["crash"]} crashes)')
    logger.info(
        f'Hover Timeout Rate: {final_hover_rate:.2%} ({outcomes["hover_timeout"]} hover timeouts)')
    logger.info(f'Other Outcomes: {outcomes["other"]}')
    logger.info(f'Best Episode Reward: {best_episode_reward:.2f}')
    logger.info(f'Best Average Reward (100 eps): {best_avg_reward:.2f}')
    logger.info(f'Final Average Reward (100 eps): {avg_reward:.2f}')
    logger.info(f'Total Training Time: {total_time/3600:.2f} hours')

    if args.performance_monitor and PSUTIL_AVAILABLE:
        logger.info(f'Average CPU Usage: {np.mean(cpu_usage):.1f}%')
        logger.info(f'Average Memory Usage: {np.mean(memory_usage):.1f}%')

    return agent, best_episode_reward


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced LunarLander Training')

    # Core arguments
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)

    # Enhanced performance arguments
    parser.add_argument('--cpu-threads', type=int, default=8,
                        help='Number of CPU threads (default: 8 for i7-8565U)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Larger batch size for better CPU usage')
    parser.add_argument('--memory-size', type=int, default=100000,
                        help='Larger memory for more diverse experience')

    # Agent hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon-min', type=float, default=0.01)
    parser.add_argument('--epsilon-decay', type=float, default=0.997)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--update-target-freq', type=int, default=15)
    parser.add_argument('--tau', type=float, default=0.001,
                        help='Soft update parameter')

    # Features
    parser.add_argument('--double-dqn', action='store_true', default=True)
    parser.add_argument('--dueling', action='store_true', default=True)
    parser.add_argument('--prioritized-replay',
                        action='store_true', default=True)

    # Monitoring
    parser.add_argument('--save-interval', type=int, default=50)
    parser.add_argument('--performance-monitor',
                        action='store_true', default=True)

    return parser.parse_args()


def setup_logger():
    """Setup logging."""
    import logging
    logger = logging.getLogger('moonlander_enhanced')
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


def main():
    """Main training function."""
    args = parse_args()

    # Apply aggressive optimizations
    apply_aggressive_cpu_optimizations(args)

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger()

    # Log configuration
    logger.info("🚀 ENHANCED MOONLANDER TRAINING")
    logger.info("="*60)
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"CPU Threads: {args.cpu_threads}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Memory Size: {args.memory_size:,}")
    logger.info(f"Double DQN: {args.double_dqn}")
    logger.info(f"Dueling Architecture: {args.dueling}")
    logger.info(f"Prioritized Replay: {args.prioritized_replay}")
    logger.info("="*60)

    # Create directories
    run_dir = os.path.join(args.output_dir, 'current_run')
    model_dir = os.path.join(run_dir, 'models')
    episode_saves_dir = os.path.join(run_dir, 'episode_saves')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(episode_saves_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(run_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    # Create environment
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    logger.info(f'Environment: LunarLander-v3')
    logger.info(f'State size: {state_size}, Action size: {action_size}')

    # Create enhanced agent
    agent = EnhancedDQNAgent(
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
        tau=args.tau,
        use_double_dqn=args.double_dqn,
        use_dueling=args.dueling
    )

    # Start training
    logger.info('🎯 Starting enhanced training...')
    agent, best_reward = enhanced_training_loop(env, agent, args, logger)

    env.close()
    logger.info(f'🏁 Training completed with best reward: {best_reward:.2f}')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
