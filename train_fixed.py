"""
FIXED Training script for LunarLander-v3 - Critical Issues Resolved
Key Changes:
1. Fixed success detection (was looking for >200, should be >0 on termination)
2. Fixed reward tracking mismatch (now tracks scaled_reward consistently)
3. Simplified network architecture (512→256→128 down to 64→64)
4. Minimal reward shaping (removed complex bonus system)
5. Proper gamma value (0.99 instead of 0.995)
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
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# CPU optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("CPU optimization applied successfully")
except Exception as e:
    print(f"CPU optimization failed: {e}, using defaults")


class EfficientReplayBuffer:
    """Efficient NumPy-based replay buffer."""

    def __init__(self, state_size, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

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
    """FIXED DQN agent with simplified architecture."""

    def __init__(self,
                 state_size,
                 action_size,
                 memory_size=10000,
                 gamma=0.99,  # FIXED: Was 0.995, now proper value
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,  # FIXED: More reasonable decay
                 learning_rate=0.0005,
                 batch_size=64,
                 update_target_freq=20):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = EfficientReplayBuffer(state_size, memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.step_counter = 0

        # Build networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_network()

    def _build_model(self):
        """FIXED: Simplified network architecture - no more overfitting."""
        model = Sequential()

        # Simple, proven architecture for LunarLander
        model.add(tf.keras.Input(shape=(self.state_size,)))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear'))

        optimizer = Adam(learning_rate=self.learning_rate, epsilon=1e-5)
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
        """FIXED: Clean training loop without reward clipping."""
        if len(self.memory) < self.batch_size * 2:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size)

        # FIXED: No reward clipping - let agent learn from real rewards

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
    """Parse command line arguments with FIXED defaults."""
    parser = argparse.ArgumentParser(
        description='FIXED LunarLander-v3 Training')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'play'])
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--restart', action='store_true')
    # FIXED: Reasonable limit
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--memory-size', type=int, default=50000)
    parser.add_argument('--gamma', type=float,
                        default=0.99)  # FIXED: Proper gamma
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon-min', type=float, default=0.01)
    parser.add_argument('--epsilon-decay', type=float,
                        default=0.995)  # FIXED: Better decay
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--update-target-freq', type=int, default=20)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--eval-freq', type=int, default=50)
    parser.add_argument('--eval-episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def setup_logger():
    """Setup logging."""
    import logging
    logger = logging.getLogger('moonlander_fixed')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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


def optimized_train_loop(env, agent, args, logger):
    """FIXED training loop with corrected success detection and reward tracking."""

    episode_rewards = np.zeros(args.episodes, dtype=np.float32)
    episode_steps = np.zeros(args.episodes, dtype=np.int32)
    episode_losses = np.zeros(args.episodes, dtype=np.float32)

    best_reward = -float('inf')

    # FIXED: Proper tracking variables
    successful_landings = 0
    crashes = 0
    timeouts = 0

    model_dir = os.path.join(args.output_dir, 'current_run', 'models')

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed if episode == 0 else None)

        total_reward = 0  # FIXED: Will track scaled_reward consistently
        losses = []

        for step in range(args.max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # FIXED: MINIMAL reward shaping - only amplify existing signals
            original_reward = reward
            scaled_reward = reward

            # Only amplify success/failure, don't create new rewards
            # FIXED: Successful landing (was looking for >200)
            if reward > 100:
                scaled_reward = reward * 1.2  # Slightly amplify success
            elif reward < -100:  # Crash
                scaled_reward = reward  # Keep crash penalty as-is
            # Everything else unchanged - let environment teach the agent

            # Store experience with scaled reward
            agent.remember(state, action, scaled_reward, next_state, done)

            # Train agent
            if step % 4 == 0:  # Train every 4 steps
                loss = agent.replay()
                if loss > 0:
                    losses.append(loss)

            state = next_state
            total_reward += scaled_reward  # FIXED: Track scaled_reward consistently

            if done:
                # FIXED: Proper success detection
                if terminated and original_reward > 0:  # FIXED: Was checking >200
                    successful_landings += 1
                    logger.info(
                        f"Episode {episode}: SUCCESSFUL LANDING! Original reward: {original_reward:.2f}, Scaled: {scaled_reward:.2f}")
                elif step >= args.max_steps - 1:
                    timeouts += 1
                elif original_reward <= -100:
                    crashes += 1
                break

        # Update metrics
        episode_rewards[episode] = total_reward
        episode_steps[episode] = step + 1
        episode_losses[episode] = np.mean(losses) if losses else 0

        # Progress logging
        if episode % 10 == 0:
            recent_rewards = episode_rewards[max(0, episode-10):episode+1]
            logger.info(f'Episode {episode}/{args.episodes} - '
                        f'Reward: {total_reward:.2f}, '
                        f'Avg10: {np.mean(recent_rewards):.2f}, '
                        f'Epsilon: {agent.epsilon:.4f}')

            # FIXED: Meaningful statistics
            if episode > 0:
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

        # Evaluation
        if args.eval_freq > 0 and episode % args.eval_freq == 0 and episode > 0:
            eval_reward, eval_steps, success_rate = evaluate(
                env, agent, args.eval_episodes, args.seed, args.max_steps)
            logger.info(f'Evaluation - Reward: {eval_reward:.2f}, '
                        f'Steps: {eval_steps:.2f}, '
                        f'Success: {success_rate:.2%}')

    # Save final model
    agent.save(os.path.join(model_dir, 'final_model'))
    logger.info('Training completed - Final model saved')

    # FIXED: Final statistics
    success_rate = successful_landings / args.episodes
    logger.info(f'\nFinal Training Statistics:')
    logger.info(
        f'Successful Landings: {successful_landings}/{args.episodes} ({success_rate:.2%})')
    logger.info(f'Crashes: {crashes}')
    logger.info(f'Timeouts: {timeouts}')

    return agent, best_reward


def evaluate(env, agent, num_episodes, seed, max_steps):
    """FIXED evaluation with proper success detection."""
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
            total_reward += reward  # Track original rewards for evaluation
            step_count += 1

            if done:
                # FIXED: Proper success detection
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
    """Main training function with FIXED setup."""
    # Setup directories
    run_dir = os.path.join(args.output_dir, 'current_run')
    model_dir = os.path.join(run_dir, 'models')

    os.makedirs(model_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(run_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    # Create environment
    render_mode = 'human' if args.render else None
    env = gym.make('LunarLander-v3', render_mode=render_mode)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    logger.info(f'State size: {state_size}, Action size: {action_size}')

    # Create agent with FIXED parameters
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        memory_size=args.memory_size,
        gamma=args.gamma,  # Now uses fixed 0.99
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        update_target_freq=args.update_target_freq
    )

    # Try to load existing model
    if not args.restart:
        model_path = args.model_path or find_best_model(args.output_dir)
        if model_path:
            try:
                agent.load(model_path)
                logger.info(f"Loaded model from {model_path}")

                # Evaluate loaded model
                eval_reward, _, success_rate = evaluate(
                    env, agent, 5, args.seed, args.max_steps)
                logger.info(
                    f"Loaded model evaluation - Reward: {eval_reward:.2f}, Success: {success_rate:.2%}")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")

    # Start training
    logger.info('Starting FIXED training loop...')
    agent, final_best_reward = optimized_train_loop(env, agent, args, logger)

    env.close()
    logger.info(
        f'Training completed with best reward: {final_best_reward:.2f}')


def main():
    """Main function."""
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger()

    if args.mode == 'train':
        train(args, logger)
    else:
        logger.error("Play mode not implemented in this fixed version")


if __name__ == "__main__":
    main()
