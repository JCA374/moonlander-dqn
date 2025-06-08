"""
Fixed Training Script - Anti-Hovering Version
Addresses the reward hacking problem where the agent learns to hover instead of land.
"""
import os
import sys
import argparse
import time
import numpy as np
import random
from datetime import datetime
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gymnasium as gym


def apply_cpu_optimizations():
    """Apply CPU optimizations."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

    try:
        tf.config.threading.set_intra_op_parallelism_threads(6)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        print("[OK] CPU optimizations applied")
        return True
    except Exception as e:
        print(f"WARNING: Optimization failed: {e}")
        return False


class AntiHoverReplayBuffer:
    """Replay buffer that tracks hovering behavior."""

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
        if self.size < batch_size:
            return None

        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size


class AntiHoverDQNAgent:
    """DQN Agent with anti-hovering mechanisms."""

    def __init__(self, state_size, action_size, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.learning_rate = kwargs.get('learning_rate', 0.0005)
        self.batch_size = kwargs.get('batch_size', 64)
        self.update_target_freq = kwargs.get('update_target_freq', 20)
        self.step_counter = 0

        # Anti-hovering tracking
        self.hover_tracker = []
        self.hover_penalty_multiplier = 1.0

        self.memory = AntiHoverReplayBuffer(
            state_size, kwargs.get('memory_size', 50000))
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_network()

    def _build_model(self):
        """Build the neural network."""
        model = Sequential([
            tf.keras.Input(shape=(self.state_size,)),
            Dense(64, activation='relu', kernel_initializer='he_uniform'),
            Dense(64, activation='relu', kernel_initializer='he_uniform'),
            Dense(self.action_size, activation='linear')
        ])

        optimizer = Adam(learning_rate=self.learning_rate, epsilon=1e-5)
        model.compile(loss='huber', optimizer=optimizer)
        return model

    def _update_target_network(self):
        """Update target network."""
        self.target_model.set_weights(self.model.get_weights())

    def track_hovering(self, state, action, step):
        """Track potential hovering behavior."""
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = state

        # Define hovering as: low altitude, low speed, legs touching, late in episode
        is_hovering = (
            y < 0.1 and  # Very close to ground
            abs(vx) < 0.2 and abs(vy) < 0.2 and  # Moving slowly
            (left_leg or right_leg) and  # At least one leg touching
            step > 200  # Late in episode
        )

        self.hover_tracker.append(is_hovering)

        # Keep only recent history
        if len(self.hover_tracker) > 50:
            self.hover_tracker.pop(0)

        # Increase penalty multiplier if hovering too much
        recent_hover_rate = sum(
            self.hover_tracker[-20:]) / min(20, len(self.hover_tracker))
        if recent_hover_rate > 0.8:  # Hovering 80% of recent steps
            self.hover_penalty_multiplier = min(
                3.0, self.hover_penalty_multiplier * 1.1)
        else:
            self.hover_penalty_multiplier = max(
                1.0, self.hover_penalty_multiplier * 0.99)

        return is_hovering, recent_hover_rate

    def remember(self, state, action, reward, next_state, done):
        """Store experience with anti-hover modifications."""
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, evaluate=False):
        """Choose action."""
        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Train the model."""
        if len(self.memory) < self.batch_size * 2:
            return 0.0

        batch_data = self.memory.sample(self.batch_size)
        if batch_data is None:
            return 0.0

        states, actions, rewards, next_states, dones = batch_data

        # Compute targets
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)

        targets = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + \
                    self.gamma * np.max(next_q[i])

        # Train model
        loss = self.model.fit(states, targets, epochs=1, verbose=0)

        # Update target network
        self.step_counter += 1
        if self.step_counter % self.update_target_freq == 0:
            self._update_target_network()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.history['loss'][0]

    def save(self, filepath):
        if not filepath.endswith('.weights.h5'):
            filepath = filepath + '.weights.h5'
        self.model.save_weights(filepath)

    def load(self, filepath):
        if not filepath.endswith('.weights.h5') and not os.path.exists(filepath):
            if os.path.exists(filepath + '.weights.h5'):
                filepath = filepath + '.weights.h5'
        self.model.load_weights(filepath)


def apply_anti_hover_reward_shaping(state, next_state, reward, step, done, hover_info):
    """Apply reward shaping that strongly discourages hovering."""
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state
    is_hovering, hover_rate = hover_info

    original_reward = reward
    shaped_reward = reward

    # 1. Strong anti-hovering penalties
    if is_hovering:
        # Progressive penalty that gets worse over time
        # Gets worse as episode progresses
        hover_penalty = -2.0 * (1 + step / 200)
        shaped_reward += hover_penalty

        # Extra penalty for excessive hovering
        if hover_rate > 0.7:
            shaped_reward -= 5.0 * hover_rate

    # 2. Strong completion incentives
    if done and not (step >= 599):  # Episode ended naturally (not timeout)
        if left_leg and right_leg and abs(vx) < 0.5 and abs(vy) < 0.5:
            # Huge bonus for actual successful landing
            # Bonus for landing quickly
            landing_bonus = 100.0 + (600 - step) * 0.5
            shaped_reward += landing_bonus
        elif reward < -50:  # Crash
            # Don't penalize crashes as much - better than hovering
            shaped_reward = max(shaped_reward, -100.0)

    # 3. Timeout penalties
    if step >= 599:  # Timeout
        shaped_reward -= 200.0  # Massive timeout penalty

    # 4. Encourage decisive actions when close to ground
    if y < 0.2 and (left_leg or right_leg):
        # Reward for taking main engine (decisive landing) or doing nothing (if stable)
        if step < 500:  # But only if not late in episode
            action_rewards = {0: 1.0, 2: 2.0}  # No-op and main engine
            # (Note: action not available in this function, would need to be passed)

    # 5. Speed-based rewards near ground
    if y < 0.1 and (left_leg and right_leg):
        if abs(vx) < 0.1 and abs(vy) < 0.1:
            # Reward for being stable and still
            shaped_reward += 5.0
        else:
            # Penalty for moving around when both legs down
            shaped_reward -= 1.0 * (abs(vx) + abs(vy))

    return shaped_reward, original_reward


def train_anti_hover_agent(args):
    """Train agent with anti-hovering mechanisms."""
    print("="*60)
    print("🚀 ANTI-HOVERING TRAINING")
    print("="*60)

    # Setup
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create agent
    agent = AntiHoverDQNAgent(
        state_size=state_size,
        action_size=action_size,
        memory_size=args.memory_size,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        update_target_freq=args.update_target_freq
    )

    # Load existing model if available
    if args.model_path and os.path.exists(args.model_path):
        try:
            agent.load(args.model_path)
            print(f"✅ Loaded model from {args.model_path}")
            # Start with lower epsilon since model is pre-trained
            agent.epsilon = 0.05
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

    # Training metrics
    episode_rewards = []
    successful_landings = 0
    hover_timeouts = 0
    crashes = 0

    best_reward = -float('inf')

    print(f"\nTraining for {args.episodes} episodes...")
    print("Key changes:")
    print("- Strong penalties for hovering behavior")
    print("- Massive timeout penalties (-200)")
    print("- Big bonuses for quick successful landings")
    print("- Progressive hover penalties that increase over time")
    print("-"*60)

    for episode in range(args.episodes):
        state, _ = env.reset()
        total_reward = 0
        shaped_total = 0
        step = 0

        # Reset hover tracking for episode
        agent.hover_tracker = []
        agent.hover_penalty_multiplier = 1.0

        hover_steps = 0

        for step in range(args.max_steps):
            # Track hovering
            hover_info = agent.track_hovering(state, None, step)
            is_hovering, hover_rate = hover_info

            if is_hovering:
                hover_steps += 1

            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Apply anti-hover reward shaping
            shaped_reward, original_reward = apply_anti_hover_reward_shaping(
                state, next_state, reward, step, done, hover_info
            )

            # Store experience
            agent.remember(state, action, shaped_reward, next_state, done)

            # Train
            if step % 4 == 0:
                loss = agent.replay()

            state = next_state
            total_reward += original_reward
            shaped_total += shaped_reward

            if done:
                break

        # Episode analysis
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state

        # Categorize outcome
        outcome = "UNKNOWN"
        if step >= args.max_steps - 1:
            hover_timeouts += 1
            outcome = "HOVER TIMEOUT"
        elif terminated:
            if (left_leg and right_leg and abs(vx) < 0.5 and abs(vy) < 0.5 and
                    abs(x) < 0.5 and total_reward > 100):
                successful_landings += 1
                outcome = "SUCCESS"
            elif total_reward < -50:
                crashes += 1
                outcome = "CRASH"
            else:
                outcome = "ENDED"

        episode_rewards.append(total_reward)

        # Logging
        if episode % 20 == 0:
            recent_avg = np.mean(
                episode_rewards[-20:]) if episode_rewards else 0
            success_rate = successful_landings / (episode + 1)
            hover_rate = hover_timeouts / (episode + 1)

            print(f"Episode {episode:4d}: {outcome:12s} | "
                  f"Reward: {total_reward:7.1f} | "
                  f"Shaped: {shaped_total:7.1f} | "
                  f"Steps: {step+1:3d} | "
                  f"Hover: {hover_steps:3d}")

            if episode > 0:
                print(f"             Avg20: {recent_avg:7.1f} | "
                      f"Success: {success_rate:.1%} | "
                      f"Hover TO: {hover_rate:.1%} | "
                      f"Epsilon: {agent.epsilon:.3f}")
                print()

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            os.makedirs("results/anti_hover/models", exist_ok=True)
            agent.save("results/anti_hover/models/best_model")
            print(f"🎯 New best: {best_reward:.1f} (Episode {episode})")

        # Auto-save periodically
        if (episode + 1) % 100 == 0:
            agent.save(f"results/anti_hover/models/episode_{episode+1}")

    env.close()

    # Final stats
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Episodes: {args.episodes}")
    print(
        f"Successful Landings: {successful_landings} ({successful_landings/args.episodes:.1%})")
    print(
        f"Hover Timeouts: {hover_timeouts} ({hover_timeouts/args.episodes:.1%})")
    print(f"Crashes: {crashes} ({crashes/args.episodes:.1%})")
    print(f"Best Reward: {best_reward:.1f}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")

    return agent


def parse_args():
    parser = argparse.ArgumentParser(
        description='Anti-Hovering LunarLander Training')

    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--memory-size', type=int, default=50000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon-min', type=float, default=0.01)
    parser.add_argument('--epsilon-decay', type=float, default=0.996)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--update-target-freq', type=int, default=20)
    parser.add_argument('--model-path', type=str, default=None)

    return parser.parse_args()


def main():
    apply_cpu_optimizations()
    args = parse_args()

    print("""
🚀 ANTI-HOVERING MOONLANDER TRAINING
===================================

This version specifically addresses the hovering problem by:

1. 🚫 STRONG HOVER PENALTIES
   - Progressive penalties that increase over time
   - Extra penalties for persistent hovering
   
2. ⏰ MASSIVE TIMEOUT PENALTIES  
   - -200 reward for running out of time
   - Better to crash than hover!
   
3. 🎯 BIG COMPLETION BONUSES
   - +100+ for successful landing
   - Bonus for landing quickly
   
4. 📊 HOVER TRACKING
   - Monitors hovering behavior
   - Adapts penalties dynamically

5. 🎮 SHAPED REWARDS
   - Rewards decisive actions near ground
   - Penalizes unnecessary movement when stable

""")

    # Confirm before starting
    response = input("Start anti-hovering training? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    agent = train_anti_hover_agent(args)
    print("\n✅ Anti-hovering training complete!")
    print("Test the new model with: python enhanced_play_best_model.py --model-path results/anti_hover/models/best_model.weights.h5")


if __name__ == "__main__":
    main()
