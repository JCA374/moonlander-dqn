"""
Enhanced visualization script for the trained MoonLander agent.
Based on your successful training with 49.12% success rate!
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
import gymnasium as gym
from datetime import datetime

# Import your successful agent architecture


class DQNAgent:
    """Enhanced DQN Agent matching your successful training architecture."""

    def __init__(self, state_size, action_size, epsilon=0.0):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.model = self._build_model()

    def _build_model(self):
        """Build the exact same architecture that achieved 49.12% success rate."""
        model = Sequential()

        # Match your successful training architecture exactly
        model.add(tf.keras.Input(shape=(self.state_size,)))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear'))

        # Use the same optimizer settings
        optimizer = Adam(learning_rate=0.0005, epsilon=1e-5)
        model.compile(loss='huber', optimizer=optimizer)

        return model

    def act(self, state, evaluate=True):
        """Choose the best action using the trained policy."""
        if not evaluate and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def load(self, filepath):
        """Load the trained model weights."""
        if not filepath.endswith('.weights.h5') and not os.path.exists(filepath):
            if os.path.exists(filepath + '.weights.h5'):
                filepath = filepath + '.weights.h5'
            else:
                filepath = filepath + '.h5'
        self.model.load_weights(filepath)
        print(f"✅ Successfully loaded model: {filepath}")


def find_best_model(results_dir="results"):
    """Find the best model from your training runs."""
    print(f"🔍 Searching for trained models in {results_dir}")

    # Priority search order based on your training structure
    search_paths = [
        os.path.join(results_dir, 'current_run',
                     'models', 'best_model.weights.h5'),
        os.path.join(results_dir, 'current_run',
                     'models', 'final_model.weights.h5'),
        os.path.join(results_dir, 'current_run', 'episode_saves'),
        os.path.join(results_dir, 'current_run', 'models'),
        results_dir
    ]

    # Check for best model first
    for path in search_paths[:2]:
        if os.path.exists(path):
            print(f"🎯 Found best model: {path}")
            return path

    # Check episode saves for recent models
    episode_saves_dir = search_paths[2]
    if os.path.exists(episode_saves_dir):
        episode_models = []
        for file in os.listdir(episode_saves_dir):
            if file.endswith('.weights.h5'):
                episode_num = int(file.split('_')[-1].split('.')[0])
                model_path = os.path.join(episode_saves_dir, file)
                episode_models.append((episode_num, model_path))

        if episode_models:
            # Get the most recent episode model
            episode_models.sort(reverse=True)
            best_episode_model = episode_models[0][1]
            print(f"📈 Found recent episode model: {best_episode_model}")
            return best_episode_model

    # Fallback search
    for search_dir in search_paths[3:]:
        if not os.path.exists(search_dir):
            continue

        model_files = []
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.weights.h5') or file.endswith('.h5'):
                    model_files.append(os.path.join(root, file))

        if model_files:
            # Sort by modification time (most recent first)
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            print(f"📁 Found fallback model: {model_files[0]}")
            return model_files[0]

    print("❌ No trained models found!")
    return None


def analyze_performance(state, action, reward, next_state, step):
    """Analyze the agent's performance in real-time."""
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state

    # Action descriptions
    action_names = ["No engines", "Left engine", "Main engine", "Right engine"]

    # Performance metrics
    altitude = y
    horizontal_distance = abs(x)
    speed = np.sqrt(vx**2 + vy**2)
    tilt = abs(angle)

    # Landing assessment
    landing_zone = horizontal_distance < 0.3
    safe_speed = speed < 1.0
    good_angle = tilt < 0.3
    legs_down = left_leg or right_leg

    return {
        'action': action_names[action],
        'altitude': altitude,
        'distance_from_center': horizontal_distance,
        'speed': speed,
        'tilt': tilt,
        'in_landing_zone': landing_zone,
        'safe_speed': safe_speed,
        'good_angle': good_angle,
        'legs_touching': legs_down,
        'landing_score': sum([landing_zone, safe_speed, good_angle, legs_down])
    }


def predict_outcome(state, total_reward, step):
    """Predict likely episode outcome based on current state."""
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = state

    # Success indicators
    if left_leg and right_leg and abs(vx) < 0.5 and abs(vy) < 0.5 and abs(x) < 0.5:
        return "🎯 SUCCESSFUL LANDING LIKELY"
    elif y < 0.1 and abs(vy) < 1.0 and abs(x) < 0.3:
        return "🟡 APPROACHING LANDING"
    elif y < 0.5 and abs(x) < 0.5:
        return "🔄 IN LANDING ZONE"
    elif abs(angle) > 1.0 or abs(vx) > 2.0:
        return "⚠️  UNSTABLE - CRASH RISK"
    elif step > 500:
        return "⏱️  TIME PRESSURE"
    else:
        return "🚀 NAVIGATING"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced MoonLander Visualization')

    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to specific model (auto-detects if not provided)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to play (default: 5)')
    parser.add_argument('--max-steps', type=int, default=600,
                        help='Maximum steps per episode (default: 600)')
    parser.add_argument('--delay', type=float, default=0.02,
                        help='Delay between steps for visualization (default: 0.02s)')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed analysis during playback')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible results')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark mode (no rendering, performance focus)')

    return parser.parse_args()


def play_enhanced():
    """Enhanced visualization with performance analysis."""
    args = parse_args()

    # Create environment
    render_mode = None if args.benchmark else 'human'
    env = gym.make('LunarLander-v3', render_mode=render_mode)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print("\n" + "="*60)
    print("🚀 ENHANCED MOONLANDER VISUALIZATION")
    print("="*60)
    print(f"🎯 Expected Success Rate: ~49% (based on training)")
    print(f"🧠 State Space: {state_size} dimensions")
    print(f"⚙️  Action Space: {action_size} discrete actions")
    print(f"🎮 Episodes: {args.episodes}")

    # Load trained agent
    agent = DQNAgent(state_size, action_size, epsilon=0.0)

    model_path = args.model_path or find_best_model()
    if not model_path:
        print("\n❌ No trained model found! Please train a model first.")
        return

    try:
        agent.load(model_path)
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return

    # Performance tracking
    episode_results = []
    successful_landings = 0
    total_rewards = []

    print(f"\n🎬 Starting visualization...")
    print("-" * 60)

    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        total_reward = 0
        step_count = 0
        done = False

        print(f"\n🎮 Episode {episode}/{args.episodes}")

        # Episode tracking
        max_altitude = state[1]
        min_distance = abs(state[0])
        actions_taken = []

        while not done and step_count < args.max_steps:
            # Agent decision
            action = agent.act(state, evaluate=True)
            actions_taken.append(action)

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update tracking
            total_reward += reward
            step_count += 1
            max_altitude = max(max_altitude, next_state[1])
            min_distance = min(min_distance, abs(next_state[0]))

            # Real-time analysis
            if args.detailed and step_count % 50 == 0:
                analysis = analyze_performance(
                    state, action, reward, next_state, step_count)
                prediction = predict_outcome(
                    next_state, total_reward, step_count)

                print(f"  Step {step_count:3d}: {analysis['action']:12s} | "
                      f"Alt: {analysis['altitude']:.2f} | "
                      f"Dist: {analysis['distance_from_center']:.2f} | "
                      f"Speed: {analysis['speed']:.2f} | "
                      f"{prediction}")

            # Visualization delay
            if not args.benchmark and args.delay > 0:
                time.sleep(args.delay)

            state = next_state

            # Check for early success prediction
            if args.detailed and not done:
                x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state
                if left_leg and right_leg and abs(vx) < 0.3 and abs(vy) < 0.3:
                    print(f"  🎯 Landing detected at step {step_count}!")

        # Episode analysis
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state

        # Determine outcome using your training's success criteria
        landed_successfully = False
        outcome_emoji = ""
        outcome_desc = ""

        if terminated:
            if left_leg and right_leg and abs(vx) < 0.5 and abs(vy) < 0.5 and abs(x) < 0.5:
                landed_successfully = True
                outcome_emoji = "🎯"
                outcome_desc = "SUCCESSFUL LANDING"
            elif total_reward > 200:
                landed_successfully = True
                outcome_emoji = "🎯"
                outcome_desc = "HIGH SCORE SUCCESS"
            elif total_reward < -50:
                outcome_emoji = "💥"
                outcome_desc = "CRASH"
            else:
                outcome_emoji = "❓"
                outcome_desc = "ENDED"
        elif step_count >= args.max_steps:
            outcome_emoji = "⏰"
            outcome_desc = "TIMEOUT"
        else:
            outcome_emoji = "❓"
            outcome_desc = "UNKNOWN"

        if landed_successfully:
            successful_landings += 1

        # Episode summary
        print(f"\n  {outcome_emoji} {outcome_desc}")
        print(f"  📊 Score: {total_reward:.2f} | Steps: {step_count} | "
              f"Final Position: ({x:.2f}, {y:.2f})")
        print(
            f"  🎯 Max Altitude: {max_altitude:.2f} | Min Distance: {min_distance:.2f}")

        # Action distribution
        if args.detailed:
            action_counts = [actions_taken.count(i) for i in range(4)]
            action_names = ["No-op", "Left", "Main", "Right"]
            print(f"  🎮 Actions: " + " | ".join([f"{name}: {count}"
                                                 for name, count in zip(action_names, action_counts)]))

        # Store results
        episode_results.append({
            'episode': episode,
            'success': landed_successfully,
            'reward': total_reward,
            'steps': step_count,
            'final_position': (x, y),
            'outcome': outcome_desc
        })

        total_rewards.append(total_reward)
        print("-" * 60)

    env.close()

    # Final performance summary
    elapsed_time = time.time() - start_time
    success_rate = successful_landings / args.episodes
    avg_reward = np.mean(total_rewards)

    print(f"\n" + "="*60)
    print("📈 PERFORMANCE SUMMARY")
    print("="*60)
    print(
        f"🎯 Success Rate: {success_rate:.2%} ({successful_landings}/{args.episodes})")
    print(f"📊 Average Score: {avg_reward:.2f}")
    print(f"🏆 Best Score: {max(total_rewards):.2f}")
    print(f"📉 Worst Score: {min(total_rewards):.2f}")
    print(f"⏱️  Total Time: {elapsed_time:.2f}s")
    print(
        f"🎮 Average Episode Length: {np.mean([r['steps'] for r in episode_results]):.1f} steps")

    # Compare to training performance
    print(f"\n🔬 COMPARISON TO TRAINING:")
    print(f"   Expected Success Rate: 49.12%")
    print(f"   Actual Success Rate: {success_rate:.2%}")

    if success_rate >= 0.45:
        print(f"   ✅ Excellent performance! Matches training expectations.")
    elif success_rate >= 0.30:
        print(f"   ✅ Good performance! Close to training results.")
    elif success_rate >= 0.15:
        print(f"   ⚠️  Moderate performance. May need more training.")
    else:
        print(f"   ❌ Low performance. Check model loading or training quality.")

    # Detailed episode breakdown
    if args.detailed:
        print(f"\n📋 EPISODE DETAILS:")
        for result in episode_results:
            status = "✅" if result['success'] else "❌"
            print(f"   Episode {result['episode']:2d}: {status} {result['outcome']:15s} | "
                  f"Score: {result['reward']:6.1f} | Steps: {result['steps']:3d}")

    print("="*60)


if __name__ == "__main__":
    # CPU optimization for best performance
    try:
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(4)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    except:
        pass

    play_enhanced()
