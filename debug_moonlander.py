"""
Debug script to diagnose MoonLander training issues
"""
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from train import DQNAgent
import os


def test_reward_shaping():
    """Test if reward shaping is working correctly."""
    print("\n=== Testing Reward Shaping ===")

    # Simulate different states
    test_cases = [
        # (x, y, vx, vy, angle, angular_vel, left_leg, right_leg, description)
        (0.0, 0.15, 0.0, -0.05, 0.0, 0.0, 0, 0, "Hovering above pad"),
        (0.0, 0.01, 0.0, -0.1, 0.0, 0.0, 1, 1, "Landing on pad"),
        (0.5, 1.5, 0.2, -0.5, 0.3, 0.1, 0, 0, "High altitude"),
        (0.0, 0.5, 0.0, -2.0, 0.0, 0.0, 0, 0, "Fast descent"),
    ]

    for i, (x, y, vx, vy, angle, angular_vel, left_leg, right_leg, desc) in enumerate(test_cases):
        state = np.array(
            [x, y, vx, vy, angle, angular_vel, left_leg, right_leg])

        # Simulate reward calculation (you'll need to extract this logic)
        base_reward = 0
        scaled_reward = base_reward

        # Your reward shaping logic here
        hover_detected = False
        if abs(x) < 0.4 and 0.02 < y < 0.3 and abs(vy) < 0.15 and abs(vx) < 0.15:
            hover_detected = True
            hover_penalty = -1.0 - (200 / 100)  # Assume step 200
            scaled_reward += hover_penalty
            print(f"  Hover penalty applied: {hover_penalty}")

        if y < 0.5 and not hover_detected:
            proximity_bonus = (0.5 - y) * 2.0
            scaled_reward += proximity_bonus
            print(f"  Proximity bonus: {proximity_bonus}")

        print(f"\nTest {i+1}: {desc}")
        print(f"  State: x={x:.2f}, y={y:.2f}, vy={vy:.2f}")
        print(f"  Final reward: {scaled_reward:.2f}")
        print(f"  Hover detected: {hover_detected}")


def test_agent_behavior():
    """Test agent's actual behavior in environment."""
    print("\n=== Testing Agent Behavior ===")

    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        memory_size=10000,
        gamma=0.99,
        epsilon=0.0,  # No exploration for testing
        epsilon_min=0.0,
        epsilon_decay=0.995,
        learning_rate=0.0005,
        batch_size=64,
        update_target_freq=50
    )

    # Load the best model
    model_path = "results/current_run/models/best_model.weights.h5"
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("No model found - using random agent")
        return

    # Test 5 episodes
    hover_count = 0
    landing_attempts = 0

    for episode in range(5):
        state, _ = env.reset()
        done = False
        steps = 0
        hover_steps = 0
        min_altitude = float('inf')

        while not done and steps < 500:
            action = agent.act(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state

            # Track behavior
            min_altitude = min(min_altitude, y)

            # Check hovering
            if abs(x) < 0.4 and 0.02 < y < 0.3 and abs(vy) < 0.15:
                hover_steps += 1

            # Check landing attempt
            if y < 0.1 and abs(vy) > 0.5:
                landing_attempts += 1

            state = next_state
            steps += 1

        if hover_steps > 50:
            hover_count += 1

        print(f"\nEpisode {episode + 1}:")
        print(f"  Min altitude: {min_altitude:.3f}")
        print(f"  Hover steps: {hover_steps}")
        print(f"  Final reward: {reward:.2f}")
        print(f"  Success: {terminated and reward > 0}")

    print(f"\nSummary: {hover_count}/5 episodes had hovering behavior")
    print(f"Landing attempts detected: {landing_attempts}")

    env.close()


def test_q_values():
    """Analyze Q-values to understand agent's decision making."""
    print("\n=== Testing Q-Values ===")

    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create and load agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        memory_size=10000,
        gamma=0.99,
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=0.995,
        learning_rate=0.0005,
        batch_size=64,
        update_target_freq=50
    )

    model_path = "results/current_run/models/best_model.weights.h5"
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print("No model found")
        return

    # Test specific states
    test_states = [
        np.array([0.0, 0.15, 0.0, -0.05, 0.0, 0.0, 0, 0]),  # Hovering
        # Good landing approach
        np.array([0.0, 0.05, 0.0, -0.5, 0.0, 0.0, 0, 0]),
        np.array([0.0, 1.5, 0.0, -1.0, 0.0, 0.0, 0, 0]),    # High altitude
    ]

    action_names = ["Nothing", "Left", "Main", "Right"]

    for i, state in enumerate(test_states):
        q_values = agent.model.predict(state.reshape(1, -1), verbose=0)[0]
        best_action = np.argmax(q_values)

        print(f"\nState {i+1}: y={state[1]:.2f}, vy={state[3]:.2f}")
        print("Q-values:")
        for j, (action, q_val) in enumerate(zip(action_names, q_values)):
            marker = " <-- BEST" if j == best_action else ""
            print(f"  {action}: {q_val:.3f}{marker}")

    env.close()


def analyze_training_history():
    """Analyze training metrics from logs."""
    print("\n=== Analyzing Training History ===")

    # This would parse your training logs
    # For now, we'll show what to look for
    print("Key metrics to check:")
    print("1. When did hovering behavior start?")
    print("2. What was the best reward achieved?")
    print("3. How often are episodes terminated for hovering?")
    print("4. Is the agent exploring enough (epsilon)?")


def create_unit_tests():
    """Create unit tests for critical components."""
    print("\n=== Unit Test Suite ===")

    tests = []

    # Test 1: Reward scaling
    def test_reward_scaling():
        rewards = [150, -150, 50, -50, 0]
        expected = [225, -150, 25, -25, 0]  # Based on your scaling logic

        for r, exp in zip(rewards, expected):
            if r > 100:
                scaled = r * 1.5
            elif r < -100:
                scaled = r
            else:
                scaled = r * 0.5

            assert abs(
                scaled - exp) < 0.01, f"Reward scaling failed: {r} -> {scaled}, expected {exp}"

        return "PASS"

    # Test 2: Hovering detection
    def test_hovering_detection():
        # (x, y, vx, vy) -> should_be_hovering
        cases = [
            (0.0, 0.15, 0.0, 0.05, True),
            (0.0, 0.01, 0.0, 0.05, False),  # Too low
            (0.5, 0.15, 0.0, 0.05, False),  # Too far from center
            (0.0, 0.15, 0.0, 0.3, False),   # Too fast
        ]

        for x, y, vx, vy, expected in cases:
            is_hovering = (abs(x) < 0.4 and 0.02 < y < 0.3 and
                           abs(vy) < 0.15 and abs(vx) < 0.15)
            assert is_hovering == expected, f"Hovering detection failed for ({x}, {y}, {vx}, {vy})"

        return "PASS"

    # Run tests
    tests = [
        ("Reward Scaling", test_reward_scaling),
        ("Hovering Detection", test_hovering_detection),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            print(f"✓ {name}: {result}")
        except AssertionError as e:
            print(f"✗ {name}: FAIL - {e}")
        except Exception as e:
            print(f"✗ {name}: ERROR - {e}")


if __name__ == "__main__":
    print("MoonLander Debug Suite")
    print("=" * 50)

    # Run all tests
    test_reward_shaping()
    test_agent_behavior()
    test_q_values()
    analyze_training_history()
    create_unit_tests()

    print("\n" + "=" * 50)
    print("Debug complete. Check results above.")
