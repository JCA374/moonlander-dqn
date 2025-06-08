"""
Diagnostic script to identify why the loaded model is performing poorly.
This will help us understand the actual problem with model loading.
"""
import os
import numpy as np
import tensorflow as tf
import gymnasium as gym
from train_unified import DQNAgent


def diagnose_model_loading():
    """Comprehensive diagnosis of model loading issues."""
    print("🔍 MOONLANDER MODEL LOADING DIAGNOSIS")
    print("="*60)

    # 1. Check if model file exists and is valid
    model_path = "results/current_run/models/best_model.weights.h5"

    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return

    file_size = os.path.getsize(model_path)
    print(f"✅ Model file exists: {file_size:,} bytes")

    # 2. Create environment and agent
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(f"🌙 Environment: {state_size} states, {action_size} actions")

    # 3. Test with fresh agent (untrained)
    print("\n" + "="*60)
    print("🧪 TESTING FRESH (UNTRAINED) AGENT")
    print("="*60)

    fresh_agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        memory_size=10000,
        epsilon=0.0,  # No exploration
        epsilon_min=0.0,
        epsilon_decay=0.995,
        learning_rate=0.0005,
        batch_size=64,
        update_target_freq=20
    )

    # Test fresh agent for a few steps
    state, _ = env.reset(seed=42)
    fresh_rewards = []

    for step in range(100):
        action = fresh_agent.act(state, evaluate=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        fresh_rewards.append(reward)
        state = next_state
        if terminated or truncated:
            break

    fresh_total = sum(fresh_rewards)
    print(f"🔮 Fresh agent (100 steps): {fresh_total:.1f} total reward")
    print(
        f"🎲 Actions taken: {[fresh_agent.act(state, evaluate=True) for _ in range(10)]}")

    # 4. Test with loaded agent
    print("\n" + "="*60)
    print("🔄 TESTING LOADED AGENT")
    print("="*60)

    loaded_agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        memory_size=10000,
        epsilon=0.0,  # No exploration
        epsilon_min=0.0,
        epsilon_decay=0.995,
        learning_rate=0.0005,
        batch_size=64,
        update_target_freq=20
    )

    try:
        loaded_agent.load(model_path)
        print(f"✅ Successfully loaded weights from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return

    # Test loaded agent on same scenario
    state, _ = env.reset(seed=42)  # Same seed as fresh agent
    loaded_rewards = []
    loaded_actions = []

    for step in range(100):
        action = loaded_agent.act(state, evaluate=True)
        loaded_actions.append(action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        loaded_rewards.append(reward)
        state = next_state
        if terminated or truncated:
            break

    loaded_total = sum(loaded_rewards)
    print(f"🎯 Loaded agent (100 steps): {loaded_total:.1f} total reward")
    print(f"🎮 Actions taken: {loaded_actions[:10]}")

    # 5. Compare Q-values
    print("\n" + "="*60)
    print("🧠 Q-VALUES COMPARISON")
    print("="*60)

    # Test on a few specific states
    test_states = [
        np.array([0.0, 1.5, 0.0, -0.5, 0.0, 0.0, 0, 0]),  # High altitude
        np.array([0.0, 0.5, 0.0, -0.3, 0.0, 0.0, 0, 0]),  # Medium altitude
        np.array([0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 0, 0]),  # Near ground
    ]

    action_names = ["Nothing", "Left", "Main", "Right"]

    for i, test_state in enumerate(test_states):
        print(
            f"\n🔬 Test State {i+1}: altitude={test_state[1]:.1f}, v_y={test_state[3]:.1f}")

        # Fresh agent Q-values
        fresh_q = fresh_agent.model.predict(
            test_state.reshape(1, -1), verbose=0)[0]
        fresh_action = np.argmax(fresh_q)

        # Loaded agent Q-values
        loaded_q = loaded_agent.model.predict(
            test_state.reshape(1, -1), verbose=0)[0]
        loaded_action = np.argmax(loaded_q)

        print("   Fresh Agent Q-values:")
        for j, (name, q_val) in enumerate(zip(action_names, fresh_q)):
            marker = " ⭐" if j == fresh_action else ""
            print(f"     {name}: {q_val:8.3f}{marker}")

        print("   Loaded Agent Q-values:")
        for j, (name, q_val) in enumerate(zip(action_names, loaded_q)):
            marker = " ⭐" if j == loaded_action else ""
            print(f"     {name}: {q_val:8.3f}{marker}")

        # Check if they're significantly different
        q_diff = np.abs(fresh_q - loaded_q).max()
        if q_diff < 0.1:
            print(
                f"   ⚠️ WARNING: Q-values too similar (max diff: {q_diff:.4f})")
        else:
            print(f"   ✅ Q-values different (max diff: {q_diff:.4f})")

    # 6. Check model architecture
    print("\n" + "="*60)
    print("🏗️ MODEL ARCHITECTURE CHECK")
    print("="*60)

    fresh_params = fresh_agent.model.count_params()
    loaded_params = loaded_agent.model.count_params()

    print(f"Fresh agent parameters: {fresh_params:,}")
    print(f"Loaded agent parameters: {loaded_params:,}")

    if fresh_params != loaded_params:
        print("❌ Parameter count mismatch!")
    else:
        print("✅ Parameter counts match")

    # 7. Check if weights actually loaded
    print("\n" + "="*60)
    print("⚖️ WEIGHT COMPARISON")
    print("="*60)

    fresh_weights = fresh_agent.model.get_weights()
    loaded_weights = loaded_agent.model.get_weights()

    total_weight_diff = 0
    for fw, lw in zip(fresh_weights, loaded_weights):
        diff = np.abs(fw - lw).mean()
        total_weight_diff += diff

    print(f"Average weight difference: {total_weight_diff:.6f}")

    if total_weight_diff < 0.001:
        print("❌ PROBLEM: Weights are too similar to fresh model!")
        print("   This suggests the weights didn't load properly.")
    else:
        print("✅ Weights are significantly different from fresh model")

    # 8. Final diagnosis
    print("\n" + "="*60)
    print("🏥 DIAGNOSIS SUMMARY")
    print("="*60)

    if total_weight_diff < 0.001:
        print("❌ CRITICAL: Model weights appear to not have loaded!")
        print("   The loaded model is behaving like an untrained model.")
        print("   Possible causes:")
        print("   • Model architecture mismatch")
        print("   • Corrupted weight file")
        print("   • Wrong loading method")

    elif abs(fresh_total - loaded_total) < 10:
        print("⚠️ WARNING: Performance too similar to random agent")
        print("   Possible causes:")
        print("   • Model was undertrained")
        print("   • Model overfitted and performs poorly")
        print("   • Wrong evaluation mode")

    else:
        print("✅ Model appears to have loaded correctly")
        print(f"   Performance difference: {loaded_total - fresh_total:.1f}")

        if loaded_total < fresh_total - 50:
            print("   But performance is surprisingly poor!")
            print("   Check if model was saved during training or at end")

    env.close()


def check_training_args():
    """Check the training arguments used."""
    args_path = "results/current_run/args.json"

    if os.path.exists(args_path):
        import json
        with open(args_path, 'r') as f:
            args = json.load(f)

        print("\n" + "="*60)
        print("📋 TRAINING CONFIGURATION")
        print("="*60)

        important_params = [
            'episodes', 'epsilon_min', 'epsilon_decay',
            'learning_rate', 'batch_size', 'gamma'
        ]

        for param in important_params:
            if param in args:
                print(f"{param}: {args[param]}")

        # Check for potential issues
        if args.get('epsilon_min', 0.1) <= 0.01:
            print("⚠️ WARNING: Very low epsilon_min - agent might not explore enough")

        if args.get('episodes', 0) < 200:
            print("⚠️ WARNING: Low episode count - model might be undertrained")


if __name__ == "__main__":
    diagnose_model_loading()
    check_training_args()
