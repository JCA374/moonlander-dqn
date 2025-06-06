"""
Quick system test to verify optimizations work on Intel i7-8565U
Run this first to ensure everything is working before full training.
"""
import time
import numpy as np
import tensorflow as tf
import gymnasium as gym
import os


def test_cpu_optimizations():
    """Test CPU optimizations."""
    print("🔧 Testing CPU optimizations for Intel i7-8565U...")

    # Apply optimizations
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    tf.config.optimizer.set_jit(True)

    print(
        f"✓ Intra-op threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
    print(
        f"✓ Inter-op threads: {tf.config.threading.get_inter_op_parallelism_threads()}")
    print(f"✓ oneDNN optimizations enabled")
    print(f"✓ XLA JIT compilation enabled")


def test_memory_allocation():
    """Test memory allocation with larger buffers."""
    print("\n💾 Testing memory allocation (32GB RAM)...")

    try:
        # Test large memory allocation
        buffer_size = 200000
        state_size = 8
        memory_mb = (buffer_size * state_size * 4 * 5) / \
            1024 / 1024  # 5 arrays, 4 bytes per float

        states = np.zeros((buffer_size, state_size), dtype=np.float32)
        actions = np.zeros(buffer_size, dtype=np.int32)
        rewards = np.zeros(buffer_size, dtype=np.float32)
        next_states = np.zeros((buffer_size, state_size), dtype=np.float32)
        dones = np.zeros(buffer_size, dtype=bool)

        print(f"✓ Allocated {buffer_size:,} experience buffer")
        print(f"✓ Memory usage: ~{memory_mb:.1f} MB")

        # Test batch sampling
        batch_size = 128
        indices = np.random.choice(buffer_size, batch_size, replace=False)
        batch_states = states[indices]

        print(f"✓ Batch sampling working (size: {batch_size})")

    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        return False

    return True


def test_network_performance():
    """Test neural network performance."""
    print("\n🧠 Testing neural network performance...")

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, BatchNormalization
        from tensorflow.keras.optimizers import Adam

        # Create test network
        model = Sequential([
            tf.keras.Input(shape=(8,)),
            Dense(128, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dense(128, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dense(4, activation='linear')
        ])

        optimizer = Adam(learning_rate=0.001, clipnorm=1.0, amsgrad=True)
        model.compile(loss='huber', optimizer=optimizer)

        print(f"✓ Network created: {model.count_params():,} parameters")

        # Test inference speed
        test_input = np.random.random((128, 8)).astype(np.float32)

        # Warmup
        for _ in range(10):
            _ = model.predict(test_input, verbose=0)

        # Timing test
        start_time = time.time()
        for _ in range(100):
            _ = model.predict(test_input, verbose=0)
        inference_time = time.time() - start_time

        predictions_per_second = (100 * 128) / inference_time
        print(
            f"✓ Inference speed: {predictions_per_second:.0f} predictions/second")

        # Test training speed
        test_targets = np.random.random((128, 4)).astype(np.float32)

        start_time = time.time()
        for _ in range(10):
            model.fit(test_input, test_targets, epochs=1, verbose=0)
        training_time = time.time() - start_time

        training_steps_per_second = 10 / training_time
        print(
            f"✓ Training speed: {training_steps_per_second:.1f} steps/second")

        return training_steps_per_second > 1.0  # Should be faster than 1 step/second

    except Exception as e:
        print(f"✗ Network test failed: {e}")
        return False


def test_environment():
    """Test environment performance."""
    print("\n🌙 Testing LunarLander environment...")

    try:
        env = gym.make('LunarLander-v3')

        # Test reset and step
        state, _ = env.reset(seed=42)
        print(f"✓ Environment created, state shape: {state.shape}")

        # Test step speed
        start_time = time.time()
        for i in range(1000):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state

        env_time = time.time() - start_time
        steps_per_second = 1000 / env_time

        print(f"✓ Environment speed: {steps_per_second:.0f} steps/second")

        env.close()
        return steps_per_second > 500  # Should be very fast

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def estimate_training_time():
    """Estimate total training time."""
    print("\n⏱️  Estimating training time...")

    # Assumptions based on optimized settings
    episodes = 1000
    avg_steps_per_episode = 400
    total_steps = episodes * avg_steps_per_episode

    # Conservative estimates for your system
    steps_per_second = 20  # Conservative estimate
    training_overhead = 1.5  # 50% overhead for training

    estimated_seconds = (total_steps / steps_per_second) * training_overhead
    estimated_hours = estimated_seconds / 3600

    print(f"📊 Training estimates:")
    print(f"  Total episodes: {episodes:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Expected speed: ~{steps_per_second} steps/second")
    print(f"  Estimated time: {estimated_hours:.1f} hours")
    print(
        f"  Expected completion: ~{estimated_hours:.0f}h {(estimated_hours % 1) * 60:.0f}m")


def run_complete_test():
    """Run all tests."""
    print("=" * 60)
    print("🚀 INTEL i7-8565U OPTIMIZATION TEST")
    print("=" * 60)

    # Run all tests
    test_cpu_optimizations()
    memory_ok = test_memory_allocation()
    network_ok = test_network_performance()
    env_ok = test_environment()
    estimate_training_time()

    print("\n" + "=" * 60)
    print("📋 TEST RESULTS")
    print("=" * 60)

    results = {
        "Memory allocation": "✓ PASS" if memory_ok else "✗ FAIL",
        "Network performance": "✓ PASS" if network_ok else "✗ FAIL",
        "Environment speed": "✓ PASS" if env_ok else "✗ FAIL"
    }

    for test, result in results.items():
        print(f"  {test}: {result}")

    all_passed = memory_ok and network_ok and env_ok

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your system is ready for optimized training.")
        print("\nNext steps:")
        print("1. Save the optimized script as 'train_optimized.py'")
        print("2. Run: python train_optimized.py --episodes 1000")
        print("3. Expected training time: ~8-12 hours")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Check the error messages above and resolve issues before training.")

    return all_passed


if __name__ == "__main__":
    run_complete_test()
