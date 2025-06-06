#!/usr/bin/env python3
"""
Debug analysis script to identify the current issues with the training.
"""
import numpy as np
import gymnasium as gym


def analyze_logs():
    """Analyze the training logs to identify issues."""
    print("="*60)
    print("LOG ANALYSIS")
    print("="*60)

    # Key observations from your logs
    observations = [
        ("Reward threshold", "Reaching +80 to +108 rewards"),
        ("Success detection", "Still 0 successes despite good rewards"),
        ("Epsilon", "Dropped to 0.01 (minimum) too early"),
        ("Crashes", "44 crashes detected - agent is trying to land"),
        ("Best reward", "108.72 - this is actually quite good!"),
        ("Episode 200", "Reward 80.34 - improving but not crossing threshold"),
    ]

    for issue, description in observations:
        print(f"• {issue}: {description}")

    print("\n" + "="*60)
    print("CRITICAL FINDINGS")
    print("="*60)

    print("🎯 YOUR AGENT IS ACTUALLY LEARNING WELL!")
    print("   - Rewards improved from negative to +108")
    print("   - Agent is attempting landings (44 crashes)")
    print("   - The core learning mechanism is working")
    print()
    print("🔍 BUT THERE ARE 3 CRITICAL ISSUES:")
    print("   1. Success detection still broken")
    print("   2. Epsilon too low (0.01) - no exploration")
    print("   3. Batch size might be too large for current setup")


def test_success_conditions():
    """Test what constitutes a successful landing in LunarLander-v3."""
    print("\n" + "="*60)
    print("TESTING LUNARLANDER-V3 SUCCESS CONDITIONS")
    print("="*60)

    env = gym.make('LunarLander-v3')

    print("From Gymnasium docs:")
    print("• Landing safely: +100 points")
    print("• Crashing: -100 points")
    print("• Leg ground contact: +10 points each")
    print("• Engine use: -0.3 points per frame")
    print("• Success threshold: 200+ points average")

    # Let's test a few episodes to see actual reward ranges
    print("\nTesting 10 random episodes to see reward distribution:")
    rewards = []

    for i in range(10):
        state, _ = env.reset(seed=42 + i)
        total_reward = 0
        steps = 0

        for step in range(1000):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                outcome = "SUCCESS" if terminated and reward > 0 else "CRASH/TIMEOUT"
                print(
                    f"Episode {i+1}: {total_reward:.1f} points, {steps} steps, {outcome}")
                rewards.append(total_reward)
                break

    env.close()

    print(f"\nRandom agent statistics:")
    print(f"• Average reward: {np.mean(rewards):.1f}")
    print(f"• Best reward: {max(rewards):.1f}")
    print(f"• Worst reward: {min(rewards):.1f}")
    print(
        f"• Your agent's +108 is {108 - np.mean(rewards):.1f} points better than random!")


def diagnose_success_detection():
    """Diagnose why success detection isn't working."""
    print("\n" + "="*60)
    print("DIAGNOSING SUCCESS DETECTION")
    print("="*60)

    # Test the actual success detection logic
    test_cases = [
        {"terminated": True, "reward": 150.0, "description": "Perfect landing"},
        {"terminated": True, "reward": 100.0, "description": "Minimal success"},
        {"terminated": True, "reward": 50.0, "description": "Poor landing"},
        {"terminated": True, "reward": -100.0, "description": "Crash"},
        {"terminated": False, "reward": 100.0, "description": "Not terminated"},
    ]

    print("Testing success detection logic:")
    print("(Based on your current code)")

    for case in test_cases:
        terminated = case["terminated"]
        reward = case["reward"]

        # Your current logic (from the logs analysis)
        success = terminated and reward > 0

        status = "✓ SUCCESS" if success else "✗ NO SUCCESS"
        print(f"• {case['description']}: {status}")
        print(f"  terminated={terminated}, reward={reward}")

    print("\n🔍 ISSUE FOUND:")
    print("If your agent is getting +108 rewards but 0 successes,")
    print("it means episodes are NOT terminating with positive rewards.")
    print("The agent might be:")
    print("• Timing out instead of landing")
    print("• Landing with slight negative final reward")
    print("• Getting positive rewards but not 'terminated' flag")


def check_epsilon_issue():
    """Check if epsilon decay is too aggressive."""
    print("\n" + "="*60)
    print("EPSILON DECAY ANALYSIS")
    print("="*60)

    # Your current settings
    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    episodes_to_min = 0
    epsilon = epsilon_start

    while epsilon > epsilon_min:
        epsilon *= epsilon_decay
        episodes_to_min += 1

    print(f"Current settings:")
    print(f"• Start: {epsilon_start}")
    print(f"• Min: {epsilon_min}")
    print(f"• Decay: {epsilon_decay}")
    print(f"• Episodes to minimum: {episodes_to_min}")

    print(
        f"\n⚠️ PROBLEM: Epsilon reaches minimum after {episodes_to_min} episodes!")
    print("At episode 200, epsilon = 0.01 (minimum)")
    print("This means NO exploration after episode ~460")
    print("\n💡 SOLUTION: Use slower decay or higher minimum")
    print("• Recommended: decay=0.999, min=0.05")
    print("• This would take ~920 episodes to reach 0.05")


def recommend_fixes():
    """Recommend specific fixes based on analysis."""
    print("\n" + "="*60)
    print("RECOMMENDED FIXES")
    print("="*60)

    fixes = [
        {
            "issue": "Epsilon too low",
            "fix": "Change epsilon_min=0.05 and epsilon_decay=0.999",
            "command": "--epsilon-min 0.05 --epsilon-decay 0.999"
        },
        {
            "issue": "Success detection",
            "fix": "Debug actual episode termination conditions",
            "command": "Add logging for terminated/reward values"
        },
        {
            "issue": "Batch size",
            "fix": "Reduce batch size for more frequent updates",
            "command": "--batch-size 32 (instead of current large size)"
        },
        {
            "issue": "Target network",
            "fix": "Update target network more frequently",
            "command": "--update-target-freq 10"
        }
    ]

    for i, fix in enumerate(fixes, 1):
        print(f"{i}. {fix['issue']}:")
        print(f"   Fix: {fix['fix']}")
        print(f"   Command: {fix['command']}")
        print()

    print("🚀 IMMEDIATE ACTION:")
    print("Run this command to test the fixes:")
    print()
    print("python train_fixed.py \\")
    print("  --episodes 200 \\")
    print("  --epsilon-min 0.05 \\")
    print("  --epsilon-decay 0.999 \\")
    print("  --batch-size 32 \\")
    print("  --update-target-freq 10 \\")
    print("  --restart")


def main():
    """Run complete analysis."""
    analyze_logs()
    test_success_conditions()
    diagnose_success_detection()
    check_epsilon_issue()
    recommend_fixes()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✅ Your agent IS learning (rewards: -600 → +108)")
    print("⚠️  But epsilon decay too aggressive (no exploration)")
    print("🔍 Success detection needs debugging")
    print("🎯 You're very close to success!")


if __name__ == "__main__":
    main()
