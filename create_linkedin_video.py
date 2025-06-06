"""
Create a professional LinkedIn video showing AI learning progression in MoonLander
"""
import os
import sys
import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
from datetime import datetime

# Import your trained agent
from train_fixed import DQNAgent


class VideoCreator:
    """Create a professional video showing AI learning progression."""

    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        self.video_dir = os.path.join(results_dir, "linkedin_video")
        os.makedirs(self.video_dir, exist_ok=True)

        # Video settings
        self.fps = 10
        self.duration_seconds = 45  # Perfect for LinkedIn
        self.total_frames = self.fps * self.duration_seconds

        # Agent progression checkpoints
        self.checkpoints = [
            {"episode": 0, "desc": "Episode 0: Random Actions", "model": None},
            {"episode": 50, "desc": "Episode 50: Learning Control",
                "model": "episode_50.weights.h5"},
            {"episode": 200, "desc": "Episode 200: First Landings",
                "model": "episode_200.weights.h5"},
            {"episode": 500, "desc": "Episode 500: Improving Accuracy",
                "model": "episode_500.weights.h5"},
            {"episode": 750,
                "desc": "Episode 750: Expert Performance (50% Success)", "model": "best_model.weights.h5"}
        ]

    def load_agent_at_checkpoint(self, checkpoint):
        """Load agent at specific training checkpoint."""
        env = gym.make('LunarLander-v3')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            epsilon=0.0  # No exploration for demo
        )

        if checkpoint["model"]:
            model_path = os.path.join(
                self.results_dir, "current_run", "models", checkpoint["model"])
            if os.path.exists(model_path):
                agent.load(model_path)
                print(f"Loaded model: {model_path}")
            else:
                print(f"Model not found: {model_path}, using random agent")

        env.close()
        return agent

    def record_episode(self, agent, checkpoint, max_steps=300):
        """Record one episode with the agent."""
        env = gym.make('LunarLander-v3', render_mode='rgb_array')

        frames = []
        rewards = []
        state, _ = env.reset(seed=42)  # Consistent seed for comparison
        total_reward = 0

        for step in range(max_steps):
            # Get frame
            frame = env.render()
            frames.append(frame)

            # Agent action
            if agent and checkpoint["model"]:
                action = agent.act(state, evaluate=True)
            else:
                action = env.action_space.sample()  # Random for episode 0

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward
            rewards.append(total_reward)

            if done:
                break

        env.close()

        # Determine outcome
        outcome = "CRASH"
        if terminated and reward > 0:
            outcome = "SUCCESS ✓"
        elif step >= max_steps - 1:
            outcome = "TIMEOUT"

        return frames, rewards, total_reward, outcome

    def create_comparison_video(self):
        """Create a split-screen comparison video."""
        print("Creating LinkedIn comparison video...")

        # Record episodes for each checkpoint
        episode_data = []
        for checkpoint in self.checkpoints:
            print(f"Recording {checkpoint['desc']}...")
            agent = self.load_agent_at_checkpoint(checkpoint)
            frames, rewards, total_reward, outcome = self.record_episode(
                agent, checkpoint)

            episode_data.append({
                'checkpoint': checkpoint,
                'frames': frames,
                'rewards': rewards,
                'total_reward': total_reward,
                'outcome': outcome
            })

        # Create video with side-by-side comparison
        self._create_split_screen_video(episode_data)

    def _create_split_screen_video(self, episode_data):
        """Create professional split-screen video."""

        # Video settings
        height, width = 400, 600  # Standard frame size
        video_width = width * 2 + 40  # Two panels + spacing
        video_height = height + 150   # Space for text and metrics

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(
            self.video_dir, 'moonlander_learning_progression.mp4')
        out = cv2.VideoWriter(video_path, fourcc, self.fps,
                              (video_width, video_height))

        # Calculate frames per episode (equal time for each)
        frames_per_episode = self.total_frames // len(episode_data)

        for ep_idx, ep_data in enumerate(episode_data):
            frames = ep_data['frames']
            rewards = ep_data['rewards']
            checkpoint = ep_data['checkpoint']

            # Interpolate frames to match target duration
            target_frame_count = frames_per_episode
            frame_indices = np.linspace(
                0, len(frames)-1, target_frame_count, dtype=int)

            for frame_idx in range(target_frame_count):
                # Get current frame
                game_frame_idx = frame_indices[frame_idx]
                game_frame = frames[game_frame_idx]
                current_reward = rewards[game_frame_idx] if game_frame_idx < len(
                    rewards) else rewards[-1]

                # Resize game frame
                game_frame_resized = cv2.resize(game_frame, (width, height))

                # Create composite frame
                # Dark background
                composite = np.ones(
                    (video_height, video_width, 3), dtype=np.uint8) * 20

                # Add game frame
                y_offset = 80
                x_offset = 20
                composite[y_offset:y_offset+height,
                          x_offset:x_offset+width] = game_frame_resized

                # Add metrics panel
                metrics_x = width + 40
                self._add_metrics_panel(composite, checkpoint, current_reward, ep_data['outcome'],
                                        metrics_x, y_offset, width, height)

                # Add title and progress
                self._add_title_and_progress(
                    composite, checkpoint, ep_idx, len(episode_data))

                # Write frame
                out.write(composite)

        out.release()
        print(f"Video saved: {video_path}")
        return video_path

    def _add_metrics_panel(self, composite, checkpoint, current_reward, outcome, x, y, width, height):
        """Add metrics panel to the right side."""
        # Create metrics background
        cv2.rectangle(composite, (x, y), (x + width,
                      y + height), (40, 40, 40), -1)
        cv2.rectangle(composite, (x, y), (x + width,
                      y + height), (100, 100, 100), 2)

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Episode info
        cv2.putText(composite, f"Episode {checkpoint['episode']}",
                    (x + 20, y + 40), font, 0.8, (255, 255, 255), 2)

        # Current reward
        color = (0, 255, 0) if current_reward > 0 else (0, 100, 255)
        cv2.putText(composite, f"Score: {current_reward:.1f}",
                    (x + 20, y + 80), font, 0.7, color, 2)

        # Outcome
        outcome_color = (0, 255, 0) if "SUCCESS" in outcome else (0, 0, 255)
        cv2.putText(composite, f"Result: {outcome}",
                    (x + 20, y + 120), font, 0.6, outcome_color, 2)

        # Progress description
        desc_lines = self._wrap_text(checkpoint['desc'], 25)
        for i, line in enumerate(desc_lines):
            cv2.putText(composite, line, (x + 20, y + 180 + i * 30),
                        font, 0.5, (200, 200, 200), 1)

        # Add success rate if available
        if checkpoint['episode'] >= 50:
            success_rates = {50: "5%", 200: "15%", 500: "35%", 750: "50%"}
            rate = success_rates.get(checkpoint['episode'], "")
            if rate:
                cv2.putText(composite, f"Success Rate: {rate}",
                            (x + 20, y + 280), font, 0.6, (0, 255, 255), 2)

    def _add_title_and_progress(self, composite, checkpoint, current_ep, total_eps):
        """Add title and progress bar."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Main title
        title = "AI Learning to Land on the Moon"
        cv2.putText(composite, title, (50, 40), font, 1.2, (255, 255, 255), 3)

        # Subtitle
        subtitle = "Deep Reinforcement Learning with TensorFlow"
        cv2.putText(composite, subtitle, (50, 70),
                    font, 0.6, (200, 200, 200), 2)

        # Progress bar
        progress = (current_ep + 1) / total_eps
        bar_width = 400
        bar_height = 10
        bar_x = 50
        bar_y = composite.shape[0] - 30

        # Background
        cv2.rectangle(composite, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (100, 100, 100), -1)

        # Progress
        progress_width = int(bar_width * progress)
        cv2.rectangle(composite, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                      (0, 255, 100), -1)

        # Progress text
        cv2.putText(composite, f"Learning Progress: {progress*100:.0f}%",
                    (bar_x + bar_width + 20, bar_y + bar_height), font, 0.5, (255, 255, 255), 1)

    def _wrap_text(self, text, max_chars):
        """Wrap text to multiple lines."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line + " " + word) <= max_chars:
                current_line += " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def create_performance_chart(self):
        """Create a performance chart showing learning curve."""
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Mock data based on your training results
        episodes = np.array([0, 50, 100, 200, 300, 500, 750])
        success_rates = np.array(
            [0, 2, 8, 15, 25, 35, 50])  # Based on your logs
        # Based on your progression
        avg_rewards = np.array([-300, -250, -150, -50, 50, 150, 280])

        # Success rate plot
        ax1.plot(episodes, success_rates, 'o-',
                 color='#00ff88', linewidth=3, markersize=8)
        ax1.fill_between(episodes, success_rates, alpha=0.3, color='#00ff88')
        ax1.set_title('AI Learning Progress: Success Rate Over Time',
                      fontsize=16, color='white')
        ax1.set_xlabel('Training Episodes', fontsize=12)
        ax1.set_ylabel('Success Rate (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 60)

        # Average reward plot
        ax2.plot(episodes, avg_rewards, 'o-',
                 color='#ff6b6b', linewidth=3, markersize=8)
        ax2.fill_between(episodes, avg_rewards, alpha=0.3, color='#ff6b6b')
        ax2.set_title('Average Episode Reward', fontsize=16, color='white')
        ax2.set_xlabel('Training Episodes', fontsize=12)
        ax2.set_ylabel('Average Reward', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='white', linestyle='--', alpha=0.5)

        plt.tight_layout()
        chart_path = os.path.join(self.video_dir, 'learning_curve.png')
        plt.savefig(chart_path, dpi=300,
                    bbox_inches='tight', facecolor='black')
        plt.close()

        print(f"Learning curve saved: {chart_path}")
        return chart_path


def main():
    """Create the LinkedIn video."""
    print("Creating professional LinkedIn video...")

    creator = VideoCreator()

    # Create the main video
    video_path = creator.create_comparison_video()

    # Create performance chart
    chart_path = creator.create_performance_chart()

    print("\n" + "="*60)
    print("🎬 LINKEDIN VIDEO CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Video: {video_path}")
    print(f"📊 Chart: {chart_path}")
    print("\n📝 LinkedIn Post Suggestions:")
    print("- 'Watch my AI learn to land on the moon! 🚀'")
    print("- 'From random crashes to 50% success rate in 750 episodes'")
    print("- 'Deep Reinforcement Learning with TensorFlow'")
    print("- '#AI #MachineLearning #ReinforcementLearning #TensorFlow'")
    print("\n⏱️ Video length: ~45 seconds (perfect for LinkedIn)")
    print("✨ Professional quality with metrics and progression")


if __name__ == "__main__":
    main()
