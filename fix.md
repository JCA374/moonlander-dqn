# Better success detection for train_fixed.py
# Replace the section around line 380-400

if done:
    # FIXED: Proper success detection for LunarLander
    # Check multiple conditions for success
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state
    
    # Success conditions:
    # 1. Both legs touching (terminated with legs down)
    # 2. Low velocity when terminated
    # 3. Near center of landing pad
    # 4. Total episode reward is positive (even if final step isn't)
    
    landed_successfully = False
    
    if terminated:  # Episode ended naturally (not truncated)
        # Check if both legs are touching
        if left_leg and right_leg:
            # Check if velocity is low (good landing)
            if abs(vx) < 0.5 and abs(vy) < 0.5:
                # Check if near landing pad center
                if abs(x) < 0.5:
                    landed_successfully = True
        
        # Alternative: if total reward is very positive, count as success
        if total_reward > 200:
            landed_successfully = True
    
    if landed_successfully:
        successful_landings += 1
        logger.info(f"Episode {episode}: SUCCESSFUL LANDING! Total reward: {total_reward:.2f}, Final reward: {original_reward:.2f}")
    elif step >= args.max_steps - 1:
        timeouts += 1
        logger.info(f"Episode {episode}: TIMEOUT after {step} steps, reward: {total_reward:.2f}")
    elif terminated and total_reward < -50:
        crashes += 1
        logger.info(f"Episode {episode}: CRASH, reward: {total_reward:.2f}")
    else:
        logger.info(f"Episode {episode}: Episode ended, reward: {total_reward:.2f}")
    
    break


# Improved reward shaping for train_fixed.py
# Replace the minimal reward shaping section around line 360-375

# Extract state components for intelligent shaping
x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_state

# FIXED: More nuanced reward shaping
original_reward = reward
scaled_reward = reward

# 1. Amplify large positive rewards (successful landing bonus)
if reward > 50:  # Lowered threshold
    scaled_reward = reward * 1.5  # Stronger amplification
elif reward < -50:  # Significant crash
    scaled_reward = reward * 0.8  # Slightly reduce to encourage exploration

# 2. Add shaping to encourage landing attempts
if y < 0.5:  # Close to ground
    # Encourage being centered over landing pad
    if abs(x) < 0.5:
        scaled_reward += 0.5 * (0.5 - abs(x))  # Up to +0.25 bonus
        
        # Encourage slow descent when close to ground
        if y < 0.2 and abs(vy) < 1.0:
            scaled_reward += 1.0  # Bonus for controlled approach
            
            # Big bonus for touching legs
            if left_leg or right_leg:
                scaled_reward += 2.0
                
                # Huge bonus for both legs touching (pre-landing)
                if left_leg and right_leg and abs(vx) < 0.5:
                    scaled_reward += 5.0  # This encourages actual landing

# 3. Gentle anti-hovering mechanism
# Only penalize hovering if near ground for too long
if 0.05 < y < 0.3 and abs(vy) < 0.1 and step > 300:
    scaled_reward -= 0.2  # Small penalty to encourage landing

# 4. Time pressure only late in episode
if step > 500:  # Only in last 100 steps
    scaled_reward -= 0.05  # Gentle encouragement to land



# Add this to the reward shaping section in train_fixed.py
# This prevents the agent from learning to hover until timeout

# Additional timeout prevention
if step >= args.max_steps - 1:  # About to timeout
    # Apply penalty for timeout
    scaled_reward -= 50.0  # Significant penalty
    logger.debug(f"Timeout penalty applied at step {step}")

# Also add this modification to the done condition check:
if done:
    # If episode ended due to max steps (timeout), treat differently
    if step >= args.max_steps - 1:
        # Override any positive reward for timeouts
        if scaled_reward > 0:
            scaled_reward = -20.0  # Ensure timeout is always negative
        
        # Don't store this as a positive experience
        agent.remember(state, action, scaled_reward, next_state, True)
    
    # Rest of the success detection code...