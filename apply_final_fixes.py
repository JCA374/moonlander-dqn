#!/usr/bin/env python3
"""Apply final fixes to train_fixed.py"""

import re

# Read the file
with open('train_fixed.py', 'r') as f:
    content = f.read()

# Fix 1: Update default parameters
content = re.sub(
    r"(--epsilon-decay.*default=)0\.995",
    r"\g<1>0.998",
    content
)
content = re.sub(
    r"(--epsilon-min.*default=)0\.01",
    r"\g<1>0.05",
    content
)
content = re.sub(
    r"(--update-target-freq.*default=)20",
    r"\g<1>10",
    content
)

# Fix 2: Add timeout override (if not already there)
if "if total_reward > 0:" not in content or "# Override any positive reward for timeouts" not in content:
    # Find the right place to insert
    done_section = content.find("if done:")
    if done_section != -1:
        # Find where to insert (after the timeout check)
        insert_point = content.find(
            "if step >= args.max_steps - 1:", done_section)
        if insert_point != -1:
            # Find the end of this if block
            next_line = content.find("\n", insert_point)
            indent_match = re.search(
                r'^(\s+)', content[insert_point:next_line])
            indent = indent_match.group(
                1) if indent_match else "                "

            # Add the override
            override_code = f"""
{indent}    # Override any positive reward for timeouts
{indent}    if total_reward > 0:
{indent}        # Ensure agent doesn't think timeout is good
{indent}        total_reward = total_reward - original_reward - 20.0
"""
            # Insert after the timeout increment
            timeout_line_end = content.find("timeouts += 1", insert_point)
            if timeout_line_end != -1:
                insert_at = content.find("\n", timeout_line_end) + 1
                content = content[:insert_at] + \
                    override_code + content[insert_at:]

# Write the fixed file
with open('train_fixed_final.py', 'w') as f:
    f.write(content)

print("✅ Fixes applied! Created train_fixed_final.py")
print("\nRun with optimal parameters:")
print("python train_fixed_final.py --episodes 500 --batch-size 32 --restart")
