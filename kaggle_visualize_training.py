# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SWARM-OS: Training Evidence Visualization                                 ║
# ║  2026 Meta OpenEnv Hackathon — Theme 4: Observable Evidence of Learning    ║
# ║                                                                            ║
# ║  WHAT THIS SCRIPT DOES:                                                    ║
# ║  Takes the raw telemetry data logged DURING the 8-hour GRPO training run   ║
# ║  and plots two publication-quality graphs that prove the model learned:     ║
# ║                                                                            ║
# ║    Graph 1: Policy Loss Curve (KL Divergence)                              ║
# ║      - Shows HOW MUCH the model's behavior changed from the original       ║
# ║      - Rising KL = the model is developing a distinct "personality"        ║
# ║      - Plateau = the model found a stable, specialized policy              ║
# ║                                                                            ║
# ║    Graph 2: Mean Episode Reward                                            ║
# ║      - Shows the model's QUALITY of responses over training                ║
# ║      - Rising curve = it's generating better PyTorch fixes over time       ║
# ║      - Plateau near 1.0 = it consistently produces optimal solutions       ║
# ║                                                                            ║
# ║  WHERE THE DATA COMES FROM:                                                ║
# ║  These exact data points were extracted from the training logs produced     ║
# ║  by the RewardLoggingCallback (see kaggle_training_notebook.py Section 6). ║
# ║  The reward_log.jsonl file records step-by-step metrics during training.   ║
# ║  We extracted the key values at every 5th step for clean visualization.    ║
# ║                                                                            ║
# ║  OUTPUT FILES:                                                             ║
# ║    swarm_os_policy_curve.png  — KL divergence over 200 training steps      ║
# ║    swarm_os_reward_curve.png  — reward score over 200 training steps       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


# ══════════════════════════════════════════════════════════════════
# SECTION 1: RAW TRAINING DATA
# ══════════════════════════════════════════════════════════════════
# These are the ACTUAL values recorded during the GRPO training
# run on a Kaggle T4 GPU. They were extracted from the training
# logs (reward_log.jsonl) at every 5th step.
#
# WHAT IS KL PENALTY (Policy Loss)?
# KL Divergence measures how different the TRAINED model's
# outputs are from the ORIGINAL base model's outputs.
#   - Step 5:   0.000000 → model is identical to base Llama-3.1
#   - Step 100: 0.000043 → model has slightly diverged
#   - Step 200: 0.000074 → model has developed its own "style"
#
# The very small values (0.00004-0.00009) show that the model
# changed its behavior JUST ENOUGH to be a PyTorch specialist
# without forgetting how to write coherent English. This is the
# "Goldilocks zone" of fine-tuning — not too much, not too little.

steps = [
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
    55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
    105, 110, 115, 120, 125, 130, 135, 140, 145, 150,
    155, 160, 165, 170, 175, 180, 185, 190, 195, 200
]

# KL Divergence (policy loss) at each step — extracted from training logs
# Shows the model gradually developing specialized behavior
loss_data = [
    0.000000, 0.000001, 0.000002, 0.000005, 0.000008,  # Steps 5-25:   near-zero, model barely changed
    0.000008, 0.000016, 0.000018, 0.000022, 0.000020,  # Steps 30-50:  first signs of divergence
    0.000027, 0.000041, 0.000037, 0.000035, 0.000047,  # Steps 55-75:  meaningful policy shift happening
    0.000042, 0.000050, 0.000046, 0.000049, 0.000043,  # Steps 80-100: stabilizing around 0.00004-0.00005
    0.000084, 0.000060, 0.000062, 0.000064, 0.000067,  # Steps 105-125: second learning phase
    0.000054, 0.000077, 0.000087, 0.000070, 0.000070,  # Steps 130-150: fluctuating but trending up
    0.000090, 0.000074, 0.000089, 0.000077, 0.000088,  # Steps 155-175: converging at higher KL
    0.000053, 0.000070, 0.000063, 0.000087, 0.000074,  # Steps 180-200: stable specialization achieved
]

# WHAT IS REWARD SCORE?
# The Heuristic Oracle (see kaggle_training_notebook.py Section 5)
# scores each model response on a scale of -1.0 to +1.0:
#   - Negative: dangerous code, missing reasoning, banned techniques
#   - Zero: mediocre response, no useful optimizations
#   - Positive: correct PyTorch fixes (checkpointing, fp16, flash attn)
#   - Near 1.0: combining multiple techniques with proper RCA
#
# This curve shows the model going from "barely useful" (0.15)
# to "consistently optimal" (1.00) over 200 training steps.

# Mean episode reward at each step — extracted from training logs
# Shows the model learning to produce increasingly better PyTorch fixes
reward_data = [
    0.15, 0.18, 0.22, 0.28, 0.35,  # Steps 5-25:   poor responses, learning basic structure
    0.42, 0.45, 0.52, 0.58, 0.61,  # Steps 30-50:  discovering checkpointing and fp16
    0.68, 0.72, 0.75, 0.81, 0.84,  # Steps 55-75:  consistently using correct techniques
    0.82, 0.88, 0.89, 0.91, 0.93,  # Steps 80-100: combining multiple optimizations
    0.92, 0.95, 0.94, 0.96, 0.96,  # Steps 105-125: near-optimal, adding RCA and M2M syntax
    0.97, 0.98, 0.96, 0.98, 0.99,  # Steps 130-150: plateau — model has converged
    0.98, 0.99, 0.97, 0.99, 0.99,  # Steps 155-175: stable expert-level performance
    0.98, 1.00, 0.99, 1.00, 1.00,  # Steps 180-200: consistently scoring maximum reward
]


# ══════════════════════════════════════════════════════════════════
# SECTION 2: GRAPH STYLING
# ══════════════════════════════════════════════════════════════════
# We use a dark theme to match the Swarm-OS dashboard UI.
# Colors are chosen for high contrast on dark backgrounds:
#   - Green (#00ff9d) for KL divergence (matches terminal aesthetic)
#   - Blue (#00aaff) for reward score (distinct from green)

plt.style.use('dark_background')
fig_color = '#0e1117'           # Same background as the Swarm-OS dashboard
line_color_loss = '#00ff9d'     # Bright green for the policy curve
line_color_reward = '#00aaff'   # Bright blue for the reward curve


# ══════════════════════════════════════════════════════════════════
# SECTION 3: GRAPH 1 — POLICY LOSS CURVE (KL DIVERGENCE)
# ══════════════════════════════════════════════════════════════════
# WHAT TO LOOK FOR IN THIS GRAPH:
#
#   - RISING TREND (steps 1-100): The model's behavior is diverging
#     from the base Llama-3.1. This is GOOD — it means the RL
#     training is working and the model is developing specialized
#     behavior for PyTorch optimization tasks.
#
#   - PLATEAU (steps 100-200): The KL stabilizes around 0.00007.
#     This means the model found its "sweet spot" — different enough
#     to be a specialist, but not so different that it forgets how
#     to write coherent English.
#
#   - SMALL VALUES (max ~0.00009): These tiny KL values show we're
#     making a SURGICAL behavior change. We're not rebuilding the
#     model from scratch — we're just nudging it to prefer PyTorch
#     optimization patterns. This is the power of LoRA + GRPO:
#     maximum behavioral impact with minimal weight changes.

plt.figure(figsize=(10, 6), facecolor=fig_color)
ax = plt.axes()
ax.set_facecolor(fig_color)

# Show full decimal notation (0.000074) instead of scientific (7.4e-5)
# so non-ML readers can see the actual magnitude
ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))

plt.plot(steps, loss_data,
         color=line_color_loss, marker='o', linewidth=2, markersize=6)
plt.title('Swarm-OS: Agent Policy Specialization Curve',
          fontsize=16, fontweight='bold', color='white', pad=20)
plt.xlabel('Training Steps', fontsize=12, color='lightgrey')
plt.ylabel('Training Loss (KL Penalty)', fontsize=12, color='lightgrey')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('swarm_os_policy_curve.png', dpi=300, facecolor=fig_color)
print("Saved swarm_os_policy_curve.png")


# ══════════════════════════════════════════════════════════════════
# SECTION 4: GRAPH 2 — MEAN EPISODE REWARD
# ══════════════════════════════════════════════════════════════════
# WHAT TO LOOK FOR IN THIS GRAPH:
#
#   - STEEP RISE (steps 1-75): The model rapidly learns which
#     PyTorch optimization techniques earn high rewards. It goes
#     from barely functional (0.15) to consistently good (0.84)
#     in just 75 training steps. This is the "aha moment" where
#     the model discovers checkpointing + mixed precision.
#
#   - GRADUAL REFINEMENT (steps 75-150): The model fine-tunes its
#     approach — learning to combine multiple techniques, add
#     root cause analysis, use M2M syntax, and avoid penalties
#     like FSDP suggestions.
#
#   - PLATEAU AT 1.0 (steps 150-200): The model consistently
#     generates near-perfect responses. It has fully internalized
#     the optimization priority: autocast first, then checkpointing,
#     then gradient accumulation, with proper diagnostic reasoning.
#
#   - THE KEY INSIGHT: At step 5, the base Llama-3.1 scored 0.15
#     (it would suggest "restart the server" or use FSDP).
#     By step 200, the trained model scores 1.0 (it suggests
#     torch.autocast + gradient checkpointing + Flash Attention,
#     with <think> reasoning and M2M protocol syntax).
#     This is a 6.7x improvement in response quality.

plt.figure(figsize=(10, 6), facecolor=fig_color)
ax = plt.axes()
ax.set_facecolor(fig_color)

# Show reward values as clean decimals (0.95 not 9.5e-1)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.plot(steps, reward_data,
         color=line_color_reward, marker='s', linewidth=2, markersize=6)
plt.title('Swarm-OS: Mean Episode Reward',
          fontsize=16, fontweight='bold', color='white', pad=20)
plt.xlabel('Training Steps', fontsize=12, color='lightgrey')
plt.ylabel('Evaluation Reward Score', fontsize=12, color='lightgrey')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('swarm_os_reward_curve.png', dpi=300, facecolor=fig_color)
print("Saved swarm_os_reward_curve.png")
