"""
Quick validation of the Docker GPU Sandbox system.
Tests imports, tensor challenges, mock sandbox, and reward calculations.
"""
import sys
print("=" * 60)
print("SWARM-OS Docker GPU Sandbox — Validation Suite")
print("=" * 60)

# 1. Import tests
print("\n[1/5] Testing imports...")
from engine.docker_sandbox import DockerGPUSandbox, VRAM_FRACTION, CONTAINER_RAM_LIMIT
from engine.tensor_challenges import TensorChallengeGenerator, CHALLENGES
from engine.evaluator import TwoStageEvaluator
from engine.rewards import RewardCalculator, OOM_CRASH_PENALTY, EFFICIENCY_BONUS_MAX
print("  OK: All modules imported successfully")

# 2. Tensor challenge tests
print("\n[2/5] Testing tensor challenges...")
tc = TensorChallengeGenerator()
for tier in [1, 2, 3]:
    c = tc.get_challenge(tier=tier)
    assert c["tier"] == tier, f"Tier mismatch: expected {tier}, got {c['tier']}"
    assert c["raw_memory_mb"] > 500, f"Challenge must exceed 500MB limit, got {c['raw_memory_mb']}MB"
    assert len(c["code"]) > 100, "Challenge code too short"
    print(f"  Tier {tier}: '{c['name']}' — {c['raw_memory_mb']}MB raw ({len(c['code'])} chars)")

stats = tc.get_stats()
assert stats["total_issued"] == 3
print(f"  Stats: {stats['total_issued']} issued, tier={stats['current_tier']}")
print("  OK: All challenges valid")

# 3. Mock sandbox tests
print("\n[3/5] Testing mock sandbox execution...")
evaluator = TwoStageEvaluator()

# Test Outcome A: Naive code → OOMKilled
naive_result = evaluator.sandbox_execute("x = torch.randn(100)", "naive.py", mock_mode=True)
assert naive_result["status"] == "OOMKilled", f"Expected OOMKilled, got {naive_result['status']}"
print(f"  Naive code: status={naive_result['status']}, vram={naive_result['vram_peak_mb']}MB — CORRECT")

# Test Outcome B (single opt): Mixed precision → PASS (borderline)
fp16_result = evaluator.sandbox_execute(
    'with torch.autocast(device_type="cuda", dtype=torch.float16): model(x)',
    "fp16.py", mock_mode=True
)
assert fp16_result["status"] == "PASS", f"Expected PASS, got {fp16_result['status']}"
print(f"  Mixed precision: status={fp16_result['status']}, vram={fp16_result['vram_peak_mb']}MB — CORRECT")

# Test Outcome B (genius): Checkpointing + fp16 → PASS (efficient)
genius_result = evaluator.sandbox_execute(
    'from torch.utils.checkpoint import checkpoint\nwith torch.autocast(device_type="cuda", dtype=torch.float16): model(x)',
    "genius.py", mock_mode=True
)
assert genius_result["status"] == "PASS", f"Expected PASS, got {genius_result['status']}"
assert genius_result["vram_peak_mb"] < 200, f"Genius code should peak under 200MB, got {genius_result['vram_peak_mb']}MB"
print(f"  Genius code: status={genius_result['status']}, vram={genius_result['vram_peak_mb']}MB — CORRECT")
print("  OK: Mock sandbox working correctly")

# 4. Reward calculation tests
print("\n[4/5] Testing reward calculations...")
rc = RewardCalculator()

# OOM crash penalty
oom_reward = rc.oom_crash(vram_peak_mb=512, error_type="OOM_CUDA")
assert oom_reward == -1.00, f"Expected -1.00, got {oom_reward}"
print(f"  OOM crash penalty: {oom_reward:+.2f} — CORRECT")

# Valid code reward
pass_reward = rc.valid_code(vram_peak_gb=0.15)
assert pass_reward > 0.40, "Should include VRAM efficiency bonus"
print(f"  Valid code reward: {pass_reward:+.2f} — CORRECT")

# Efficiency bonus
eff_bonus = rc.efficiency_bonus(vram_peak_mb=150, budget_mb=500)
assert 0 < eff_bonus <= 0.30, f"Efficiency bonus out of range: {eff_bonus}"
print(f"  Efficiency bonus (150MB/500MB): {eff_bonus:+.2f} — CORRECT")

fpsr = rc.get_fpsr()
print(f"  FPSR: {fpsr['fpsr']}% ({fpsr['successes']}/{fpsr['attempts']})")
print("  OK: Reward system working correctly")

# 5. Constraint constants check
print("\n[5/5] Validating constraint constants...")
assert CONTAINER_RAM_LIMIT == "500m", f"RAM limit wrong: {CONTAINER_RAM_LIMIT}"
assert VRAM_FRACTION == 0.042, f"VRAM fraction wrong: {VRAM_FRACTION}"
vram_budget_mb = int(VRAM_FRACTION * 12 * 1024)
print(f"  RAM cgroup limit: {CONTAINER_RAM_LIMIT}")
print(f"  VRAM fraction: {VRAM_FRACTION} (= {vram_budget_mb}MB of 12GB)")
print(f"  OOM penalty: {OOM_CRASH_PENALTY:+.2f}")
print(f"  Efficiency bonus max: {EFFICIENCY_BONUS_MAX:+.2f}")
print("  OK: All constants correct")

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED — Docker GPU Sandbox system validated")
print("=" * 60)
print("\nNew files created:")
print("  backend/engine/docker_sandbox.py    — Docker GPU sandbox with double-lock")
print("  backend/engine/tensor_challenges.py — Calibrated dummy tensor workloads")
print("  backend/Dockerfile.sandbox          — GPU container image definition")
print("\nModified files:")
print("  backend/engine/evaluator.py         — Integrated sandbox + challenges")
print("  backend/engine/rewards.py           — Added oom_crash() + efficiency_bonus()")
print("  backend/main.py                     — New /api/sandbox/* endpoints")
