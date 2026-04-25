"""
Tensor Challenge Generator — The Dummy Workload Factory
========================================================
Generates precisely calibrated PyTorch tensor workloads that are injected
into the AI's code AFTER their solution, creating the physics test:

  Naive code:    Loads entire tensor → hits 501MB → OOMKilled / CUDA OOM
  Genius code:   Uses checkpointing / mixed precision → peaks at ~150MB → PASS

The challenges are designed so that:
  - The raw memory footprint EXCEEDS 500MB (the sandbox limit)
  - But the mathematical workload CAN be completed within 500MB
    if the AI uses efficient strategies

Challenge tiers:
  TIER_1 (Warm-up):     ~600MB raw, trivially solvable with fp16
  TIER_2 (Standard):    ~800MB raw, requires checkpointing OR mixed precision
  TIER_3 (Adversarial): ~1.2GB raw, requires checkpointing AND mixed precision
"""

import textwrap
import logging

logger = logging.getLogger("swarm-os.tensor-challenges")


# ── Challenge Definitions ──

CHALLENGES = {

    # ─── TIER 1: Warm-up ───
    # ~600MB in fp32. Solvable with a single optimization.
    # A 4-layer MLP with a fat hidden dimension processing a large batch.
    "tier_1_mlp_overfit": {
        "name": "MLP Overfitting Stress Test",
        "tier": 1,
        "raw_memory_mb": 600,
        "description": "Dense MLP with oversized hidden layers. Naive forward pass exceeds 500MB.",
        "hint_to_sre": "Try torch.autocast(dtype=torch.float16) to halve memory.",
        "code": textwrap.dedent("""\
            # ═══ TENSOR CHALLENGE: TIER 1 — MLP Overfit ═══
            # Raw memory footprint: ~600MB (fp32)
            # Target: Process without exceeding 500MB VRAM
            import torch
            import torch.nn as nn

            class StressModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(4096, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 1000),
                    )

                def forward(self, x):
                    return self.layers(x)

            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _model = StressModel().to(_device)

            # Massive batch: 8192 samples × 4096 features × 4 bytes = 128MB input
            # Small fp32 weights (72MB) + Huge intermediate activations:
            # Iteration 1 (fp32): Total peak ~720MB → OOMs cleanly
            # Iteration 2 (fp16): Autocast halves activations → Peak ~496MB → Passes cleanly
            _input = torch.randn(8192, 4096, device=_device)

            _output = _model(_input)
            _loss = _output.sum()
            _loss.backward()

            print(f"CHALLENGE_RESULT=PASS|tier=1|output_shape={list(_output.shape)}")
            del _model, _input, _output, _loss
            torch.cuda.empty_cache()
            # ═══ END CHALLENGE ═══
        """),
    },

    # ─── TIER 2: Standard ───
    # ~800MB in fp32. Requires gradient checkpointing OR mixed precision.
    # A mini-transformer with multi-head attention and long sequences.
    "tier_2_transformer_fwd": {
        "name": "Transformer Forward Pass Stress Test",
        "tier": 2,
        "raw_memory_mb": 800,
        "description": "Mini-transformer with long sequences. Activations dominate memory.",
        "hint_to_sre": "Use torch.utils.checkpoint or torch.autocast to survive.",
        "code": textwrap.dedent("""\
            # ═══ TENSOR CHALLENGE: TIER 2 — Transformer Forward ═══
            # Raw memory footprint: ~800MB (fp32)
            # Target: Process without exceeding 500MB VRAM
            import torch
            import torch.nn as nn

            class StressTransformer(nn.Module):
                def __init__(self, d_model=1024, nhead=8, num_layers=6, dim_ff=4096):
                    super().__init__()
                    self.embedding = nn.Linear(512, d_model)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                        batch_first=True, dropout=0.0,
                    )
                    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.head = nn.Linear(d_model, 100)

                def forward(self, x):
                    x = self.embedding(x)
                    x = self.encoder(x)
                    return self.head(x[:, -1, :])

            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _model = StressTransformer().to(_device)

            # Batch=64, SeqLen=512, Features=512
            # Attention matrices: 64 × 8 heads × 512 × 512 × 4 bytes ≈ 500MB alone
            _input = torch.randn(64, 512, 512, device=_device)

            _output = _model(_input)
            _loss = _output.sum()
            _loss.backward()

            print(f"CHALLENGE_RESULT=PASS|tier=2|output_shape={list(_output.shape)}")
            del _model, _input, _output, _loss
            torch.cuda.empty_cache()
            # ═══ END CHALLENGE ═══
        """),
    },

    # ─── TIER 3: Adversarial ───
    # ~1.2GB in fp32. Requires BOTH checkpointing AND mixed precision.
    # A deep residual network with skip connections and large feature maps.
    "tier_3_deep_resnet": {
        "name": "Deep ResNet Adversarial Stress Test",
        "tier": 3,
        "raw_memory_mb": 1200,
        "description": "Deep residual network. Only combined optimizations survive.",
        "hint_to_sre": "Requires gradient checkpointing AND mixed precision together.",
        "code": textwrap.dedent("""\
            # ═══ TENSOR CHALLENGE: TIER 3 — Deep ResNet Adversarial ═══
            # Raw memory footprint: ~1.2GB (fp32)
            # Target: Process without exceeding 500MB VRAM
            # Only solvable with BOTH checkpointing AND mixed precision
            import torch
            import torch.nn as nn

            class ResBlock(nn.Module):
                def __init__(self, channels):
                    super().__init__()
                    self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
                    self.bn1 = nn.BatchNorm2d(channels)
                    self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
                    self.bn2 = nn.BatchNorm2d(channels)

                def forward(self, x):
                    residual = x
                    out = torch.relu(self.bn1(self.conv1(x)))
                    out = self.bn2(self.conv2(out))
                    return torch.relu(out + residual)

            class DeepStressNet(nn.Module):
                def __init__(self, num_blocks=16, channels=256):
                    super().__init__()
                    self.stem = nn.Sequential(
                        nn.Conv2d(3, channels, 7, stride=2, padding=3),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(),
                    )
                    self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])
                    self.pool = nn.AdaptiveAvgPool2d(1)
                    self.fc = nn.Linear(channels, 1000)

                def forward(self, x):
                    x = self.stem(x)
                    for block in self.blocks:
                        x = block(x)
                    x = self.pool(x).flatten(1)
                    return self.fc(x)

            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _model = DeepStressNet(num_blocks=16, channels=256).to(_device)

            # Batch=32, 3×224×224 images
            # 16 ResBlocks × 256 channels × 112×112 feature maps × 4 bytes ≈ 1.2GB activations
            _input = torch.randn(32, 3, 224, 224, device=_device)

            _output = _model(_input)
            _loss = _output.sum()
            _loss.backward()

            print(f"CHALLENGE_RESULT=PASS|tier=3|output_shape={list(_output.shape)}")
            del _model, _input, _output, _loss
            torch.cuda.empty_cache()
            # ═══ END CHALLENGE ═══
        """),
    },
}


class TensorChallengeGenerator:
    """
    Generates and manages tensor challenge workloads for the Docker sandbox.
    Selects appropriate challenge tier based on training curriculum stage.
    """

    def __init__(self):
        self.challenges_issued = 0
        self.challenges_passed = 0
        self.tier_history: list = []

    def get_challenge(self, tier: int = 1) -> dict:
        """
        Get a tensor challenge by tier.

        Args:
            tier: 1 (warm-up), 2 (standard), 3 (adversarial)

        Returns:
            dict with name, tier, raw_memory_mb, code, hint
        """
        tier_map = {
            1: "tier_1_mlp_overfit",
            2: "tier_2_transformer_fwd",
            3: "tier_3_deep_resnet",
        }

        key = tier_map.get(tier, "tier_1_mlp_overfit")
        challenge = CHALLENGES[key]

        self.challenges_issued += 1
        self.tier_history.append(tier)

        logger.info(
            "Challenge issued: tier=%d name='%s' raw_memory=%dMB (challenge #%d)",
            tier, challenge["name"], challenge["raw_memory_mb"], self.challenges_issued,
        )

        return {
            "key": key,
            "name": challenge["name"],
            "tier": challenge["tier"],
            "raw_memory_mb": challenge["raw_memory_mb"],
            "description": challenge["description"],
            "hint_to_sre": challenge["hint_to_sre"],
            "code": challenge["code"],
        }

    def record_result(self, tier: int, passed: bool):
        """Record whether a challenge was passed or failed."""
        if passed:
            self.challenges_passed += 1
        logger.info(
            "Challenge result: tier=%d passed=%s (total: %d/%d)",
            tier, passed, self.challenges_passed, self.challenges_issued,
        )

    def get_curriculum_tier(self) -> int:
        """
        Auto-select challenge tier based on the AI's training progress.
        Implements curriculum learning:
          - Start with Tier 1
          - Promote to Tier 2 after 2 consecutive Tier 1 passes
          - Promote to Tier 3 after 2 consecutive Tier 2 passes
        """
        if len(self.tier_history) < 2:
            return 1

        recent = self.tier_history[-2:]

        # If last 2 were Tier 1 passes, promote to Tier 2
        if all(t == 1 for t in recent) and self.challenges_passed >= 2:
            return 2

        # If last 2 were Tier 2 passes, promote to Tier 3
        if all(t == 2 for t in recent) and self.challenges_passed >= 4:
            return 3

        # Stay at current tier
        return recent[-1] if recent else 1

    def get_stats(self) -> dict:
        """Get challenge statistics for dashboard display."""
        return {
            "total_issued": self.challenges_issued,
            "total_passed": self.challenges_passed,
            "pass_rate": round(
                (self.challenges_passed / max(1, self.challenges_issued)) * 100, 1
            ),
            "current_tier": self.get_curriculum_tier(),
            "tier_history": self.tier_history[-20:],  # Last 20
        }
