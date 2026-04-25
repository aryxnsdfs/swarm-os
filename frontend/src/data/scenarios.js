// ── Simulation Scenarios ──
// Pre-built event scripts with timed actions for the demo simulation.
// Each event has a delay (ms from start), type, and payload.

export const AGENT_ROLES = {
  COMMANDER: {
    id: 'commander',
    name: 'Commander',
    icon: 'CMD',
    color: '#3b82f6',
    model: 'llama-3.1-8b-instruct-4bit',
  },
  DETECTIVE: {
    id: 'detective',
    name: 'Detective',
    icon: 'DET',
    color: '#f59e0b',
    model: 'deepseek-r1-distill-llama-8b-4bit',
  },
  CODER: {
    id: 'coder',
    name: 'Coder',
    icon: 'CDR',
    color: '#10b981',
    model: 'llama-3.1-8b-instruct-4bit',
  },
  MANAGER: {
    id: 'manager',
    name: 'Manager',
    icon: 'MGR',
    color: '#a855f7',
    model: 'llama-3.1-8b-instruct-4bit',
  },
  EVALUATOR: {
    id: 'evaluator',
    name: 'Evaluator',
    icon: 'EVL',
    color: '#ec4899',
    model: 'system',
  },
  DBA_AGENT: {
    id: 'dba_agent',
    name: 'Data Agent',
    icon: 'DBA',
    color: '#22c55e',
    model: 'llama-3.1-8b-instruct-4bit',
  },
  SRE_AGENT: {
    id: 'sre_agent',
    name: 'Reliability Agent',
    icon: 'SRE',
    color: '#06b6d4',
    model: 'llama-3.1-8b-instruct-4bit',
  },
  SECURITY_AGENT: {
    id: 'security_agent',
    name: 'Security Agent',
    icon: 'SEC',
    color: '#f97316',
    model: 'llama-3.1-8b-instruct-4bit',
  },
  COMPLIANCE_AGENT: {
    id: 'compliance_agent',
    name: 'Compliance Agent',
    icon: 'CMP',
    color: '#eab308',
    model: 'llama-3.1-8b-instruct-4bit',
  },
};

export const M2M_TRANSLATIONS = {
  'ERR_OOM | NODE_2 | VRAM_11.8GB': 'Node 2 has run out of GPU memory. VRAM usage has hit 11.8GB, which exceeds the physical limit of 12GB.',
  'ACK | DIAG_INIT | PRIO_CRIT': 'Acknowledged. Initiating critical-priority diagnostic sequence now.',
  'TRACE_OOM | torch.cuda.OutOfMemoryError | alloc_2.1GB | model.layer[24]': 'Root cause identified: PyTorch threw an out-of-memory error when trying to allocate 2.1GB for layer 24 of the model.',
  'REC_FSDP | shard_factor=4 | est_vram=3.2GB': 'Recommendation: Apply Fully Sharded Data Parallel with shard factor 4. Estimated VRAM reduction to 3.2GB per node.',
  'REQ_FSDP | CODER': 'Requesting the Coder agent to implement the FSDP sharding fix.',
  'ACK | IMPL_FSDP | ETA_45s': 'Acknowledged. Implementing FSDP configuration. Estimated time: 45 seconds.',
  'CODE_SUBMIT | fsdp_wrap.py | 23_lines': 'Code submitted: fsdp_wrap.py (23 lines). Ready for sandbox execution.',
  'SANDBOX_EXEC | fsdp_wrap.py | STATUS: RUNNING': 'Executing fsdp_wrap.py in the Docker sandbox now...',
  'SANDBOX_PASS | VRAM_3.1GB | LATENCY_12ms': 'Sandbox execution passed! VRAM reduced to 3.1GB. Inference latency: 12ms.',
  'WARN_NET | BW_SPIKE_95% | POST_FSDP': 'Warning: Network bandwidth spiked to 95% utilization after FSDP enable — an expected causal side-effect of model sharding.',
  'ERR_TCP | TIMEOUT_30s | NODE_2→NODE_3': 'TCP timeout: Node 2 cannot reach Node 3. Connection timed out after 30 seconds.',
  'DIAG_NET | NCCL_BLOCK | ring_allreduce': 'Network diagnosis: NCCL collective is blocked during ring all-reduce operation.',
  'REC_GRADIENT_CKPT | reduce_comm_40%': 'Recommendation: Enable gradient checkpointing to reduce inter-node communication by 40%.',
  'ACK | IMPL_GRAD_CKPT': 'Acknowledged. Implementing gradient checkpointing now.',
  'CODE_SUBMIT | grad_checkpoint.py | 15_lines': 'Code submitted: grad_checkpoint.py (15 lines). Ready for sandbox execution.',
  'SANDBOX_PASS | NET_BW_52% | STABLE': 'Sandbox passed! Network bandwidth stabilized at 52%. Cluster is stable.',
  'RESOLVE | INCIDENT_CLOSED | COST_$8.40 | TIME_4m12s': 'Incident resolved and closed. Total cost: $8.40. Total resolution time: 4 minutes 12 seconds.',
  'RCA_GEN | AUTO | 5_SECTIONS': 'Auto-generating Root Cause Analysis document with 5 sections.',
  'DISAGREE | CMD:RESTART vs DET:FSDP | FORK': 'DISAGREEMENT: Commander recommends full restart, Detective recommends FSDP fix. Entering structured resolution.',
  'FORK_RESOLVE | DET_WIN | REASON: cost_delta_$38.60': 'Fork resolved: Detective wins. Reason: FSDP saves $38.60 vs restart approach.',
  'ERR_SQL_TIMEOUT | REQ_EXPERT': 'SQL database timeout detected. Requesting specialist agent — this is outside core team expertise.',
  'SPAWN | DB_ADMIN | GATE_CHECK': 'Spawning DB_Admin agent. Running System Prompt Integrity Gate check...',
  'GATE_PASS | DB_ADMIN | 3/3_PROBES': 'System Prompt Integrity Gate passed: DB_Admin answered all 3 probe questions correctly.',
  'DISMISS | DB_ADMIN | VRAM_FREE': 'Dismissing DB_Admin agent. Freeing VRAM and context window.',
};

export const PRIMARY_SCENARIO = {
  name: 'PyTorch OOM → FSDP → Network Cascade',
  description: 'A cascading failure starting from GPU memory exhaustion, through FSDP fix, to network saturation.',
  totalDuration: 252000, // 4m12s in ms
  budget: 50.00,
  slaWindow: 600, // 10 minutes in seconds
  events: [
    // ── Phase 1: OOM Detection (0-15s) ──
    { delay: 0, type: 'telemetry', payload: { ram: 320, vram: 8.2, network: 30, cpu: 45, containerStatus: 'running' }},
    { delay: 2000, type: 'telemetry', payload: { ram: 380, vram: 10.5, network: 32, cpu: 62, containerStatus: 'running' }},
    { delay: 4000, type: 'telemetry', payload: { ram: 450, vram: 11.8, network: 35, cpu: 78, containerStatus: 'warning' }},
    { delay: 5000, type: 'chat', payload: { agent: 'DETECTIVE', m2m: 'ERR_OOM | NODE_2 | VRAM_11.8GB', think: 'Looking at the telemetry feed, Node 2 VRAM is at 11.8GB which is dangerously close to the 12GB physical limit on the RTX 3060. The allocation pattern suggests a model layer is trying to grab a contiguous 2GB block. This is a classic OOM precursor — I need to alert the Commander immediately before the container crashes.' }},
    { delay: 7000, type: 'chat', payload: { agent: 'COMMANDER', m2m: 'ACK | DIAG_INIT | PRIO_CRIT', think: 'Detective flagged critical OOM on Node 2. VRAM at 98.3% — we have maybe 30 seconds before an OOMKilled crash. Setting priority to CRITICAL. Need Detective to trace the exact allocation point so Coder can write a targeted fix, not a brute-force restart.' }},
    { delay: 8000, type: 'causal', payload: { id: 'oom_error', label: 'OOM Error', type: 'error', detail: 'Node 2: 11.8GB / 12GB VRAM' }},
    { delay: 9000, type: 'telemetry', payload: { ram: 480, vram: 11.9, network: 35, cpu: 85, containerStatus: 'critical' }},
    { delay: 10000, type: 'preflight', payload: { budget: true, spof: true, sla: true }},

    // ── Phase 2: Diagnosis (15-30s) ──
    { delay: 12000, type: 'chat', payload: { agent: 'DETECTIVE', m2m: 'TRACE_OOM | torch.cuda.OutOfMemoryError | alloc_2.1GB | model.layer[24]', think: 'Traced the stack: torch.cuda.OutOfMemoryError at model.layer[24] trying to allocate 2.1GB for a single attention head computation. The model has 32 layers and is not sharded — it is trying to fit the entire model on a single GPU. FSDP is the correct architectural fix here, not gradient accumulation (which would only help with batch size, not model size).' }},
    { delay: 14000, type: 'chat', payload: { agent: 'DETECTIVE', m2m: 'REC_FSDP | shard_factor=4 | est_vram=3.2GB', think: 'Running the math: current model is ~10GB. With FSDP shard_factor=4, each node holds ~2.5GB of parameters plus ~0.7GB activations = 3.2GB estimated. This leaves 8.8GB headroom on a 12GB card. The tradeoff is increased inter-node communication — I need to flag this to the Commander.' }},
    { delay: 15000, type: 'causal', payload: { id: 'diagnosis', label: 'Root Cause Found', type: 'fix', detail: 'layer[24] alloc 2.1GB', parent: 'oom_error' }},
    { delay: 16000, type: 'reward', payload: { agent: 'DETECTIVE', action: 'Correct diagnosis', value: 0.40, tag: 'DIAG' }},

    // ── Phase 3: Agent Disagreement Fork (30-45s) ──
    { delay: 18000, type: 'chat', payload: { agent: 'COMMANDER', m2m: 'DISAGREE | CMD:RESTART vs DET:FSDP | FORK', think: 'I am considering a full cluster restart as the safe option — it guarantees we return to a known-good state. However, Detective recommends FSDP which is riskier but cheaper. Let me evaluate: restart costs ~$47 (downtime penalty + SLA breach risk) vs FSDP at ~$8.40. The math favors FSDP but only if the implementation is correct on first pass.' }},
    { delay: 19000, type: 'disagreement', payload: { active: true, position1: { agent: 'COMMANDER', action: 'Full Cluster Restart', cost: '$47.00', risk: 'SLA breach likely' }, position2: { agent: 'DETECTIVE', action: 'FSDP Sharding Fix', cost: '$8.40', risk: 'Network spike possible' }}},
    { delay: 20000, type: 'causal', payload: { id: 'fork', label: 'Reasoning Fork', type: 'fork', detail: 'RESTART vs FSDP', parent: 'diagnosis' }},
    { delay: 22000, type: 'chat', payload: { agent: 'COMMANDER', m2m: 'FORK_RESOLVE | DET_WIN | REASON: cost_delta_$38.60', think: 'Resolved: Detective approach wins. Cost delta is $38.60 in favor of FSDP. The network spike risk is manageable — if it occurs, we can apply gradient checkpointing as a secondary fix. Proceeding with FSDP.' }},
    { delay: 23000, type: 'disagreement', payload: { active: false, resolution: 'DETECTIVE wins — FSDP saves $38.60 vs restart' }},
    { delay: 24000, type: 'causal', payload: { id: 'fork_resolve', label: 'Fork Resolved → FSDP', type: 'resolution', detail: 'Detective wins: cost_delta $38.60', parent: 'fork' }},

    // ── Phase 4: FSDP Implementation (45-75s) ──
    { delay: 26000, type: 'chat', payload: { agent: 'COMMANDER', m2m: 'REQ_FSDP | CODER', think: 'Dispatching FSDP implementation task to Coder. Setting a 60-second timeout — if Coder does not submit code within that window, I will escalate.' }},
    { delay: 28000, type: 'chat', payload: { agent: 'CODER', m2m: 'ACK | IMPL_FSDP | ETA_45s', think: 'Received FSDP task. I will wrap the model with FullyShardedDataParallel using shard_factor=4 as Detective recommended. I need to import from torch.distributed.fsdp, set the auto_wrap_policy, and configure the sharding strategy. Estimated 23 lines of code.' }},
    { delay: 30000, type: 'preflight', payload: { budget: true, spof: true, sla: true }},
    { delay: 35000, type: 'chat', payload: { agent: 'CODER', m2m: 'CODE_SUBMIT | fsdp_wrap.py | 23_lines', think: 'Code complete. FSDP wrapper with auto_wrap_policy targeting transformer layers, shard_factor=4, mixed precision enabled for memory savings. Submitting to sandbox for validation.' }},
    { delay: 36000, type: 'reward', payload: { agent: 'CODER', action: 'Code submitted', value: 0.0, tag: 'SUBMIT' }},
    { delay: 37000, type: 'git', payload: { hash: 'a3f7c2d', message: 'feat: add FSDP wrapper with shard_factor=4', files: ['fsdp_wrap.py'] }},

    // ── Phase 5: Sandbox Execution (75-90s) ──
    { delay: 38000, type: 'chat', payload: { agent: 'CODER', m2m: 'SANDBOX_EXEC | fsdp_wrap.py | STATUS: RUNNING' }},
    { delay: 39000, type: 'telemetry', payload: { ram: 420, vram: 9.8, network: 38, cpu: 72, containerStatus: 'running' }},
    { delay: 42000, type: 'telemetry', payload: { ram: 350, vram: 5.2, network: 55, cpu: 55, containerStatus: 'running' }},
    { delay: 45000, type: 'chat', payload: { agent: 'CODER', m2m: 'SANDBOX_PASS | VRAM_3.1GB | LATENCY_12ms', think: 'Sandbox passed! VRAM dropped from 11.8GB to 3.1GB — a 73.7% reduction. Inference latency is 12ms which is within acceptable range. The Docker container survived the 500MB RAM constraint.' }},
    { delay: 46000, type: 'telemetry', payload: { ram: 280, vram: 3.1, network: 85, cpu: 42, containerStatus: 'running' }},
    { delay: 47000, type: 'causal', payload: { id: 'fsdp_fix', label: 'FSDP Fix Applied', type: 'fix', detail: 'VRAM: 11.8GB → 3.1GB', parent: 'fork_resolve' }},
    { delay: 48000, type: 'reward', payload: { agent: 'CODER', action: 'Sandbox PASS (Mock Tensor)', value: 0.40, tag: 'SANDBOX' }},

    // ── Phase 6: Causal Escalation — Network Spike (90-130s) ──
    { delay: 50000, type: 'telemetry', payload: { ram: 280, vram: 3.1, network: 95, cpu: 45, containerStatus: 'warning' }},
    { delay: 51000, type: 'chat', payload: { agent: 'DETECTIVE', m2m: 'WARN_NET | BW_SPIKE_95% | POST_FSDP', think: 'As I predicted — FSDP sharding means the 4 GPU nodes now need constant communication for gradient synchronization. Network bandwidth jumped from 38% to 95%. This is the causal butterfly effect: fixing OOM created a network bottleneck. Need to monitor for TCP timeouts.' }},
    { delay: 52000, type: 'causal', payload: { id: 'net_spike', label: 'Network Spike 95%', type: 'escalation', detail: 'Causal: FSDP → all-reduce traffic', parent: 'fsdp_fix' }},
    { delay: 55000, type: 'telemetry', payload: { ram: 285, vram: 3.2, network: 98, cpu: 48, containerStatus: 'critical' }},
    { delay: 57000, type: 'chat', payload: { agent: 'DETECTIVE', m2m: 'ERR_TCP | TIMEOUT_30s | NODE_2→NODE_3', think: 'Confirmed: TCP timeout between Node 2 and Node 3. The ring all-reduce pattern means Node 3 cannot receive gradient shards from Node 2 in time. NCCL will block and eventually crash the training loop if not resolved.' }},
    { delay: 58000, type: 'causal', payload: { id: 'tcp_timeout', label: 'TCP Timeout', type: 'error', detail: 'Node 2→3, 30s timeout', parent: 'net_spike' }},

    // ── Phase 7: Network Fix (130-180s) ──
    { delay: 60000, type: 'chat', payload: { agent: 'DETECTIVE', m2m: 'DIAG_NET | NCCL_BLOCK | ring_allreduce' }},
    { delay: 62000, type: 'chat', payload: { agent: 'DETECTIVE', m2m: 'REC_GRADIENT_CKPT | reduce_comm_40%', think: 'Gradient checkpointing will trade compute for communication. Instead of storing all activations and syncing them, we recompute during backward pass. This reduces the volume of inter-node gradient data by approximately 40%, which should bring bandwidth within tolerance.' }},
    { delay: 64000, type: 'reward', payload: { agent: 'DETECTIVE', action: 'Network diagnosis', value: 0.40, tag: 'DIAG' }},
    { delay: 65000, type: 'chat', payload: { agent: 'CODER', m2m: 'ACK | IMPL_GRAD_CKPT' }},
    { delay: 68000, type: 'preflight', payload: { budget: true, spof: true, sla: true }},
    { delay: 72000, type: 'chat', payload: { agent: 'CODER', m2m: 'CODE_SUBMIT | grad_checkpoint.py | 15_lines' }},
    { delay: 73000, type: 'git', payload: { hash: 'b8e1f4a', message: 'feat: enable gradient checkpointing for comm reduction', files: ['grad_checkpoint.py'] }},
    { delay: 76000, type: 'telemetry', payload: { ram: 290, vram: 3.4, network: 52, cpu: 55, containerStatus: 'running' }},
    { delay: 78000, type: 'chat', payload: { agent: 'CODER', m2m: 'SANDBOX_PASS | NET_BW_52% | STABLE' }},
    { delay: 79000, type: 'causal', payload: { id: 'grad_ckpt', label: 'Gradient Checkpointing', type: 'fix', detail: 'Network: 95% → 52%', parent: 'tcp_timeout' }},
    { delay: 80000, type: 'reward', payload: { agent: 'CODER', action: 'Sandbox PASS (Network fix)', value: 0.40, tag: 'SANDBOX' }},
    { delay: 81000, type: 'telemetry', payload: { ram: 260, vram: 3.3, network: 48, cpu: 50, containerStatus: 'stable' }},

    // ── Phase 8: Resolution & RCA (180-252s) ──
    { delay: 84000, type: 'chat', payload: { agent: 'COMMANDER', m2m: 'RESOLVE | INCIDENT_CLOSED | COST_$8.40 | TIME_4m12s', think: 'All systems green. VRAM stable at 3.3GB, network at 48%, CPU nominal. Total incident cost: $8.40 well under the $50 budget ceiling. SLA maintained with 5m48s remaining. Initiating auto-RCA generation.' }},
    { delay: 85000, type: 'causal', payload: { id: 'resolved', label: 'Incident Resolved', type: 'resolution', detail: '$8.40 | 4m12s | SLA Safe', parent: 'grad_ckpt' }},
    { delay: 86000, type: 'telemetry', payload: { ram: 240, vram: 3.1, network: 42, cpu: 38, containerStatus: 'stable' }},
    { delay: 88000, type: 'chat', payload: { agent: 'COMMANDER', m2m: 'RCA_GEN | AUTO | 5_SECTIONS' }},
    { delay: 89000, type: 'reward', payload: { agent: 'COMMANDER', action: 'Auto-RCA generated', value: 0.20, tag: 'RCA' }},
    { delay: 90000, type: 'git', payload: { hash: 'c9d2e5f', message: 'docs: auto-generated RCA report', files: ['RCA_REPORT.md'] }},
    { delay: 92000, type: 'counterfactual', payload: {
      actual: { time: '4m 12s', cost: '$8.40', sla: 'SAFE', outcome: 'RESOLVED' },
      dead: { time: '10m 00s', cost: '$47.00', sla: 'BREACHED', outcome: 'CLUSTER_DOWN' },
    }},
  ],
};

export const REWARD_HISTORY_SEED = [
  { episode: 1, reward: -2.1 },
  { episode: 2, reward: -1.8 },
  { episode: 3, reward: -1.2 },
  { episode: 4, reward: -0.8 },
  { episode: 5, reward: -0.3 },
  { episode: 6, reward: 0.1 },
  { episode: 7, reward: 0.4 },
  { episode: 8, reward: 0.6 },
  { episode: 9, reward: 0.9 },
  { episode: 10, reward: 1.1 },
  { episode: 11, reward: 1.0 },
  { episode: 12, reward: 1.3 },
  { episode: 13, reward: 1.5 },
  { episode: 14, reward: 1.4 },
  { episode: 15, reward: 1.7 },
  { episode: 16, reward: 1.8 },
  { episode: 17, reward: 1.6 },
  { episode: 18, reward: 1.9 },
  { episode: 19, reward: 2.0 },
  { episode: 20, reward: 2.1 },
];

export const COMPRESSION_DATA = [
  { round: 1, avgTokens: 12, example: 'ERR_OOM | REQ_FSDP | NODE_2' },
  { round: 2, avgTokens: 10, example: 'ERR_OOM|FSDP|N2' },
  { round: 3, avgTokens: 8,  example: 'OOM|FSDP|N2' },
  { round: 4, avgTokens: 7,  example: 'O|F|N2' },
  { round: 5, avgTokens: 6,  example: 'O|F|2' },
];

export const FPSR_DATA = [
  { label: 'Baseline (Untrained)', fpsr: 15 },
  { label: 'After 5 Episodes', fpsr: 35 },
  { label: 'After 10 Episodes', fpsr: 58 },
  { label: 'After 20 Episodes', fpsr: 82 },
];

export const BEFORE_AFTER = {
  before: {
    title: 'Untrained Agent (Episode 1)',
    code: `# Agent tries brute-force restart
import subprocess
subprocess.run(["docker", "restart", "node_2"])
# Result: OOMKilled again in 30s
# No FSDP, no sharding
# Cost: $47.00 (SLA breached)`,
    reward: -1.00,
    fpsr: '15%',
    tokens: 12,
  },
  after: {
    title: 'Trained Agent (Episode 20)',
    code: `# Agent applies targeted FSDP fix
from torch.distributed.fsdp import FSDP
model = FSDP(model, sharding_strategy=
    ShardingStrategy.FULL_SHARD)
# Result: VRAM 11.8GB → 3.1GB
# Cost: $8.40 (SLA safe)`,
    reward: 0.40,
    fpsr: '82%',
    tokens: 6,
  },
};

export const RCA_DOCUMENT = `# Root Cause Analysis Report
## Incident: PyTorch Cluster OOM Failure

### 1. Executive Summary
A cascading failure originating from GPU memory exhaustion on Node 2 was detected, diagnosed, and resolved by the Swarm-OS agent coalition within 4 minutes 12 seconds, well within the 10-minute SLA window. Total incident cost: $8.40 (budget: $50.00).

### 2. Timeline of Events
| Time | Event | Agent |
|------|-------|-------|
| 00:00 | OOM detected on Node 2 (11.8GB / 12GB VRAM) | Detective |
| 00:12 | Root cause traced to layer[24] allocating 2.1GB | Detective |
| 00:18 | Reasoning fork: Restart vs FSDP | Commander |
| 00:22 | Fork resolved: FSDP selected (saves $38.60) | Commander |
| 00:35 | FSDP wrapper submitted (23 lines) | Coder |
| 00:45 | Sandbox PASS — VRAM reduced to 3.1GB | Coder |
| 00:51 | Causal escalation: Network spike to 95% | Detective |
| 01:12 | Gradient checkpointing fix submitted | Coder |
| 01:18 | Sandbox PASS — Network stabilized at 52% | Coder |
| 01:24 | Incident resolved | Commander |

### 3. Root Cause
The model was loaded without distributed sharding, attempting to fit the entire 10GB model on a single 12GB GPU. Layer 24 attempted a contiguous 2.1GB allocation that exceeded available VRAM.

### 4. Resolution
Two-phase fix:
1. **FSDP Wrapping** — Model sharded across 4 nodes with \`shard_factor=4\`, reducing per-node VRAM from 11.8GB to 3.1GB.
2. **Gradient Checkpointing** — Reduced inter-node communication overhead from 95% → 52% bandwidth utilization.

### 5. Lessons Learned
- FSDP fixes OOM but creates network pressure (butterfly effect)
- Gradient checkpointing is an effective secondary fix for FSDP-induced network saturation
- Cost-benefit analysis during reasoning fork prevented $38.60 in unnecessary spending
`;

export const MODEL_CONFIG = {
  active_model: 'llama-3.1-8b-instruct-4bit',
  models: {
    'llama-3.1-8b-instruct-4bit': {
      name: 'Llama-3.1-8B-Instruct (4-bit QLoRA)',
      hf_id: 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
      max_vram_gb: 6,
      context_length: 8192,
      quantization: '4bit',
      tokens_per_sec: '35-50',
      best_for: ['COMMANDER', 'CODER'],
      suitable_for: 'RTX 3060 12GB',
    },
    'deepseek-r1-distill-llama-8b-4bit': {
      name: 'DeepSeek-R1-Distill-Llama-8B (4-bit QLoRA)',
      hf_id: 'unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit',
      max_vram_gb: 5.5,
      context_length: 8192,
      quantization: '4bit',
      tokens_per_sec: '30-45',
      best_for: ['DETECTIVE'],
      suitable_for: 'RTX 3060 12GB',
      notes: 'Superior CoT reasoning for causal trace analysis.',
    },
    'llama-3.1-70b-instruct': {
      name: 'Llama-3.1-70B-Instruct',
      hf_id: 'meta-llama/Llama-3.1-70B-Instruct',
      max_vram_gb: 40,
      context_length: 8192,
      quantization: 'none',
      tokens_per_sec: '10-20',
      best_for: ['COMMANDER', 'DETECTIVE', 'CODER'],
      suitable_for: 'A100/H100',
    },
    'llama-3.1-8b-instruct': {
      name: 'Llama-3.1-8B-Instruct (Full Precision)',
      hf_id: 'meta-llama/Llama-3.1-8B-Instruct',
      max_vram_gb: 16,
      context_length: 8192,
      quantization: 'none',
      tokens_per_sec: '20-35',
      best_for: ['COMMANDER', 'CODER'],
      suitable_for: 'RTX 4090 24GB',
    },
  },
  agent_model_overrides: {
    DETECTIVE: 'deepseek-r1-distill-llama-8b-4bit',
  },
};
