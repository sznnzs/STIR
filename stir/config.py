from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name_or_path: str
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    enable_chunked_prefill: bool = False
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.6
    # vLLM memory knobs (important for stability on shared GPUs):
    # - max_model_len controls KV cache scaling (we don't need 32k for these tasks).
    # - max_num_seqs controls sampler warmup peak memory (default can be 1024 and OOM).
    max_model_len: int = 8192
    max_num_seqs: int = 64


@dataclass
class TaskConfig:
    dataset: str = "gsm8k"
    # Prefer local datasets to avoid network dependency.
    data_root: str = "datasets"

    # Offline stages use train_split; online eval uses eval_split.
    train_split: str = "train"
    eval_split: str = "test"

    # Limits for fast iteration (None means full split)
    max_train_examples: int | None = 200
    max_eval_examples: int | None = 200


@dataclass
class PromptConfig:
    template: str = "gsm8k_0shot"


@dataclass
class DecodeConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    logprobs: int | None = 1


@dataclass
class ControlPointConfig:
    M: int = 6
    # Segment boundary marker. Online/offline segmented decoding stops at this delimiter.
    segment_delimiter: str = "\n\n"


@dataclass
class OfflineMineConfig:
    # Mining method:
    # - "contrastive": use reward-ranked positives/negatives (default, original behavior)
    # - "random": sample positives/negatives at random for ablation
    method: str = "contrastive"
    # Rollouts for contrastive mining
    K: int = 8
    # Number of dataset examples to process per vLLM batch during mining.
    # Each example still produces K rollouts (SamplingParams.n=K).
    batch_size_examples: int = 1
    K_pos: int = 4
    K_neg: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 512
    eta0: float = 0.001

    # Candidate selection before Stage II
    keep_top_C: int = 512

    # If None, use "top third layers" heuristic based on model depth.
    # Supports:
    # - absolute indices (0-based), e.g. [18]
    # - ratios, e.g. ["1/5", "0.6"] (resolved at runtime based on model depth)
    candidate_layers: list[int | str] | None = None

    # Quality knob: require contrast between correct and incorrect rollouts when mining.
    # When enabled, positives are drawn from correct rollouts and negatives from incorrect rollouts;
    # control points with no such contrast are skipped (prevents mining "short-vs-long wrong" vectors).
    require_correct_and_incorrect: bool = True
    min_correct_rollouts: int = 1
    min_incorrect_rollouts: int = 1

    # Positive source for mining:
    # - "rollout": use sampled correct rollouts as positives (requires correct rollouts to exist)
    # - "gold": use dataset gold solution text as positive (GSM8K answer contains rationale)
    # - "rollout_or_gold": prefer rollout positives; fallback to gold when no correct rollouts
    pos_source: str = "rollout"


@dataclass
class OfflineSelectConfig:
    # Library size
    B: int = 64
    # Diversity weight (lambda in the draft)
    lambda_diversity: float = 0.5
    # Jitter for numerical stability
    epsilon: float = 1e-4
    # Selection method: dpp / top / random
    method: str = "dpp"
    random_seed: int = 0
    # Ensure some coverage across control points m (min tools per m) before global selection.
    min_per_control_point: int = 0


@dataclass
class OfflineMemoryConfig:
    # Stage III writes a retrieval index from Stage II library.
    # Keys are L2-normalized by default for cosine-style similarity search.
    normalize_keys: bool = True


@dataclass
class OnlineConfig:
    # Retrieval: consider top-k nearest memory entries at each control point.
    k_retrieve: int = 16
    # Score: Score_i = beta * Ahat_i + rho * (logprob_i - logprob_null)
    beta: float = 1.0
    rho: float = 0.1
    # Probe: number of candidates to probe (null is always added).
    L: int = 2
    # Probe: per-candidate probe token cap (e.g., 4-8).
    probe_tokens: int = 8
    # Injection strength: steer = (k_scale * Score_i) * v_i  (v_i is raw delta; no normalization)
    k_scale: float = 1.0
    # If best non-null score < tau_null, force null action.
    tau_null: float = 0.0
    # Protocol name: P1 or P2 (P2 matching is handled in eval runner)
    protocol: str = "P1"
    # Variant for ablations:
    # - "stir": episodic retrieval + probing
    # - "no_memory": disable retrieval/probing (always null)
    # - "no_probing": episodic retrieval, but commit best Score without probing
    # - "use_random_memory": use retrieved scores but inject random steer vectors
    # - "random_use_memory": pick a random retrieved candidate (probing unchanged)
    variant: str = "stir"
    # Retrieval confidence gating:
    # If max cosine similarity(top-1) < min_sim, treat memory as uninformative (skip probing / behave like greedy).
    min_sim: float = 0.0
    # If available memory entries for the current control point m < min_entries, treat memory as uninformative.
    min_entries: int = 1
    # If set, disable tool usage (and probing) after this OFFLINE control_point_m.
    # Note: offline control_point_m refers to the state AFTER generating m segments.
    max_tool_m: int | None = None
    # If set, disable tool usage (and probing) before this OFFLINE control_point_m.
    min_tool_m: int | None = None
    # Online eval batching: number of examples to process together per segment step.
    # - 1: legacy per-example loop (most stable).
    # - >1: segment-wise batching across examples for higher throughput.
    batch_size_examples: int = 1


@dataclass
class EvalConfig:
    # Methods to run
    methods: list[str] = None  # type: ignore[assignment]
    # Number of improve/regress examples to include in cases markdown.
    case_top_n: int = 20
    # Optional: load offline artifacts (library/memory) from another run directory.
    # This enables fast online hyperparameter sweeps without rerunning mine/select/memory.
    artifact_run_dir: str | None = None

    def __post_init__(self) -> None:
        if self.methods is None:
            self.methods = ["greedy", "stir"]
        if self.ablations is None:
            self.ablations = []

    # Ablation variants to run at the largest budget, e.g. ["no_memory", "no_probing"]
    ablations: list[str] = None  # type: ignore[assignment]


@dataclass
class OutputConfig:
    root_dir: str
    run_name: str = "debug"
    run_id: str | None = None

    def __post_init__(self) -> None:
        # IMPORTANT: run_id must be stable within a run, otherwise artifacts
        # (logs/config/metrics) will be scattered across different folders.
        if self.run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            return
        if isinstance(self.run_id, str) and self.run_id.lower() == "latest":
            latest_path = Path(self.root_dir) / self.run_name / "LATEST"
            if not latest_path.exists():
                raise FileNotFoundError(
                    f"outputs.run_id=latest but {latest_path} is missing. Run mine first, or set run_id explicitly in the config."
                )
            rid = latest_path.read_text(encoding="utf-8").strip()
            if not rid:
                raise ValueError(f"{latest_path} is empty; cannot resolve run_id.")
            self.run_id = rid
    
    def set_run_id(self, run_id: str | None) -> None:
        self.run_id = run_id
        self.__post_init__()

    @property
    def run_dir(self) -> str:
        assert self.run_id is not None
        return str(Path(self.root_dir) / self.run_name / self.run_id)

    @property
    def log_dir(self) -> str:
        return str(Path(self.run_dir) / "logs")


@dataclass
class STIRConfig:
    seed: int
    model: ModelConfig
    task: TaskConfig
    prompt: PromptConfig
    decode: DecodeConfig
    control_points: ControlPointConfig
    offline_mine: OfflineMineConfig
    offline_select: OfflineSelectConfig
    offline_memory: OfflineMemoryConfig
    online: OnlineConfig
    eval: EvalConfig
    outputs: OutputConfig


def _dc(cls, d: dict[str, Any]) -> Any:
    """
    Dataclass constructor that ignores unknown keys.

    This keeps configs forward/backward compatible when the schema changes.
    """
    if d is None:
        d = {}
    if not isinstance(d, dict):
        raise TypeError(f"Expected dict for {cls.__name__}, got {type(d)}")
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in d.items() if k in allowed}
    return cls(**filtered)


def _apply_legacy_aliases(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Backward-compatible aliases for older YAML keys.

    The config loader ignores unknown keys, so we remap a few commonly-used legacy fields
    into the current schema to keep historical configs runnable.
    """
    online = raw.get("online")
    if isinstance(online, dict):
        # online.k (legacy) -> online.k_retrieve (current)
        if "k_retrieve" not in online and "k" in online:
            online["k_retrieve"] = online.get("k")
        # online.alpha_mult (legacy) -> online.k_scale (current)
        if "k_scale" not in online and "alpha_mult" in online:
            online["k_scale"] = online.get("alpha_mult")
    return raw


def load_config(path: str) -> STIRConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config root must be a dict, got {type(raw)}")
    raw = _apply_legacy_aliases(raw)

    cfg = STIRConfig(
        seed=int(raw.get("seed", 0)),
        model=_dc(ModelConfig, raw.get("model", {})),
        task=_dc(TaskConfig, raw.get("task", {})),
        prompt=_dc(PromptConfig, raw.get("prompt", {})),
        decode=_dc(DecodeConfig, raw.get("decode", {})),
        control_points=_dc(ControlPointConfig, raw.get("control_points", {})),
        offline_mine=_dc(OfflineMineConfig, raw.get("offline_mine", {})),
        offline_select=_dc(OfflineSelectConfig, raw.get("offline_select", {})),
        offline_memory=_dc(OfflineMemoryConfig, raw.get("offline_memory", {})),
        online=_dc(OnlineConfig, raw.get("online", {})),
        eval=_dc(EvalConfig, raw.get("eval", {})),
        outputs=_dc(OutputConfig, raw.get("outputs", {})),
    )
    return cfg
