from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams  # type: ignore
from vllm.steer_vectors.request import SteerVectorRequest  # type: ignore

import easysteer.hidden_states as hs

from stir.config import STIRConfig
from stir.data.loaders import load_task_dataset
from stir.data.prompts import DEFAULT_SYSTEM_PROMPT, build_chat_messages, render_user_prompt
from stir.data.tasks import TaskExample
from stir.eval.extractors import extract_gold, extract_pred
from stir.eval.metrics import is_correct as is_correct_pred
from stir.online.memory import EpisodicMemory
from stir.utils.io import ensure_dir, write_jsonl

logger = logging.getLogger(__name__)


def _resolve_artifact_root(run_root: Path, artifact_run_dir: str | None) -> Path:
    """
    Resolve eval.artifact_run_dir to a concrete run directory.

    Supported values:
    - null / empty: use current run_root
    - existing directory: use it as-is
    - ".../<run_name>/latest" (case-insensitive): resolve via ".../<run_name>/LATEST"
    """
    if artifact_run_dir is None:
        return run_root
    s = str(artifact_run_dir).strip()
    if s == "":
        return run_root
    p = Path(s)
    if p.is_dir():
        return p
    if p.name.lower() == "latest":
        latest_path = p.parent / "LATEST"
        if latest_path.exists():
            rid = latest_path.read_text(encoding="utf-8").strip()
            if rid:
                cand = p.parent / rid
                if cand.is_dir():
                    return cand
    return p


def _build_query_by_layer(
    llm: LLM,
    prefix: str,
    *,
    needed_layers: set[int],
) -> dict[int, np.ndarray]:
    all_hidden_states, _ = hs.get_all_hidden_states_generate(
        llm,
        [prefix],
        max_tokens=1,
        split_by_samples=True,
    )
    if not all_hidden_states:
        return {}
    states = all_hidden_states[0]
    out: dict[int, np.ndarray] = {}
    for lid in needed_layers:
        if int(lid) < 0 or int(lid) >= len(states):
            continue
        try:
            h = states[int(lid)][-1].to(torch.float32).cpu().numpy().astype(np.float32)
        except Exception:
            continue
        out[int(lid)] = h
    return out


def _first_completion(ro: Any) -> tuple[str, list[int], float | None, str | None]:
    """
    Convert a vLLM RequestOutput to (text, token_ids, avg_logprob, finish_reason).
    """
    if not getattr(ro, "outputs", None):
        return "", [], None, "empty"

    co = ro.outputs[0]
    token_ids = list(getattr(co, "token_ids", []) or [])
    text = str(getattr(co, "text", ""))
    finish_reason = getattr(co, "finish_reason", None)

    cum_lp = getattr(co, "cumulative_logprob", None)
    if cum_lp is None:
        # Recover cumulative logprob from per-token logprobs when available.
        logprobs = getattr(co, "logprobs", None)
        if logprobs is not None and token_ids:
            s = 0.0
            ok = False
            for tid, d in zip(token_ids, logprobs):
                try:
                    lp_obj = d.get(tid)
                    if lp_obj is None:
                        continue
                    ok = True
                    s += float(lp_obj.logprob)
                except Exception:
                    continue
            if ok:
                cum_lp = float(s)

    avg_lp = None
    if cum_lp is not None and token_ids:
        avg_lp = float(cum_lp) / float(len(token_ids))

    return text, token_ids, avg_lp, finish_reason


@dataclass(frozen=True)
class Candidate:
    idx: int
    tool_name: str
    vector_path: str
    layer: int
    entry_type: str | None
    sim: float
    quality: float
    ahat: float  # similarity-weighted quality


@dataclass
class _BatchState:
    ex: TaskExample
    gold: str | None
    prefix: str
    assistant_text: str = ""
    committed_token_ids: list[int] = field(default_factory=list)
    committed_used: int = 0
    probe_used: int = 0
    final_finish_reason: str | None = None
    stopped_early: bool = False
    step_logs: list[dict[str, Any]] = field(default_factory=list)


def run_stir_dataset(
    cfg: STIRConfig,
    *,
    max_new_tokens: int | None = None,
    out_tag: str = "stir",
    examples: list[TaskExample] | None = None,
    llm: LLM | None = None,
) -> float:
    """
    Online STIR decoding over a dataset split (segmented decoding + episodic retrieval + probing).

    Writes:
      - outputs/<run>/eval/<out_tag>/per_example.jsonl
      - outputs/<run>/eval/<out_tag>/summary.jsonl
    """
    run_root = Path(cfg.outputs.run_dir)
    out_dir = ensure_dir(run_root / "eval" / out_tag)
    rng = np.random.default_rng(int(cfg.seed))
    noise_cache: dict[str, str] = {}
    noise_dir = ensure_dir(run_root / "eval" / "random_noise")

    def _random_vector_like(vector_path: str) -> str:
        cached = noise_cache.get(vector_path)
        if cached is not None:
            return cached
        import torch

        v = torch.load(vector_path, map_location="cpu", weights_only=False)
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v)
        gen = torch.Generator(device="cpu")
        seed = (abs(hash(vector_path)) + int(cfg.seed)) % (2**31)
        gen.manual_seed(int(seed))
        # torch.randn_like may not accept generator on some torch versions; use randn with explicit shape.
        noise = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=gen)
        noise_path = Path(noise_dir) / f"noise_{abs(hash(vector_path)) % (10**12)}.pt"
        torch.save(noise, noise_path)
        noise_cache[vector_path] = str(noise_path)
        return str(noise_path)

    artifact_root = _resolve_artifact_root(run_root, getattr(cfg.eval, "artifact_run_dir", None))
    memory = EpisodicMemory.load(artifact_root / "memory")

    # Pre-compute which layers are needed for retrieval.
    needed_layers: set[int] = set()
    for e in memory.entries:
        try:
            needed_layers.add(int(e.get("layer")))
        except Exception:
            continue

    variant = str(getattr(cfg.online, "variant", "stir")).strip().lower()
    if variant not in {"stir", "no_memory", "no_probing", "use_random_memory", "random_use_memory"}:
        raise ValueError(f"Unknown online.variant: {cfg.online.variant}")
    use_random_noise = bool(variant == "use_random_memory")
    use_random_choice = bool(variant == "random_use_memory")

    # Online hyperparameters
    k_retrieve = int(getattr(cfg.online, "k_retrieve", 16))
    L = int(getattr(cfg.online, "L", 2))
    probe_tokens = int(getattr(cfg.online, "probe_tokens", 8))
    beta = float(getattr(cfg.online, "beta", 1.0))
    rho = float(getattr(cfg.online, "rho", 0.1))
    k_scale = float(getattr(cfg.online, "k_scale", 1.0))
    tau_null = float(getattr(cfg.online, "tau_null", 0.0))
    min_sim = float(getattr(cfg.online, "min_sim", 0.0))
    min_entries = int(getattr(cfg.online, "min_entries", 1))
    min_tool_m_raw = getattr(cfg.online, "min_tool_m", None)
    max_tool_m_raw = getattr(cfg.online, "max_tool_m", None)
    min_tool_m = None if min_tool_m_raw is None else int(min_tool_m_raw)
    max_tool_m = None if max_tool_m_raw is None else int(max_tool_m_raw)

    # Models
    owns_llm = llm is None
    if llm is None:
        llm = LLM(
            model=str(cfg.model.name_or_path),
            tensor_parallel_size=int(cfg.model.tensor_parallel_size),
            enforce_eager=bool(cfg.model.enforce_eager),
            enable_chunked_prefill=bool(cfg.model.enable_chunked_prefill),
            dtype=str(cfg.model.dtype),
            seed=int(cfg.seed),
            gpu_memory_utilization=float(cfg.model.gpu_memory_utilization),
            max_model_len=int(cfg.model.max_model_len),
            max_num_seqs=int(cfg.model.max_num_seqs),
            enable_prefix_caching=False,
            enable_steer_vector=True,
            max_steer_vectors=max(8, int(cfg.model.max_num_seqs)),
        )
    tokenizer = llm.get_tokenizer()
    _steer_req_id = int(getattr(llm, "_stir_steer_req_id", 0))

    def _new_steer_request(*, name: str, vector_path: str, scale: float, layer: int) -> SteerVectorRequest:
        nonlocal _steer_req_id
        _steer_req_id += 1
        setattr(llm, "_stir_steer_req_id", int(_steer_req_id))
        uniq_name = f"{str(name)}__{int(_steer_req_id)}"
        return SteerVectorRequest(
            steer_vector_name=str(uniq_name),
            steer_vector_int_id=int(_steer_req_id),
            steer_vector_local_path=str(vector_path),
            scale=float(scale),
            target_layers=[int(layer)],
            prefill_trigger_positions=[-1],
            generate_trigger_tokens=None,
            algorithm="direct",
            normalize=False,
        )

    if examples is None:
        examples = load_task_dataset(
            task=cfg.task.dataset,
            split=cfg.task.eval_split,
            max_examples=cfg.task.max_eval_examples,
            seed=cfg.seed,
            data_root=cfg.task.data_root,
        )

    delimiter = str(getattr(cfg.control_points, "segment_delimiter", "\n\n"))
    M = int(cfg.control_points.M)
    T_max = int(max_new_tokens) if max_new_tokens is not None else int(cfg.decode.max_new_tokens)

    def _generate_one(
        prompt_text: str,
        *,
        max_tokens: int,
        stop: str | None = None,
        include_stop: bool = False,
        steer_req: SteerVectorRequest | None = None,
    ) -> tuple[str, list[int], float | None, str | None]:
        sampling_params = SamplingParams(
            temperature=float(cfg.decode.temperature),
            top_p=float(cfg.decode.top_p),
            max_tokens=int(max_tokens),
            logprobs=int(cfg.decode.logprobs) if cfg.decode.logprobs is not None else 1,
            stop=stop,
            include_stop_str_in_output=bool(include_stop),
        )
        ro = llm.generate(
            prompt_text,
            sampling_params=sampling_params,
            steer_vector_request=steer_req,
            use_tqdm=False,
        )[0]
        return _first_completion(ro)

    def _build_prompt_for_ex(ex: TaskExample) -> str:
        user_prompt = render_user_prompt(ex, cfg.prompt.template)
        msgs = build_chat_messages(user_prompt, system_prompt=DEFAULT_SYSTEM_PROMPT)
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        parts = []
        for m_ in msgs:
            parts.append(f"{m_['role'].upper()}:\n{m_['content']}\n")
        parts.append("ASSISTANT:\n")
        return "\n".join(parts)

    def _steer_slots() -> int | None:
        try:
            eng = getattr(llm, "llm_engine", None)
            vcfg = getattr(eng, "vllm_config", None) if eng is not None else None
            sv_cfg = getattr(vcfg, "steer_vector_config", None) if vcfg is not None else None
            if sv_cfg is None:
                return None
            return int(getattr(sv_cfg, "max_steer_vectors"))
        except Exception:
            return None

    def _run_batched(exs: list[TaskExample], batch_size: int) -> tuple[list[dict[str, Any]], int]:
        out_rows: list[dict[str, Any]] = []
        correct_total = 0

        bs = max(1, int(batch_size))
        steer_slots = _steer_slots()
        if steer_slots is None or int(steer_slots) <= 0:
            steer_slots = max(1, int(cfg.model.max_num_seqs))

        def _commit_params(max_tokens: int, *, stop: str | None, include_stop: bool) -> SamplingParams:
            return SamplingParams(
                temperature=float(cfg.decode.temperature),
                top_p=float(cfg.decode.top_p),
                max_tokens=int(max_tokens),
                logprobs=int(cfg.decode.logprobs) if cfg.decode.logprobs is not None else 1,
                stop=stop,
                include_stop_str_in_output=bool(include_stop),
            )

        for batch_start in tqdm(range(0, len(exs), bs), desc=f"eval/{out_tag}/batch"):
            batch = exs[batch_start : batch_start + bs]
            if not batch:
                continue

            states: list[_BatchState] = []
            for ex in batch:
                states.append(_BatchState(ex=ex, gold=extract_gold(ex.task, ex.answer), prefix=_build_prompt_for_ex(ex)))

            for m in range(1, M + 1):
                active: list[int] = []
                for i, st in enumerate(states):
                    if st.stopped_early:
                        continue
                    if int(T_max) - int(st.committed_used) <= 0:
                        continue
                    active.append(int(i))
                if not active:
                    break

                mem_m = int(m) - 1

                step_ctx: dict[int, dict[str, Any]] = {}
                chosen_req_by_i: dict[int, SteerVectorRequest | None] = {int(i): None for i in active}

                tool_idxs: list[int] = []
                for i in active:
                    st = states[int(i)]
                    remaining_before = int(T_max) - int(st.committed_used)
                    ctx: dict[str, Any] = {
                        "m": int(m),
                        "mem_m": int(mem_m),
                        "remaining_before": int(remaining_before),
                    }
                    step_ctx[int(i)] = ctx

                    if int(mem_m) <= 0:
                        ctx.update(
                            {
                                "spent_probe": 0,
                                "probe_tokens_per_cand": 0,
                                "chosen": "null",
                                "chosen_score": 0.0,
                                "chosen_scale": 0.0,
                                "candidates": [{"name": "null", "score": 0.0}],
                                "note": "mem_m=0 (no offline control point)",
                            }
                        )
                        continue

                    if min_tool_m is not None and int(mem_m) < int(min_tool_m):
                        note = f"min_tool_m={int(min_tool_m)} (mem_m={int(mem_m)})"
                        ctx.update(
                            {
                                "spent_probe": 0,
                                "probe_tokens_per_cand": 0,
                                "chosen": "null",
                                "chosen_score": 0.0,
                                "chosen_scale": 0.0,
                                "candidates": [{"name": "null", "score": 0.0}],
                                "note": note,
                            }
                        )
                        continue

                    if max_tool_m is not None and int(mem_m) > int(max_tool_m):
                        note = f"max_tool_m={int(max_tool_m)} (mem_m={int(mem_m)})"
                        ctx.update(
                            {
                                "spent_probe": 0,
                                "probe_tokens_per_cand": 0,
                                "chosen": "null",
                                "chosen_score": 0.0,
                                "chosen_scale": 0.0,
                                "candidates": [{"name": "null", "score": 0.0}],
                                "note": note,
                            }
                        )
                        continue

                    if variant == "no_memory":
                        ctx.update(
                            {
                                "spent_probe": 0,
                                "probe_tokens_per_cand": 0,
                                "chosen": "null",
                                "chosen_score": 0.0,
                                "chosen_scale": 0.0,
                                "candidates": [{"name": "null", "score": 0.0}],
                                "note": "variant=no_memory",
                            }
                        )
                        continue

                    tool_idxs.append(int(i))

                if tool_idxs:
                    # Hidden-state capture for all tool-eligible prefixes in this step.
                    queries_by_i: dict[int, dict[int, np.ndarray]] = {}
                    hs_bs = max(1, int(cfg.model.max_num_seqs))
                    for start in range(0, len(tool_idxs), hs_bs):
                        chunk_ids = tool_idxs[start : start + hs_bs]
                        chunk_prompts = [states[int(i)].prefix for i in chunk_ids]
                        try:
                            chunk_hidden_states, _ = hs.get_all_hidden_states_generate(
                                llm,
                                chunk_prompts,
                                max_tokens=1,
                                split_by_samples=True,
                            )
                        except Exception:
                            chunk_hidden_states = []

                        for off, i in enumerate(chunk_ids):
                            q: dict[int, np.ndarray] = {}
                            if off < len(chunk_hidden_states):
                                sample_layers = chunk_hidden_states[off]
                                for lid in needed_layers:
                                    ilid = int(lid)
                                    if ilid < 0 or ilid >= len(sample_layers):
                                        continue
                                    try:
                                        h = sample_layers[ilid][-1].to(torch.float32).cpu().numpy().astype(np.float32)
                                    except Exception:
                                        continue
                                    q[int(ilid)] = h
                            queries_by_i[int(i)] = q

                    per_probe = max(0, int(probe_tokens))
                    tool_steps: dict[int, dict[str, Any]] = {}
                    probe_needed: list[int] = []

                    for i in tool_idxs:
                        q = queries_by_i.get(int(i), {})
                        idx, sims, mem_info = memory.topk_debug(
                            q,
                            k=int(k_retrieve),
                            control_point_m=int(mem_m),
                            min_sim=float(min_sim) if min_sim is not None else None,
                            min_entries=int(min_entries),
                        )

                        null_ahat = 0.0
                        cands: list[Candidate] = []
                        for ii, sim in zip(idx.tolist(), sims.tolist()):
                            e = memory.entries[int(ii)]
                            tool_name = str(e.get("tool_name", f"mem_{int(ii)}"))
                            entry_type = str(e.get("entry_type")) if e.get("entry_type") is not None else None
                            layer = int(e.get("layer"))
                            quality = float(e.get("quality", 0.0))
                            ahat = float(sim) * float(quality)
                            vp_raw = e.get("vector_path", None)
                            vp = None if vp_raw is None or str(vp_raw).strip() == "" else str(vp_raw)
                            if vp is None:
                                null_ahat = max(float(null_ahat), float(ahat))
                                continue
                            cands.append(
                                Candidate(
                                    idx=int(ii),
                                    tool_name=tool_name,
                                    vector_path=str(vp),
                                    layer=int(layer),
                                    entry_type=entry_type,
                                    sim=float(sim),
                                    quality=float(quality),
                                    ahat=float(ahat),
                                )
                            )

                        cands.sort(key=lambda x: (float(x.ahat), float(x.sim), float(x.quality)), reverse=True)
                        cands = cands[: max(0, int(L))]

                        tool_steps[int(i)] = {
                            "cands": cands,
                            "null_ahat": float(null_ahat),
                            "mem_info": mem_info,
                            "per_probe": int(per_probe),
                            "spent_probe": 0,
                            "did_probe": False,
                            "best_name": "null",
                            "best_score": float(beta) * float(null_ahat),
                            "chosen_scale": 0.0,
                            "chosen_req": None,
                            "scored": [],
                        }

                        if variant != "no_probing" and int(per_probe) > 0 and cands:
                            probe_needed.append(int(i))
                            continue

                        # No probing: choose by beta*Ahat only.
                        scored: list[dict[str, Any]] = []
                        best_score = float(beta) * float(null_ahat)
                        best_name = "null"
                        chosen_entry: Candidate | None = None
                        scored.append({"name": "null", "score": float(best_score), "ahat": float(null_ahat), "sim": None})
                        for c in cands:
                            s = float(beta) * float(c.ahat)
                            scored.append({"name": c.tool_name, "score": float(s), "ahat": float(c.ahat), "sim": float(c.sim)})
                            if float(s) > float(best_score):
                                best_name = str(c.tool_name)
                                best_score = float(s)
                                chosen_entry = c

                        if use_random_choice and cands:
                            rand_c = cands[int(rng.integers(len(cands)))]
                            best_name = str(rand_c.tool_name)
                            best_score = float(beta) * float(rand_c.ahat)
                            chosen_entry = rand_c

                        if chosen_entry is not None and float(best_score) < float(tau_null):
                            best_name = "null"
                            best_score = float(beta) * float(null_ahat)
                            chosen_entry = None

                        chosen_req: SteerVectorRequest | None = None
                        chosen_scale = 0.0
                        if chosen_entry is not None:
                            chosen_scale = float(k_scale) * float(best_score)
                            if float(chosen_scale) > 0.0:
                                vp_for_req = str(chosen_entry.vector_path)
                                if use_random_noise:
                                    vp_for_req = _random_vector_like(vp_for_req)
                                chosen_req = _new_steer_request(
                                    name=str(chosen_entry.tool_name),
                                    vector_path=str(vp_for_req),
                                    scale=float(chosen_scale),
                                    layer=int(chosen_entry.layer),
                                )
                            else:
                                best_name = "null"
                                chosen_entry = None
                                chosen_scale = 0.0

                        tool_steps[int(i)].update(
                            {
                                "did_probe": False,
                                "spent_probe": 0,
                                "best_name": str(best_name),
                                "best_score": float(best_score),
                                "chosen_scale": float(chosen_scale),
                                "chosen_req": chosen_req,
                                "scored": scored,
                            }
                        )

                    if probe_needed:
                        probe_params = SamplingParams(
                            temperature=float(cfg.decode.temperature),
                            top_p=float(cfg.decode.top_p),
                            max_tokens=int(per_probe),
                            logprobs=int(cfg.decode.logprobs) if cfg.decode.logprobs is not None else 1,
                            stop=delimiter,
                            include_stop_str_in_output=True,
                        )

                        # Null probes (unsteered) for all probe-needed examples.
                        null_prompts = [states[int(i)].prefix for i in probe_needed]
                        null_outs = llm.generate(null_prompts, sampling_params=probe_params, use_tqdm=False)
                        null_comps = [_first_completion(ro) for ro in null_outs]

                        probe_tok_counts: dict[int, int] = {int(i): 0 for i in probe_needed}
                        null_conf: dict[int, float] = {}
                        for i, comp in zip(probe_needed, null_comps):
                            _, toks, avg_lp, _ = comp
                            probe_tok_counts[int(i)] += int(len(toks))
                            null_conf[int(i)] = float(avg_lp) if avg_lp is not None else 0.0

                        # Candidate probes (steered), chunked to respect steer slots.
                        cand_items: list[tuple[int, int, SteerVectorRequest]] = []
                        for i in probe_needed:
                            cands = tool_steps[int(i)]["cands"]
                            for j, c in enumerate(cands):
                                cand_items.append(
                                    (
                                        int(i),
                                        int(j),
                                        _new_steer_request(
                                            name=str(c.tool_name),
                                            vector_path=str(c.vector_path),
                                            scale=1.0,
                                            layer=int(c.layer),
                                        ),
                                    )
                                )

                        cand_conf: dict[int, dict[int, float]] = {int(i): {} for i in probe_needed}
                        if cand_items:
                            chunk_sz = max(1, int(steer_slots))
                            for start in range(0, len(cand_items), chunk_sz):
                                chunk = cand_items[start : start + chunk_sz]
                                chunk_prompts = [states[int(i)].prefix for i, _, _ in chunk]
                                chunk_steers = [sv for _, _, sv in chunk]
                                chunk_outs = llm.generate(
                                    chunk_prompts,
                                    sampling_params=probe_params,
                                    steer_vector_request=chunk_steers,
                                    use_tqdm=False,
                                )
                                chunk_comps = [_first_completion(ro) for ro in chunk_outs]
                                for (i, j, _), comp in zip(chunk, chunk_comps):
                                    _, toks, avg_lp, _ = comp
                                    probe_tok_counts[int(i)] += int(len(toks))
                                    cand_conf[int(i)][int(j)] = float(avg_lp) if avg_lp is not None else 0.0

                        # Finalize probing scores for each example.
                        for i in probe_needed:
                            st = states[int(i)]
                            spent_probe = int(probe_tok_counts.get(int(i), 0))
                            st.probe_used += int(spent_probe)

                            meta = tool_steps[int(i)]
                            cands = meta["cands"]
                            null_ahat = float(meta["null_ahat"])
                            mem_info = meta["mem_info"]

                            null_c = float(null_conf.get(int(i), 0.0))

                            scored: list[dict[str, Any]] = []
                            best_score = float(beta) * float(null_ahat)
                            best_name = "null"
                            chosen_entry: Candidate | None = None
                            scored.append(
                                {
                                    "name": "null",
                                    "score": float(best_score),
                                    "ahat": float(null_ahat),
                                    "sim": None,
                                    "conf_gain": 0.0,
                                }
                            )

                            for j, c in enumerate(cands):
                                conf = float(cand_conf.get(int(i), {}).get(int(j), 0.0))
                                conf_gain = float(conf - null_c)
                                score = float(beta) * float(c.ahat) + float(rho) * float(conf_gain)
                                scored.append(
                                    {
                                        "name": str(c.tool_name),
                                        "score": float(score),
                                        "ahat": float(c.ahat),
                                        "sim": float(c.sim),
                                        "conf_gain": float(conf_gain),
                                    }
                                )
                                if float(score) > float(best_score):
                                    best_score = float(score)
                                    best_name = str(c.tool_name)
                                    chosen_entry = c

                            if use_random_choice and cands:
                                rand_j = int(rng.integers(len(cands)))
                                rand_c = cands[rand_j]
                                conf = float(cand_conf.get(int(i), {}).get(int(rand_j), 0.0))
                                conf_gain = float(conf - null_c)
                                best_score = float(beta) * float(rand_c.ahat) + float(rho) * float(conf_gain)
                                best_name = str(rand_c.tool_name)
                                chosen_entry = rand_c

                            eps = 1e-12
                            if chosen_entry is not None and abs(float(best_score) - float(beta) * float(null_ahat)) <= eps:
                                best_name = "null"
                                best_score = float(beta) * float(null_ahat)
                                chosen_entry = None

                            if chosen_entry is not None and float(best_score) < float(tau_null):
                                best_name = "null"
                                best_score = float(beta) * float(null_ahat)
                                chosen_entry = None

                            chosen_req: SteerVectorRequest | None = None
                            chosen_scale = 0.0
                            if chosen_entry is not None:
                                chosen_scale = float(k_scale) * float(best_score)
                                if float(chosen_scale) > 0.0:
                                    vp_for_req = str(chosen_entry.vector_path)
                                    if use_random_noise:
                                        vp_for_req = _random_vector_like(vp_for_req)
                                    chosen_req = _new_steer_request(
                                        name=str(chosen_entry.tool_name),
                                        vector_path=str(vp_for_req),
                                        scale=float(chosen_scale),
                                        layer=int(chosen_entry.layer),
                                    )
                                else:
                                    best_name = "null"
                                    chosen_entry = None
                                    chosen_scale = 0.0

                            tool_steps[int(i)].update(
                                {
                                    "did_probe": True,
                                    "spent_probe": int(spent_probe),
                                    "best_name": str(best_name),
                                    "best_score": float(best_score),
                                    "chosen_scale": float(chosen_scale),
                                    "chosen_req": chosen_req,
                                    "scored": scored,
                                    "mem_info": mem_info,
                                }
                            )

                    # Fill per-example step context + chosen steer request.
                    for i in tool_idxs:
                        meta = tool_steps.get(int(i), {})
                        null_ahat = float(meta.get("null_ahat", 0.0))
                        mem_info = meta.get("mem_info", None)
                        did_probe = bool(meta.get("did_probe", False))
                        spent_probe = int(meta.get("spent_probe", 0))
                        best_name = str(meta.get("best_name", "null"))
                        best_score = float(meta.get("best_score", float(beta) * float(null_ahat)))
                        chosen_scale = float(meta.get("chosen_scale", 0.0))
                        scored = meta.get("scored") or [{"name": "null", "score": 0.0}]

                        ctx = step_ctx[int(i)]
                        ctx.update(
                            {
                                "probe_tokens_per_cand": int(per_probe) if did_probe else 0,
                                "spent_probe": int(spent_probe),
                                "did_probe": bool(did_probe),
                                "chosen": str(best_name),
                                "chosen_score": float(best_score),
                                "chosen_scale": float(chosen_scale),
                                "null_ahat": float(null_ahat),
                                "candidates": scored,
                                "mem_reason": str(mem_info.get("reason")) if isinstance(mem_info, dict) else None,
                                "mem_n_entries": int(mem_info.get("n_entries_m"))
                                if isinstance(mem_info, dict) and mem_info.get("n_entries_m") is not None
                                else None,
                                "mem_top_sim": float(mem_info.get("top_sim"))
                                if isinstance(mem_info, dict) and mem_info.get("top_sim") is not None
                                else None,
                                "note": f"variant={variant}",
                            }
                        )
                        chosen_req_by_i[int(i)] = meta.get("chosen_req", None)

                # Commit: generate 1 segment for each active example.
                def _apply_commit(i: int, comp: tuple[str, list[int], float | None, str | None]) -> None:
                    seg_text, seg_ids, _, finish_reason = comp
                    spent_commit = int(len(seg_ids))
                    st = states[int(i)]
                    st.committed_used += int(spent_commit)
                    st.prefix = st.prefix + seg_text
                    st.assistant_text += seg_text
                    st.committed_token_ids.extend(seg_ids)
                    st.final_finish_reason = finish_reason
                    ctx = step_ctx.get(int(i), {"m": int(m), "mem_m": int(mem_m), "remaining_before": 0})
                    ctx["spent_commit"] = int(spent_commit)
                    ctx["spent_total"] = int(int(ctx.get("spent_probe", 0)) + int(spent_commit))
                    st.step_logs.append(ctx)
                    if not bool(st.prefix.endswith(delimiter)):
                        st.stopped_early = True

                unsteered = [i for i in active if chosen_req_by_i.get(int(i)) is None]
                steered = [i for i in active if chosen_req_by_i.get(int(i)) is not None]

                if unsteered:
                    prompts = [states[int(i)].prefix for i in unsteered]
                    params = [
                        _commit_params(int(T_max) - int(states[int(i)].committed_used), stop=delimiter, include_stop=True)
                        for i in unsteered
                    ]
                    outs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
                    comps = [_first_completion(ro) for ro in outs]
                    for i, comp in zip(unsteered, comps):
                        _apply_commit(int(i), comp)

                if steered:
                    chunk_sz = max(1, int(steer_slots))
                    for start in range(0, len(steered), chunk_sz):
                        chunk = steered[start : start + chunk_sz]
                        prompts = [states[int(i)].prefix for i in chunk]
                        params = [
                            _commit_params(int(T_max) - int(states[int(i)].committed_used), stop=delimiter, include_stop=True)
                            for i in chunk
                        ]
                        steers = [chosen_req_by_i[int(i)] for i in chunk]
                        outs = llm.generate(
                            prompts,
                            sampling_params=params,
                            steer_vector_request=steers,
                            use_tqdm=False,
                        )
                        comps = [_first_completion(ro) for ro in outs]
                        for i, comp in zip(chunk, comps):
                            _apply_commit(int(i), comp)

            # Tail: finish unsteered with any remaining committed budget.
            tail_ids: list[int] = []
            for i, st in enumerate(states):
                if st.stopped_early:
                    continue
                if int(T_max) - int(st.committed_used) <= 0:
                    continue
                tail_ids.append(int(i))

            if tail_ids:
                tail_prompts = [states[int(i)].prefix for i in tail_ids]
                tail_params = [
                    _commit_params(int(T_max) - int(states[int(i)].committed_used), stop=None, include_stop=False)
                    for i in tail_ids
                ]
                tail_outs = llm.generate(tail_prompts, sampling_params=tail_params, use_tqdm=False)
                tail_comps = [_first_completion(ro) for ro in tail_outs]
                for i, comp in zip(tail_ids, tail_comps):
                    tail_text, tail_tok_ids, _, finish_reason = comp
                    spent_tail = int(len(tail_tok_ids))
                    st = states[int(i)]
                    st.committed_used += int(spent_tail)
                    st.prefix = st.prefix + tail_text
                    st.assistant_text += tail_text
                    st.committed_token_ids.extend(tail_tok_ids)
                    st.final_finish_reason = finish_reason
                    st.step_logs.append(
                        {
                            "m": int(M) + 1,
                            "phase": "tail",
                            "remaining_before": int(T_max) - int(st.committed_used) + int(spent_tail),
                            "probe_tokens_per_cand": 0,
                            "spent_probe": 0,
                            "spent_commit": int(spent_tail),
                            "spent_total": int(spent_tail),
                            "chosen": "null",
                            "chosen_score": 0.0,
                            "chosen_scale": 0.0,
                            "candidates": [{"name": "null", "score": 0.0}],
                        }
                    )

            for st in states:
                pred = extract_pred(st.ex.task, st.assistant_text)
                correct_flag = bool(is_correct_pred(st.ex.task, pred, st.gold))
                correct_total += int(correct_flag)
                out_rows.append(
                    {
                        "task": st.ex.task,
                        "example_id": st.ex.id,
                        "pred": pred,
                        "gold": st.gold,
                        "correct": bool(correct_flag),
                        "tokens_used": int(len(st.committed_token_ids)),
                        "budget_used": int(len(st.committed_token_ids)) + int(st.probe_used),
                        "probe_tokens_used": int(st.probe_used),
                        "finish_reason": st.final_finish_reason,
                        "text": st.assistant_text,
                        "steps": st.step_logs,
                    }
                )

        return out_rows, int(correct_total)

    batch_size_examples = int(getattr(cfg.online, "batch_size_examples", 1) or 1)
    if int(batch_size_examples) <= 0:
        batch_size_examples = int(len(examples))
    if int(batch_size_examples) > 1:
        rows_b, correct_b = _run_batched(examples, int(batch_size_examples))
        write_jsonl(Path(out_dir) / "per_example.jsonl", rows_b)
        acc_b = float(correct_b) / float(max(1, len(rows_b)))
        write_jsonl(Path(out_dir) / "summary.jsonl", [{"n": len(rows_b), "acc": float(acc_b), "T_max": int(T_max)}])
        logger.info("%s eval done. n=%d acc=%.4f", out_tag, len(rows_b), float(acc_b))
        if owns_llm:
            try:
                eng = getattr(llm, "llm_engine", None)
                engine_core = getattr(eng, "engine_core", None) if eng is not None else None
                if engine_core is not None and hasattr(engine_core, "shutdown"):
                    engine_core.shutdown()
            except Exception:
                pass
        return float(acc_b)

    rows: list[dict[str, Any]] = []
    correct = 0

    for ex in tqdm(examples, desc=f"eval/{out_tag}"):
        gold = extract_gold(ex.task, ex.answer)
        user_prompt = render_user_prompt(ex, cfg.prompt.template)
        msgs = build_chat_messages(user_prompt, system_prompt=DEFAULT_SYSTEM_PROMPT)
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            parts = []
            for m_ in msgs:
                parts.append(f"{m_['role'].upper()}:\n{m_['content']}\n")
            parts.append("ASSISTANT:\n")
            prompt = "\n".join(parts)

        prefix = prompt
        assistant_text = ""
        committed_token_ids: list[int] = []
        committed_used = 0
        probe_used = 0
        final_finish_reason: str | None = None
        stopped_early = False

        step_logs: list[dict[str, Any]] = []

        for m in range(1, M + 1):
            remaining_before = int(T_max) - int(committed_used)
            if remaining_before <= 0:
                break

            # Offline artifacts are keyed by control_point_m = "state AFTER generating m segments".
            # Online chooses a correction BEFORE generating segment m, i.e. at the state after (m-1) segments.
            mem_m = int(m) - 1

            # No offline state exists for mem_m=0 (prompt-only). Behave like greedy for the first segment.
            if mem_m <= 0:
                seg_text, seg_ids, _, finish_reason = _generate_one(
                    prefix,
                    max_tokens=int(remaining_before),
                    stop=delimiter,
                    include_stop=True,
                    steer_req=None,
                )
                spent_commit = int(len(seg_ids))
                committed_used += int(spent_commit)
                prefix = prefix + seg_text
                assistant_text += seg_text
                committed_token_ids.extend(seg_ids)
                final_finish_reason = finish_reason
                step_logs.append(
                    {
                        "m": int(m),
                        "mem_m": int(mem_m),
                        "remaining_before": int(remaining_before),
                        "spent_probe": 0,
                        "spent_commit": int(spent_commit),
                        "spent_total": int(spent_commit),
                        "probe_tokens_per_cand": 0,
                        "chosen": "null",
                        "chosen_score": 0.0,
                        "chosen_scale": 0.0,
                        "candidates": [{"name": "null", "score": 0.0}],
                        "note": "mem_m=0 (no offline control point)",
                    }
                )
                if not bool(prefix.endswith(delimiter)):
                    stopped_early = True
                    break
                continue

            if min_tool_m is not None and int(mem_m) < int(min_tool_m):
                note = f"min_tool_m={int(min_tool_m)} (mem_m={int(mem_m)})"
                seg_text, seg_ids, _, finish_reason = _generate_one(
                    prefix,
                    max_tokens=int(remaining_before),
                    stop=delimiter,
                    include_stop=True,
                    steer_req=None,
                )
                spent_commit = int(len(seg_ids))
                committed_used += int(spent_commit)
                prefix = prefix + seg_text
                assistant_text += seg_text
                committed_token_ids.extend(seg_ids)
                final_finish_reason = finish_reason
                step_logs.append(
                    {
                        "m": int(m),
                        "mem_m": int(mem_m),
                        "remaining_before": int(remaining_before),
                        "spent_probe": 0,
                        "spent_commit": int(spent_commit),
                        "spent_total": int(spent_commit),
                        "probe_tokens_per_cand": 0,
                        "chosen": "null",
                        "chosen_score": 0.0,
                        "chosen_scale": 0.0,
                        "candidates": [{"name": "null", "score": 0.0}],
                        "note": note,
                    }
                )
                if not bool(prefix.endswith(delimiter)):
                    stopped_early = True
                    break
                continue

            if max_tool_m is not None and int(mem_m) > int(max_tool_m):
                note = f"max_tool_m={int(max_tool_m)} (mem_m={int(mem_m)})"
                seg_text, seg_ids, _, finish_reason = _generate_one(
                    prefix,
                    max_tokens=int(remaining_before),
                    stop=delimiter,
                    include_stop=True,
                    steer_req=None,
                )
                spent_commit = int(len(seg_ids))
                committed_used += int(spent_commit)
                prefix = prefix + seg_text
                assistant_text += seg_text
                committed_token_ids.extend(seg_ids)
                final_finish_reason = finish_reason
                step_logs.append(
                    {
                        "m": int(m),
                        "mem_m": int(mem_m),
                        "remaining_before": int(remaining_before),
                        "spent_probe": 0,
                        "spent_commit": int(spent_commit),
                        "spent_total": int(spent_commit),
                        "probe_tokens_per_cand": 0,
                        "chosen": "null",
                        "chosen_score": 0.0,
                        "chosen_scale": 0.0,
                        "candidates": [{"name": "null", "score": 0.0}],
                        "note": note,
                    }
                )
                if not bool(prefix.endswith(delimiter)):
                    stopped_early = True
                    break
                continue

            if variant == "no_memory":
                seg_text, seg_ids, _, finish_reason = _generate_one(
                    prefix,
                    max_tokens=int(remaining_before),
                    stop=delimiter,
                    include_stop=True,
                    steer_req=None,
                )
                spent_commit = int(len(seg_ids))
                committed_used += int(spent_commit)
                prefix = prefix + seg_text
                assistant_text += seg_text
                committed_token_ids.extend(seg_ids)
                final_finish_reason = finish_reason
                step_logs.append(
                    {
                        "m": int(m),
                        "mem_m": int(mem_m),
                        "remaining_before": int(remaining_before),
                        "spent_probe": 0,
                        "spent_commit": int(spent_commit),
                        "spent_total": int(spent_commit),
                        "probe_tokens_per_cand": 0,
                        "chosen": "null",
                        "chosen_score": 0.0,
                        "chosen_scale": 0.0,
                        "candidates": [{"name": "null", "score": 0.0}],
                        "note": "variant=no_memory",
                    }
                )
                if not bool(prefix.endswith(delimiter)):
                    stopped_early = True
                    break
                continue

            # Build layer->query vectors at the current prefix (last-token hidden state).
            query_by_layer = _build_query_by_layer(llm, prefix, needed_layers=needed_layers)
            idx, sims, mem_info = memory.topk_debug(
                query_by_layer,
                k=int(k_retrieve),
                control_point_m=int(mem_m),
                min_sim=float(min_sim) if min_sim is not None else None,
                min_entries=int(min_entries),
            )

            # Candidate pool from top-k retrieval
            null_ahat = 0.0
            cands: list[Candidate] = []
            for ii, sim in zip(idx.tolist(), sims.tolist()):
                e = memory.entries[int(ii)]
                tool_name = str(e.get("tool_name", f"mem_{int(ii)}"))
                entry_type = str(e.get("entry_type")) if e.get("entry_type") is not None else None
                layer = int(e.get("layer"))
                quality = float(e.get("quality", 0.0))
                ahat = float(sim) * float(quality)
                vp_raw = e.get("vector_path", None)
                vp = None if vp_raw is None or str(vp_raw).strip() == "" else str(vp_raw)
                if vp is None:
                    null_ahat = max(float(null_ahat), float(ahat))
                    continue
                cands.append(
                    Candidate(
                        idx=int(ii),
                        tool_name=tool_name,
                        vector_path=vp,
                        layer=int(layer),
                        entry_type=entry_type,
                        sim=float(sim),
                        quality=float(quality),
                        ahat=float(ahat),
                    )
                )

            # Keep top-L candidates by Ahat (similarity-weighted quality)
            cands.sort(key=lambda x: (float(x.ahat), float(x.sim), float(x.quality)), reverse=True)
            cands = cands[: max(0, int(L))]

            # Fixed per-candidate probe length; probing tokens do NOT reduce the committed max_new_tokens budget.
            per_probe = max(0, int(probe_tokens))

            scored: list[dict[str, Any]] = []
            best_name = "null"
            best_score = float(beta) * float(null_ahat)
            chosen_entry: Candidate | None = None
            spent_probe = 0
            did_probe = False

            if variant == "no_probing" or per_probe <= 0 or not cands:
                # No probing: choose by beta*Ahat only (null has support from 'right'/null-like entries).
                scored.append({"name": "null", "score": float(best_score), "ahat": float(null_ahat), "sim": None})
                for c in cands:
                    s = float(beta) * float(c.ahat)
                    scored.append({"name": c.tool_name, "score": float(s), "ahat": float(c.ahat), "sim": float(c.sim)})
                    if float(s) > float(best_score):
                        best_name = c.tool_name
                        best_score = float(s)
                        chosen_entry = c
                if use_random_choice and cands:
                    rand_c = cands[int(rng.integers(len(cands)))]
                    best_name = str(rand_c.tool_name)
                    best_score = float(beta) * float(rand_c.ahat)
                    chosen_entry = rand_c
            else:
                sampling_params = SamplingParams(
                    temperature=float(cfg.decode.temperature),
                    top_p=float(cfg.decode.top_p),
                    max_tokens=int(per_probe),
                    logprobs=int(cfg.decode.logprobs) if cfg.decode.logprobs is not None else 1,
                    stop=delimiter,
                    include_stop_str_in_output=True,
                )

                steer_reqs = [
                    _new_steer_request(
                        name=str(c.tool_name),
                        vector_path=str(c.vector_path),
                        scale=1.0,  # probe at unit scale; final scale uses k_scale*Score
                        layer=int(c.layer),
                    )
                    for c in cands
                ]
                ro_steered = llm.generate(
                    [prefix] * len(cands),
                    sampling_params=sampling_params,
                    steer_vector_request=steer_reqs,
                    use_tqdm=False,
                )
                steered_outs = [_first_completion(ro) for ro in ro_steered]

                # Null probe (unsteered)
                ro_null = llm.generate(prefix, sampling_params=sampling_params, use_tqdm=False)[0]
                null_out = _first_completion(ro_null)

                probe_outs = steered_outs + [null_out]

                spent_probe = int(sum(len(toks) for _, toks, _, _ in probe_outs))
                probe_used += int(spent_probe)
                did_probe = True

                null_conf = float(probe_outs[-1][2]) if probe_outs[-1][2] is not None else 0.0

                # Null candidate score includes support from retrieved null-like entries
                best_score = float(beta) * float(null_ahat)
                best_name = "null"
                chosen_entry = None
                scored.append(
                    {
                        "name": "null",
                        "score": float(best_score),
                        "ahat": float(null_ahat),
                        "sim": None,
                        "conf_gain": 0.0,
                    }
                )

                for c, (_, _, avg_lp, _) in zip(cands, probe_outs[:-1]):
                    conf = float(avg_lp) if avg_lp is not None else 0.0
                    conf_gain = float(conf - null_conf)
                    score = float(beta) * float(c.ahat) + float(rho) * float(conf_gain)
                    scored.append(
                        {
                            "name": str(c.tool_name),
                            "score": float(score),
                            "ahat": float(c.ahat),
                            "sim": float(c.sim),
                            "conf_gain": float(conf_gain),
                        }
                    )
                    if float(score) > float(best_score):
                        best_score = float(score)
                        best_name = str(c.tool_name)
                        chosen_entry = c

                if use_random_choice and cands:
                    rand_j = int(rng.integers(len(cands)))
                    rand_c = cands[rand_j]
                    conf = float(probe_outs[rand_j][2]) if probe_outs[rand_j][2] is not None else 0.0
                    conf_gain = float(conf - null_conf)
                    best_score = float(beta) * float(rand_c.ahat) + float(rho) * float(conf_gain)
                    best_name = str(rand_c.tool_name)
                    chosen_entry = rand_c

                # Tie-break: prefer null when indistinguishable.
                eps = 1e-12
                if chosen_entry is not None and abs(float(best_score) - float(beta) * float(null_ahat)) <= eps:
                    best_name = "null"
                    best_score = float(beta) * float(null_ahat)
                    chosen_entry = None
            # End probing block.

            # Commit threshold: if the best non-null score is weak, force null.
            if chosen_entry is not None and float(best_score) < float(tau_null):
                best_name = "null"
                best_score = float(beta) * float(null_ahat)
                chosen_entry = None

            # Generate the committed segment once (probe tokens are discarded).
            remaining_commit = int(T_max) - int(committed_used)
            if int(remaining_commit) <= 0:
                break

            chosen_req: SteerVectorRequest | None = None
            chosen_scale = 0.0
            if chosen_entry is not None:
                chosen_scale = float(k_scale) * float(best_score)
                if float(chosen_scale) > 0.0:
                    vp_for_req = str(chosen_entry.vector_path)
                    if use_random_noise:
                        vp_for_req = _random_vector_like(vp_for_req)
                    chosen_req = _new_steer_request(
                        name=str(chosen_entry.tool_name),
                        vector_path=str(vp_for_req),
                        scale=float(chosen_scale),
                        layer=int(chosen_entry.layer),
                    )
                else:
                    chosen_entry = None
                    best_name = "null"

            seg_text, seg_ids, _, finish_reason = _generate_one(
                prefix,
                max_tokens=int(remaining_commit),
                stop=delimiter,
                include_stop=True,
                steer_req=chosen_req,
            )
            spent_commit = int(len(seg_ids))
            committed_used += int(spent_commit)
            prefix = prefix + seg_text
            assistant_text += seg_text
            committed_token_ids.extend(seg_ids)
            final_finish_reason = finish_reason

            step_logs.append(
                {
                    "m": int(m),
                    "mem_m": int(mem_m),
                    "remaining_before": int(remaining_before),
                    "probe_tokens_per_cand": int(per_probe) if did_probe else 0,
                    "spent_probe": int(spent_probe),
                    "spent_commit": int(spent_commit),
                    "spent_total": int(int(spent_probe) + int(spent_commit)),
                    "did_probe": bool(did_probe),
                    "chosen": str(best_name),
                    "chosen_score": float(best_score),
                    "chosen_scale": float(chosen_scale),
                    "null_ahat": float(null_ahat),
                    "candidates": scored,
                    "mem_reason": str(mem_info.get("reason")) if isinstance(mem_info, dict) else None,
                    "mem_n_entries": int(mem_info.get("n_entries_m")) if isinstance(mem_info, dict) and mem_info.get("n_entries_m") is not None else None,
                    "mem_top_sim": float(mem_info.get("top_sim")) if isinstance(mem_info, dict) and mem_info.get("top_sim") is not None else None,
                    "note": f"variant={variant}",
                }
            )

            if not bool(prefix.endswith(delimiter)):
                stopped_early = True
                break

        # Tail: finish unsteered with any remaining committed budget
        remaining_tail = int(T_max) - int(committed_used)
        if not stopped_early and int(remaining_tail) > 0:
            tail_text, tail_ids, _, finish_reason = _generate_one(
                prefix,
                max_tokens=int(remaining_tail),
                stop=None,
                include_stop=False,
                steer_req=None,
            )
            spent_tail = int(len(tail_ids))
            committed_used += int(spent_tail)
            prefix = prefix + tail_text
            assistant_text += tail_text
            committed_token_ids.extend(tail_ids)
            final_finish_reason = finish_reason
            step_logs.append(
                {
                    "m": int(M) + 1,
                    "phase": "tail",
                    "remaining_before": int(remaining_tail),
                    "probe_tokens_per_cand": 0,
                    "spent_probe": 0,
                    "spent_commit": int(spent_tail),
                    "spent_total": int(spent_tail),
                    "chosen": "null",
                    "chosen_score": 0.0,
                    "chosen_scale": 0.0,
                    "candidates": [{"name": "null", "score": 0.0}],
                }
            )

        pred = extract_pred(ex.task, assistant_text)
        correct_flag = bool(is_correct_pred(ex.task, pred, gold))
        correct += int(correct_flag)

        rows.append(
            {
                "task": ex.task,
                "example_id": ex.id,
                "pred": pred,
                "gold": gold,
                "correct": bool(correct_flag),
                "tokens_used": int(len(committed_token_ids)),
                # NOTE: "budget_used" is kept for backward-compatible analysis scripts.
                # It reflects TOTAL generated tokens including discarded probes.
                "budget_used": int(len(committed_token_ids)) + int(probe_used),
                "probe_tokens_used": int(probe_used),
                "finish_reason": final_finish_reason,
                "text": assistant_text,
                "steps": step_logs,
            }
        )

    write_jsonl(Path(out_dir) / "per_example.jsonl", rows)
    acc = correct / max(1, len(rows))
    write_jsonl(Path(out_dir) / "summary.jsonl", [{"n": len(rows), "acc": acc, "T_max": int(T_max)}])
    logger.info("%s eval done. n=%d acc=%.4f", out_tag, len(rows), acc)

    if owns_llm:
        try:
            eng = getattr(llm, "llm_engine", None)
            engine_core = getattr(eng, "engine_core", None) if eng is not None else None
            if engine_core is not None and hasattr(engine_core, "shutdown"):
                engine_core.shutdown()
        except Exception:
            pass
    return float(acc)
