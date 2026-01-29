from __future__ import annotations

import heapq
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams  # type: ignore

import easysteer.hidden_states as hs

from stir.config import STIRConfig
from stir.data.loaders import load_task_dataset
from stir.data.prompts import DEFAULT_SYSTEM_PROMPT, build_chat_messages, render_user_prompt
from stir.eval.extractors import extract_gold, extract_pred
from stir.eval.metrics import is_correct as is_correct_pred
from stir.utils.io import ensure_dir, write_jsonl, write_text

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    text: str
    token_ids: list[int]
    prompt_token_ids: list[int]
    reward: float
    pred: str | None
    gold: str | None
    correct: bool


def _length_regularized_reward(correct: bool, gen_tokens: int, T_max: int, eta: float) -> float:
    r = 1.0 if correct else 0.0
    return float(r) - float(eta) * (float(gen_tokens) / float(max(1, T_max)))


def _delimiter_end_offsets(text: str, delimiter: str) -> list[int]:
    """
    Return character offsets (end-exclusive) of each delimiter occurrence in text.
    """
    out: list[int] = []
    start = 0
    while True:
        idx = text.find(delimiter, start)
        if idx < 0:
            break
        end = idx + len(delimiter)
        out.append(int(end))
        start = end
    return out


def _select_pos_neg(rewards: list[float], k_pos: int, k_neg: int) -> tuple[list[int], list[int]]:
    idx = list(range(len(rewards)))
    idx_sorted = sorted(idx, key=lambda i: rewards[i], reverse=True)
    pos = idx_sorted[: max(1, k_pos)]
    neg = list(reversed(idx_sorted))[: max(1, k_neg)]
    return pos, neg


def _select_pos_neg_random(
    valid_idx: list[int],
    k_pos: int,
    k_neg: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """
    Randomly split valid rollout indices into pos/neg buckets (disjoint when possible).
    """
    if not valid_idx:
        return [], []
    order = list(valid_idx)
    rng.shuffle(order)
    pos_take = min(max(1, k_pos), len(order))
    pos = order[:pos_take]
    rem = order[pos_take:]
    neg_take = min(max(1, k_neg), len(rem)) if rem else min(max(1, k_neg), len(order))
    neg_source = rem if rem else order
    neg = neg_source[:neg_take]
    return pos, neg


def _resolve_candidate_layers(specs: list[Any], *, num_layers: int) -> list[int]:
    """
    Resolve layer specs into concrete 0-based layer indices.

    Supports:
      - int indices (e.g. 18)
      - negative int indices (Python-style; -1 means last)
      - ratio strings like "1/5"
      - ratio floats like 0.6 (or "0.6"), mapped to round(r*(num_layers-1))
        (ratios are expected in [0, 1]; 1.0 means the last layer)
    """
    if int(num_layers) <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")

    resolved: list[int] = []
    seen: set[int] = set()

    for raw in specs:
        if raw is None:
            continue
        if isinstance(raw, bool):
            raise TypeError(f"Invalid candidate layer spec (bool): {raw}")

        # 1) Absolute integer index.
        if isinstance(raw, (int, np.integer)):
            lid = int(raw)
            if lid < 0:
                lid = int(num_layers) + lid
            if lid < 0 or lid >= int(num_layers):
                raise ValueError(f"candidate layer index out of range: {raw} (num_layers={num_layers})")
            if lid not in seen:
                resolved.append(lid)
                seen.add(lid)
            continue

        # 2) Float: treat [0, 1] as ratio; otherwise require integer-like as absolute index.
        if isinstance(raw, (float, np.floating)):
            f = float(raw)
            if not math.isfinite(f):
                raise ValueError(f"Invalid candidate layer spec: {raw!r}")
            if 0.0 <= f <= 1.0:
                lid = int(round(f * float(int(num_layers) - 1)))
                lid = max(0, min(int(num_layers) - 1, lid))
                if lid not in seen:
                    resolved.append(lid)
                    seen.add(lid)
                continue
            if abs(f - round(f)) < 1e-9:
                lid = int(round(f))
                if lid < 0:
                    lid = int(num_layers) + lid
                if lid < 0 or lid >= int(num_layers):
                    raise ValueError(f"candidate layer index out of range: {raw!r} (num_layers={num_layers})")
                if lid not in seen:
                    resolved.append(lid)
                    seen.add(lid)
                continue
            raise ValueError(f"Invalid candidate layer spec (float): {raw!r}")

        s = str(raw).strip()
        if s == "":
            continue

        # 3) Fraction a/b.
        if "/" in s:
            parts = [p.strip() for p in s.split("/")]
            if len(parts) != 2 or parts[0] == "" or parts[1] == "":
                raise ValueError(f"Invalid candidate layer ratio spec: {s!r}")
            try:
                a = float(parts[0])
                b = float(parts[1])
            except Exception as e:
                raise ValueError(f"Invalid candidate layer ratio spec: {s!r}") from e
            if not math.isfinite(a) or not math.isfinite(b) or b == 0.0:
                raise ValueError(f"Invalid candidate layer ratio spec: {s!r}")
            ratio = float(a / b)
            if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
                raise ValueError(f"Invalid candidate layer ratio spec: {s!r}")
            lid = int(round(ratio * float(int(num_layers) - 1)))
            lid = max(0, min(int(num_layers) - 1, lid))
            if lid not in seen:
                resolved.append(lid)
                seen.add(lid)
            continue

        # 4) Numeric string:
        # - integers => absolute indices (0-based, supports negative)
        # - floats in [0, 1] => ratios
        try:
            lid = int(s)
        except Exception:
            lid = None
        if lid is not None:
            if lid < 0:
                lid = int(num_layers) + lid
            if lid < 0 or lid >= int(num_layers):
                raise ValueError(f"candidate layer index out of range: {s!r} (num_layers={num_layers})")
            if lid not in seen:
                resolved.append(lid)
                seen.add(lid)
            continue

        try:
            f = float(s)
        except Exception as e:
            raise ValueError(f"Invalid candidate layer spec: {s!r}") from e
        if not math.isfinite(f):
            raise ValueError(f"Invalid candidate layer spec: {s!r}")

        if 0.0 <= f <= 1.0:
            lid = int(round(f * float(int(num_layers) - 1)))
            lid = max(0, min(int(num_layers) - 1, lid))
            if lid not in seen:
                resolved.append(lid)
                seen.add(lid)
            continue
        if abs(f - round(f)) < 1e-9:
            lid = int(round(f))
            if lid < 0:
                lid = int(num_layers) + lid
            if lid < 0 or lid >= int(num_layers):
                raise ValueError(f"candidate layer index out of range: {s!r} (num_layers={num_layers})")
            if lid not in seen:
                resolved.append(lid)
                seen.add(lid)
            continue
        raise ValueError(f"Invalid candidate layer ratio: {s!r} (expected float in [0, 1])")

    if not resolved:
        raise ValueError(f"offline_mine.candidate_layers resolved to empty list from specs={specs!r}")
    return resolved


def mine_candidates(cfg: STIRConfig) -> None:
    """
    Stage I: Mine candidate memory entries from contrastive rollouts.

    Output:
      - outputs/<run>/mine/rollouts.jsonl
      - outputs/<run>/mine/candidates.jsonl
      - outputs/<run>/mine/keys/*.pt
      - outputs/<run>/mine/vectors/*.pt
    """
    out_root = Path(cfg.outputs.run_dir) / "mine"
    key_dir = ensure_dir(out_root / "keys")
    vec_dir = ensure_dir(out_root / "vectors")
    ensure_dir(out_root / "tmp")

    # 1) Load dataset
    examples = load_task_dataset(
        task=cfg.task.dataset,
        split=cfg.task.train_split,
        max_examples=cfg.task.max_train_examples,
        seed=cfg.seed,
        data_root=cfg.task.data_root,
    )
    logger.info(
        "Loaded %d examples for mining (%s/%s).",
        len(examples),
        cfg.task.dataset,
        cfg.task.train_split,
    )

    # 2) vLLM generator (EasySteer) + vLLM-V1 hidden-states capture.
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
        max_steer_vectors=8,
    )
    tokenizer = llm.get_tokenizer()
    rng = np.random.default_rng(int(cfg.seed))
    layer_specs = cfg.offline_mine.candidate_layers
    layers: list[int] | None = None
    if layer_specs is not None:
        logger.info("Mining candidate layer spec: %s", layer_specs)

    # 4) Rollout generation + memory-candidate mining with top-C heap
    # (quality, tie_breaker, meta, h_wrong, h_right, delta) - tie_breaker avoids dict comparison.
    heap: list[tuple[float, int, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
    tie = 0
    all_rollout_rows: list[dict[str, Any]] = []

    M = int(cfg.control_points.M)

    mine_method = str(getattr(cfg.offline_mine, "method", "contrastive")).strip().lower()
    if mine_method not in {"contrastive", "random"}:
        raise ValueError(f"Unknown offline_mine.method: {cfg.offline_mine.method}")
    random_mining = bool(mine_method == "random")

    require_contrast = False if random_mining else bool(getattr(cfg.offline_mine, "require_correct_and_incorrect", False))
    min_correct = int(getattr(cfg.offline_mine, "min_correct_rollouts", 1))
    min_incorrect = int(getattr(cfg.offline_mine, "min_incorrect_rollouts", 1))
    delimiter = str(getattr(cfg.control_points, "segment_delimiter", "\n\n"))
    k_pos = max(1, int(cfg.offline_mine.K_pos))
    k_neg = max(1, int(cfg.offline_mine.K_neg))
    pos_source = str(getattr(cfg.offline_mine, "pos_source", "rollout")).strip().lower()
    if pos_source not in {"rollout", "gold", "rollout_or_gold"}:
        raise ValueError(f"Unknown offline_mine.pos_source: {cfg.offline_mine.pos_source}")

    skipped_contrast_total = 0
    contrast_considered_total = 0

    K = int(cfg.offline_mine.K)
    batch_size_examples = max(1, int(getattr(cfg.offline_mine, "batch_size_examples", 1)))

    rollout_params = SamplingParams(
        n=int(K),
        temperature=float(cfg.offline_mine.temperature),
        top_p=float(cfg.offline_mine.top_p),
        max_tokens=int(cfg.offline_mine.max_new_tokens),
        logprobs=None,
    )
    hs_batch_size = max(1, int(cfg.model.max_num_seqs))

    pbar = tqdm(total=len(examples), desc="mine/examples")
    for batch_start in range(0, len(examples), batch_size_examples):
        batch = examples[batch_start : batch_start + batch_size_examples]
        if not batch:
            continue

        batch_prompts: list[str] = []
        for ex in batch:
            user_prompt = render_user_prompt(ex, cfg.prompt.template)
            msgs = build_chat_messages(user_prompt, system_prompt=DEFAULT_SYSTEM_PROMPT)
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            else:
                parts = []
                for m in msgs:
                    parts.append(f"{m['role'].upper()}:\n{m['content']}\n")
                parts.append("ASSISTANT:\n")
                prompt = "\n".join(parts)
            batch_prompts.append(prompt)

        # Generate K rollouts per example in one batched call (SamplingParams.n=K).
        batch_outs = llm.generate(batch_prompts, sampling_params=rollout_params, use_tqdm=False)

        # Collect all prefixes to capture hidden states for this batch.
        all_texts: list[str] = []
        text_meta: list[tuple[int, int, bool]] = []  # (batch_ex_idx, m, is_pos)
        expected_counts: dict[tuple[int, int], tuple[int, int]] = {}
        reward_meta: dict[tuple[int, int], dict[str, Any]] = {}

        # Per-example rollout processing (reward, correctness, segmentation).
        for ex_i, ex in enumerate(batch):
            prompt = batch_prompts[ex_i]
            ro = batch_outs[ex_i] if ex_i < len(batch_outs) else None

            prompt_token_ids: list[int] | None = getattr(ro, "prompt_token_ids", None) if ro is not None else None
            if not prompt_token_ids:
                try:
                    prompt_token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
                except Exception:
                    prompt_token_ids = list((tokenizer(prompt) or {}).get("input_ids", []))

            completions = list(getattr(ro, "outputs", []) or [])
            rollouts: list[Rollout] = []
            gold = extract_gold(ex.task, ex.answer)

            # Optional: use the dataset gold solution as a positive source (useful when sampled correct rollouts are rare).
            gold_gen_ids: list[int] | None = None
            gold_text_full: str | None = None
            gold_reward: float | None = None
            gold_boundaries: list[int] | None = None
            if pos_source in {"gold", "rollout_or_gold"}:
                try:
                    # Some tasks (e.g., AIME) have very short gold answers (a single number). When control points
                    # require multiple segments, we pad a "gold completion" with neutral filler so that gold_gen_ids
                    # can cover all control points. This enables contrast mining even when correct rollouts are rare.
                    gold_text = str(ex.answer)
                    if str(ex.task).lower() in {"aime_2024", "aime2024", "aime25", "aime_25", "aime_2025"}:
                        ans = str(ex.answer).strip() or "0"
                        prefix = "Let's think step by step.\n"
                        filler = "We compute carefully.\n"
                        final = f"Final answer: {ans}\n#### {ans}"
                        reps = 0
                        target_segments = int(M)
                        while True:
                            gold_text = prefix + (filler * reps) + final
                            full_ids = tokenizer(prompt + gold_text, return_tensors="pt")["input_ids"][0].tolist()
                            prompt_len = len(prompt_token_ids)
                            gold_gen_ids = full_ids[prompt_len:] if len(full_ids) > prompt_len else []
                            gold_text_full = str(gold_text)
                            gold_boundaries = _delimiter_end_offsets(gold_text, delimiter)
                            if len(gold_boundaries) >= int(target_segments) or reps >= 256:
                                break
                            reps += 1
                    else:
                        full_ids = tokenizer(prompt + gold_text, return_tensors="pt")["input_ids"][0].tolist()
                        prompt_len = len(prompt_token_ids)
                        gold_gen_ids = full_ids[prompt_len:] if len(full_ids) > prompt_len else []
                        gold_text_full = str(gold_text)
                        gold_boundaries = _delimiter_end_offsets(gold_text, delimiter)

                    gold_pred = extract_pred(ex.task, gold_text)
                    gold_correct = bool(is_correct_pred(ex.task, gold_pred, gold))
                    gold_reward = _length_regularized_reward(
                        correct=gold_correct,
                        gen_tokens=len(gold_gen_ids),
                        T_max=int(cfg.offline_mine.max_new_tokens),
                        eta=float(cfg.offline_mine.eta0),
                    )
                    all_rollout_rows.append(
                        {
                            "task": ex.task,
                            "example_id": ex.id,
                            "k": -1,
                            "source": "gold",
                            "reward": float(gold_reward),
                            "correct": bool(gold_correct),
                            "pred": gold_pred,
                            "gold": gold,
                            "gen_tokens": len(gold_gen_ids),
                            "text": gold_text,
                        }
                    )
                except Exception:
                    gold_gen_ids = None
                    gold_text_full = None
                    gold_reward = None
                    gold_boundaries = None

            for k, co in enumerate(completions):
                gen_ids = list(getattr(co, "token_ids", []) or [])
                gen_text = str(getattr(co, "text", ""))
                pred = extract_pred(ex.task, gen_text)
                correct = bool(is_correct_pred(ex.task, pred, gold))
                rew = _length_regularized_reward(
                    correct=correct,
                    gen_tokens=len(gen_ids),
                    T_max=int(cfg.offline_mine.max_new_tokens),
                    eta=float(cfg.offline_mine.eta0),
                )
                rollouts.append(
                    Rollout(
                        text=gen_text,
                        token_ids=gen_ids,
                        prompt_token_ids=prompt_token_ids,
                        reward=rew,
                        pred=pred,
                        gold=gold,
                        correct=correct,
                    )
                )
                all_rollout_rows.append(
                    {
                        "task": ex.task,
                        "example_id": ex.id,
                        "k": int(k),
                        "source": "rollout",
                        "reward": float(rew),
                        "correct": bool(correct),
                        "pred": pred,
                        "gold": gold,
                        "gen_tokens": int(len(gen_ids)),
                        "text": gen_text,
                    }
                )

            # For each control point m, select pos/neg prefixes for contrast.
            rollout_boundaries = [_delimiter_end_offsets(ro_.text, delimiter) for ro_ in rollouts]
            for m in range(1, M + 1):
                valid_idx = [i for i, b in enumerate(rollout_boundaries) if len(b) >= m]
                if not valid_idx:
                    continue

                if random_mining:
                    pos_idx, neg_idx = _select_pos_neg_random(valid_idx, k_pos, k_neg, rng)
                    pos_texts = [prompt + rollouts[i].text[: rollout_boundaries[i][m - 1]] for i in pos_idx]
                    neg_texts = [prompt + rollouts[i].text[: rollout_boundaries[i][m - 1]] for i in neg_idx]
                    pos_rewards = [float(rollouts[i].reward) for i in pos_idx]
                    neg_rewards = [float(rollouts[i].reward) for i in neg_idx]
                elif require_contrast:
                    contrast_considered_total += 1
                    correct_idx = [i for i in valid_idx if bool(rollouts[i].correct)]
                    incorrect_idx = [i for i in valid_idx if not bool(rollouts[i].correct)]
                    if len(incorrect_idx) < min_incorrect:
                        skipped_contrast_total += 1
                        continue
                    neg_idx = sorted(incorrect_idx, key=lambda i: float(rollouts[i].reward))[:k_neg]

                    use_rollout_pos = pos_source in {"rollout", "rollout_or_gold"} and len(correct_idx) >= min_correct
                    use_gold_pos = pos_source in {"gold", "rollout_or_gold"}
                    if use_rollout_pos:
                        pos_idx = sorted(correct_idx, key=lambda i: float(rollouts[i].reward), reverse=True)[:k_pos]
                        pos_texts = [prompt + rollouts[i].text[: rollout_boundaries[i][m - 1]] for i in pos_idx]
                        pos_rewards = [float(rollouts[i].reward) for i in pos_idx]
                    elif (
                        use_gold_pos
                        and gold_text_full is not None
                        and gold_reward is not None
                        and gold_boundaries is not None
                        and len(gold_boundaries) >= m
                    ):
                        need_end = int(gold_boundaries[m - 1])
                        pos_texts = [prompt + str(gold_text_full)[:need_end]]
                        pos_rewards = [float(gold_reward)]
                    else:
                        skipped_contrast_total += 1
                        continue

                    neg_texts = [prompt + rollouts[i].text[: rollout_boundaries[i][m - 1]] for i in neg_idx]
                    neg_rewards = [float(rollouts[i].reward) for i in neg_idx]
                else:
                    rewards = [float(rollouts[i].reward) for i in valid_idx]
                    pos_i, neg_i = _select_pos_neg(rewards, k_pos, k_neg)
                    pos_idx = [valid_idx[i] for i in pos_i]
                    neg_idx = [valid_idx[i] for i in neg_i]
                    pos_texts = [prompt + rollouts[i].text[: rollout_boundaries[i][m - 1]] for i in pos_idx]
                    neg_texts = [prompt + rollouts[i].text[: rollout_boundaries[i][m - 1]] for i in neg_idx]
                    pos_rewards = [float(rollouts[i].reward) for i in pos_idx]
                    neg_rewards = [float(rollouts[i].reward) for i in neg_idx]

                if not pos_texts or not neg_texts:
                    continue

                reward_gap = float(np.mean(pos_rewards) - np.mean(neg_rewards))
                if not np.isfinite(reward_gap):
                    continue
                quality = float(reward_gap)
                if random_mining:
                    quality = max(0.0, float(quality))
                elif reward_gap <= 0.0:
                    continue

                key = (int(ex_i), int(m))
                expected_counts[key] = (len(pos_texts), len(neg_texts))
                reward_meta[key] = {
                    "reward_gap": float(reward_gap),
                    "quality": float(quality),
                    "r_pos_mean": float(np.mean(pos_rewards)),
                    "r_neg_mean": float(np.mean(neg_rewards)),
                }
                for t in pos_texts:
                    all_texts.append(t)
                    text_meta.append((int(ex_i), int(m), True))
                for t in neg_texts:
                    all_texts.append(t)
                    text_meta.append((int(ex_i), int(m), False))

        # Hidden-state capture for all selected prefixes in this batch.
        if all_texts:
            pos_sums: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
            neg_sums: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
            pos_counts: dict[tuple[int, int], int] = {}
            neg_counts: dict[tuple[int, int], int] = {}

            for start in range(0, len(all_texts), hs_batch_size):
                chunk_texts = all_texts[start : start + hs_batch_size]
                try:
                    chunk_hidden_states, _ = hs.get_all_hidden_states_generate(
                        llm,
                        chunk_texts,
                        max_tokens=1,
                        split_by_samples=True,
                    )
                except Exception:
                    continue

                if not chunk_hidden_states:
                    continue
                if layers is None:
                    n_layers = int(len(chunk_hidden_states[0]))
                    if layer_specs is None:
                        start_layer = (2 * n_layers) // 3
                        layers = list(range(start_layer, n_layers))
                        logger.info("Mining candidate layers (top third): %s", layers)
                    else:
                        layers = _resolve_candidate_layers(list(layer_specs), num_layers=n_layers)
                        logger.info("Mining candidate layers (resolved): %s (spec=%s)", layers, layer_specs)
                assert layers is not None

                for i, sample_layers in enumerate(chunk_hidden_states):
                    global_i = int(start + i)
                    if global_i >= len(text_meta):
                        break
                    ex_i, m, is_pos = text_meta[global_i]
                    key = (int(ex_i), int(m))

                    if is_pos:
                        pos_counts[key] = int(pos_counts.get(key, 0)) + 1
                        layer_sums = pos_sums.setdefault(key, {})
                    else:
                        neg_counts[key] = int(neg_counts.get(key, 0)) + 1
                        layer_sums = neg_sums.setdefault(key, {})

                    for layer_id in layers:
                        lid = int(layer_id)
                        if lid < 0 or lid >= len(sample_layers):
                            continue
                        try:
                            v = sample_layers[lid][-1].to(torch.float32)
                        except Exception:
                            continue
                        v = v.detach().cpu()
                        if lid not in layer_sums:
                            layer_sums[lid] = v.clone()
                        else:
                            layer_sums[lid].add_(v)

            if layers is not None:
                for (ex_i, m), (need_pos, need_neg) in expected_counts.items():
                    got_pos = int(pos_counts.get((ex_i, m), 0))
                    got_neg = int(neg_counts.get((ex_i, m), 0))
                    if got_pos < int(need_pos) or got_neg < int(need_neg):
                        continue

                    meta_rew = reward_meta.get((ex_i, m), {})
                    reward_gap = float(meta_rew.get("reward_gap", 0.0))
                    quality = float(meta_rew.get("quality", reward_gap))
                    if not np.isfinite(reward_gap) or not np.isfinite(quality):
                        continue
                    if not random_mining and float(reward_gap) <= 0.0:
                        continue

                    ex = batch[int(ex_i)]
                    for layer_id in layers:
                        lid = int(layer_id)
                        pos_sum = pos_sums.get((ex_i, m), {}).get(lid)
                        neg_sum = neg_sums.get((ex_i, m), {}).get(lid)
                        if pos_sum is None or neg_sum is None:
                            continue

                        h_right = (pos_sum / float(got_pos)).cpu().numpy().astype(np.float32)  # (H,)
                        h_wrong = (neg_sum / float(got_neg)).cpu().numpy().astype(np.float32)  # (H,)
                        delta = (h_right - h_wrong).astype(np.float32)  # (H,)
                        if not np.isfinite(delta).all():
                            continue

                        q_val = float(quality)
                        if not random_mining:
                            q_val = float(reward_gap)
                        q_val = max(0.0, float(q_val))

                        meta = {
                            "task": ex.task,
                            "example_id": ex.id,
                            "control_point_m": int(m),
                            "layer": int(lid),
                            "quality": float(q_val),
                            "reward_gap": float(reward_gap),
                            "r_pos_mean": float(meta_rew.get("r_pos_mean", 0.0)),
                            "r_neg_mean": float(meta_rew.get("r_neg_mean", 0.0)),
                            "n_pos": int(need_pos),
                            "n_neg": int(need_neg),
                        }

                        tie += 1
                        item = (float(q_val), tie, meta, h_wrong, h_right, delta)
                        if len(heap) < int(cfg.offline_mine.keep_top_C):
                            heapq.heappush(heap, item)
                        else:
                            if float(q_val) > float(heap[0][0]):
                                heapq.heapreplace(heap, item)

        pbar.update(len(batch))
    pbar.close()

    if require_contrast and contrast_considered_total:
        logger.info(
            "Stage I contrast mining: skipped %d/%d control points (no correct/incorrect contrast).",
            int(skipped_contrast_total),
            int(contrast_considered_total),
        )

    # Persist rollouts
    write_jsonl(out_root / "rollouts.jsonl", all_rollout_rows)
    logger.info("Wrote rollouts: %s", str(out_root / "rollouts.jsonl"))

    # Export top-C candidates to .pt (DirectAlgorithm can load .pt tensors/ndarrays)
    # Sort descending for deterministic IDs.
    heap_sorted = sorted(heap, key=lambda x: x[0], reverse=True)
    cand_rows = []

    tool_id = 0
    for pair_id, (quality, _tie, meta, h_wrong, h_right, delta) in enumerate(tqdm(heap_sorted, desc="mine/export")):
        layer_id = int(meta["layer"])
        m = int(meta["control_point_m"])

        vec_path = vec_dir / f"pair_{pair_id:05d}_L{layer_id}_m{m:02d}_delta.pt"
        key_wrong_path = key_dir / f"pair_{pair_id:05d}_L{layer_id}_m{m:02d}_wrong.pt"
        key_right_path = key_dir / f"pair_{pair_id:05d}_L{layer_id}_m{m:02d}_right.pt"
        torch.save(np.asarray(delta, dtype=np.float32), str(vec_path))
        torch.save(np.asarray(h_wrong, dtype=np.float32), str(key_wrong_path))
        torch.save(np.asarray(h_right, dtype=np.float32), str(key_right_path))

        common = {
            "pair_id": int(pair_id),
            "layer": int(layer_id),
            "control_point_m": int(m),
            "quality": float(quality),
            "reward_gap": float(meta.get("reward_gap", quality)),
            "r_pos_mean": float(meta.get("r_pos_mean", 0.0)),
            "r_neg_mean": float(meta.get("r_neg_mean", 0.0)),
            "n_pos": int(meta.get("n_pos", 0)),
            "n_neg": int(meta.get("n_neg", 0)),
            "task": meta.get("task"),
            "example_id": meta.get("example_id"),
        }

        cand_rows.append(
            {
                "tool_id": int(tool_id),
                "tool_name": f"pair_{pair_id:05d}_wrong",
                "entry_type": "wrong",
                "key_path": str(key_wrong_path),
                "vector_path": str(vec_path),
                **common,
            }
        )
        tool_id += 1
        cand_rows.append(
            {
                "tool_id": int(tool_id),
                "tool_name": f"pair_{pair_id:05d}_right",
                "entry_type": "right",
                "key_path": str(key_right_path),
                "vector_path": None,
                **common,
            }
        )
        tool_id += 1

    write_jsonl(out_root / "candidates.jsonl", cand_rows)
    logger.info("Exported %d candidate memory entries to %s", len(cand_rows), str(out_root))
    logger.info("Wrote candidate manifest: %s", str(out_root / "candidates.jsonl"))

    # Mark this run as the latest for convenience (so later stages can set outputs.run_id=latest).
    try:
        latest_path = Path(cfg.outputs.root_dir) / cfg.outputs.run_name / "LATEST"
        write_text(latest_path, str(cfg.outputs.run_id))
        logger.info("Updated latest run pointer: %s -> %s", str(latest_path), str(cfg.outputs.run_id))
    except Exception as e:
        logger.warning("Failed to write LATEST pointer: %s", e)

    try:
        eng = getattr(llm, "llm_engine", None)
        engine_core = getattr(eng, "engine_core", None) if eng is not None else None
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
    except Exception:
        pass
