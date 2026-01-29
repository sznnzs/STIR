from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams  # type: ignore

from stir.config import STIRConfig
from stir.data.loaders import load_task_dataset
from stir.data.tasks import TaskExample
from stir.data.prompts import DEFAULT_SYSTEM_PROMPT, build_chat_messages, render_user_prompt
from stir.eval.extractors import extract_gold, extract_pred
from stir.eval.metrics import is_correct as is_correct_pred
from stir.utils.io import ensure_dir, write_jsonl

logger = logging.getLogger(__name__)


def run_greedy_dataset(
    cfg: STIRConfig,
    *,
    max_new_tokens: int | None = None,
    out_tag: str = "greedy",
    examples: list[TaskExample] | None = None,
    llm: LLM | None = None,
) -> float:
    run_root = Path(cfg.outputs.run_dir)
    out_dir = ensure_dir(run_root / "eval" / out_tag)

    T_max = int(max_new_tokens) if max_new_tokens is not None else int(cfg.decode.max_new_tokens)

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
        )
    tokenizer = llm.get_tokenizer()

    if examples is None:
        examples = load_task_dataset(
            task=cfg.task.dataset,
            split=cfg.task.eval_split,
            max_examples=cfg.task.max_eval_examples,
            seed=cfg.seed,
            data_root=cfg.task.data_root,
        )

    if not examples:
        write_jsonl(Path(out_dir) / "per_example.jsonl", [])
        write_jsonl(Path(out_dir) / "summary.jsonl", [{"n": 0, "acc": 0.0, "T_max": int(T_max)}])
        logger.info("%s eval done. n=%d acc=%.4f", out_tag, 0, 0.0)
        if owns_llm:
            try:
                eng = getattr(llm, "llm_engine", None)
                engine_core = getattr(eng, "engine_core", None) if eng is not None else None
                if engine_core is not None and hasattr(engine_core, "shutdown"):
                    engine_core.shutdown()
            except Exception:
                pass
        return 0.0

    prompts: list[str] = []
    golds: list[str | None] = []
    example_ids: list[str] = []
    tasks: list[str] = []
    for ex in examples:
        golds.append(extract_gold(ex.task, ex.answer))
        example_ids.append(str(ex.id))
        tasks.append(str(ex.task))

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
        prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=float(cfg.decode.temperature),
        top_p=float(cfg.decode.top_p),
        max_tokens=int(T_max),
        logprobs=int(cfg.decode.logprobs) if cfg.decode.logprobs is not None else None,
    )
    outs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    rows: list[dict[str, Any]] = []
    correct = 0
    for i, ro in enumerate(outs):
        task = tasks[i]
        gold = golds[i]
        if not getattr(ro, "outputs", None):
            out_text = ""
            out_token_ids: list[int] = []
            finish_reason: str | None = "empty"
        else:
            co = ro.outputs[0]
            out_text = str(getattr(co, "text", ""))
            out_token_ids = list(getattr(co, "token_ids", []) or [])
            finish_reason = getattr(co, "finish_reason", None)

        pred = extract_pred(task, out_text)
        correct_flag = bool(is_correct_pred(task, pred, gold))
        correct += int(correct_flag)
        rows.append(
            {
                "task": tasks[i],
                "example_id": example_ids[i],
                "pred": pred,
                "gold": gold,
                "correct": bool(correct_flag),
                "tokens_used": int(len(out_token_ids)),
                "finish_reason": finish_reason,
                "text": out_text,
            }
        )

    write_jsonl(Path(out_dir) / "per_example.jsonl", rows)
    acc = correct / max(1, len(rows))
    write_jsonl(Path(out_dir) / "summary.jsonl", [{"n": len(rows), "acc": acc, "T_max": T_max}])
    logger.info("%s eval done. n=%d acc=%.4f", out_tag, len(rows), acc)

    # region agent log
    try:
        toks = [int(r.get("tokens_used", 0)) for r in rows]
        payload = {
            "sessionId": "debug-session",
            "runId": str(getattr(cfg.outputs, "run_id", "")),
            "hypothesisId": "H1_budget_overhead",
            "location": "stir/online/greedy.py:run_greedy_dataset",
            "message": "greedy_summary",
            "data": {
                "run_name": str(getattr(cfg.outputs, "run_name", "")),
                "dataset": str(cfg.task.dataset),
                "split": str(cfg.task.eval_split),
                "out_tag": str(out_tag),
                "T_max": int(T_max),
                "n": int(len(rows)),
                "acc": float(acc),
                "tokens_used_mean": float(sum(toks) / max(1, len(toks))),
            },
            "timestamp": int(time.time() * 1000),
        }
        agent_log = Path(cfg.outputs.log_dir) / "agent_debug.jsonl"
        agent_log.parent.mkdir(parents=True, exist_ok=True)
        with agent_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion

    if owns_llm:
        try:
            eng = getattr(llm, "llm_engine", None)
            engine_core = getattr(eng, "engine_core", None) if eng is not None else None
            if engine_core is not None and hasattr(engine_core, "shutdown"):
                engine_core.shutdown()
        except Exception:
            pass
    return float(acc)
