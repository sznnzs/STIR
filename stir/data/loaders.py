from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from stir.data.tasks import TaskExample

logger = logging.getLogger(__name__)


def _try_import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import huggingface datasets. Use the easysteer env or install a compatible version: "
            "pip install -U datasets httpx\n"
            f"Import error: {type(e).__name__}: {e}"
        ) from e
    return load_dataset


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _read_parquet(path: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Reading local parquet requires pyarrow. Install with: pip install pyarrow\n"
            f"Import error: {type(e).__name__}: {e}"
        ) from e
    table = pq.read_table(path)
    return table.to_pylist()


def load_task_dataset(
    task: str,
    split: str,
    max_examples: int | None,
    seed: int = 0,
    data_root: str | None = None,
) -> list[TaskExample]:
    """
    Load dataset examples in a paper-friendly, unified schema.

    Notes:
    - We prefer HF datasets for convenience; if your environment is offline,
      consider setting HF mirror/caches or providing local jsonl (future extension).
    """
    task = str(task).lower().strip()
    # Be forgiving to path-like dataset names (common in local caches), e.g.:
    # - "datasets/math-ai/amc23" -> "amc23"
    # - "HuggingFaceH4/MATH-500" -> "math-500"
    if "/" in task:
        task = task.split("/")[-1]
    split = str(split)
    limit = int(max_examples) if max_examples is not None else None

    def _maybe_local_path(rel: str) -> str | None:
        if not data_root:
            return None
        p = Path(data_root) / rel
        return str(p) if p.exists() else None

    if task == "gsm8k":
        # Prefer local parquet if provided (offline-friendly).
        local = _maybe_local_path(f"openai/gsm8k/main/{split}-00000-of-00001.parquet")
        if local:
            ds = _read_parquet(local)
        else:
            load_dataset = _try_import_datasets()
            ds = load_dataset("gsm8k", "main", split=split)
        rows = []
        for i, ex in enumerate(ds):
            q = ex["question"]
            a = ex["answer"]
            rows.append(
                TaskExample(
                    task="gsm8k",
                    id=str(i),
                    question=q,
                    answer=a,
                    meta={},
                )
            )
            if limit is not None and len(rows) >= limit:
                break
        return _maybe_truncate(rows, max_examples)

    if task == "svamp":
        # NOTE: HF hub may be unavailable. If you have a local SVAMP, please put it under:
        #   {data_root}/svamp/{split}.jsonl  (fields: Body, Question, Answer)
        # or
        #   {data_root}/ChilleD/SVAMP/{split}.jsonl
        local = _maybe_local_path(f"svamp/{split}.jsonl") or _maybe_local_path(
            f"ChilleD/SVAMP/{split}.jsonl"
        )
        if local:
            ds = _read_jsonl(local)
        else:
            load_dataset = _try_import_datasets()
            ds = load_dataset("ChilleD/SVAMP", split=split)
        rows = []
        for i, ex in enumerate(ds):
            q = ex.get("Body", "") + "\n" + ex.get("Question", "")
            a = str(ex.get("Answer", ex.get("answer", "")))
            rows.append(
                TaskExample(task="svamp", id=str(i), question=q.strip(), answer=a, meta={})
            )
            if limit is not None and len(rows) >= limit:
                break
        return _maybe_truncate(rows, max_examples)

    if task in {"math500", "math-500"}:
        # Local MATH-500 (HuggingFaceH4/MATH-500/test.jsonl)
        local = _maybe_local_path("HuggingFaceH4/MATH-500/test.jsonl")
        if not local:
            raise FileNotFoundError(
                "Local MATH-500 not found. Expected at "
                f"{(data_root or '<data_root>')}/HuggingFaceH4/MATH-500/test.jsonl"
            )
        ds = _read_jsonl(local)
        rows = []
        for i, ex in enumerate(ds):
            q = ex["problem"]
            # Use 'solution' to let gold extractor pick \\boxed{...}
            a = ex.get("solution", ex.get("answer", ""))
            rows.append(
                TaskExample(
                    task="math500",
                    id=str(ex.get("unique_id", i)),
                    question=q,
                    answer=a,
                    meta={"answer": ex.get("answer"), "subject": ex.get("subject"), "level": ex.get("level")},
                )
            )
            if limit is not None and len(rows) >= limit:
                break
        return _maybe_truncate(rows, max_examples)

    if task in {"aime_2024", "aime2024"}:
        local = _maybe_local_path("HuggingFaceH4/aime_2024/data/train-00000-of-00001.parquet")
        if not local:
            raise FileNotFoundError(
                "Local AIME_2024 parquet not found. Expected at "
                f"{(data_root or '<data_root>')}/HuggingFaceH4/aime_2024/data/train-00000-of-00001.parquet"
            )
        ds = _read_parquet(local)
        rows = []
        for i, ex in enumerate(ds):
            q = ex["problem"]
            a = str(ex.get("answer", ""))
            rows.append(
                TaskExample(
                    task="aime_2024",
                    id=str(ex.get("id", i)),
                    question=q,
                    answer=a,
                    meta={"year": ex.get("year"), "url": ex.get("url")},
                )
            )
            if limit is not None and len(rows) >= limit:
                break
        return _maybe_truncate(rows, max_examples)

    if task in {"aime25", "aime_25", "aime_2025"}:
        # Local AIME 2025 (math-ai/aime25/test.jsonl)
        local = _maybe_local_path("math-ai/aime25/test.jsonl")
        if not local:
            raise FileNotFoundError(
                "Local AIME25 jsonl not found. Expected at "
                f"{(data_root or '<data_root>')}/math-ai/aime25/test.jsonl"
            )
        ds = _read_jsonl(local)
        rows = []
        for i, ex in enumerate(ds):
            q = str(ex.get("problem", ex.get("question", "")))
            a = str(ex.get("answer", ""))
            rows.append(TaskExample(task="aime25", id=str(ex.get("id", i)), question=q, answer=a, meta={}))
            if limit is not None and len(rows) >= limit:
                break
        return _maybe_truncate(rows, max_examples)

    if task in {"amc23", "amc_23", "amc_2023"}:
        # Local AMC 2023 (math-ai/amc23/test-*.parquet)
        local = _maybe_local_path(f"math-ai/amc23/{split}-00000-of-00001.parquet")
        if not local and data_root:
            cand_dir = Path(data_root) / "math-ai" / "amc23"
            if cand_dir.is_dir():
                matches = sorted(cand_dir.glob(f"{split}-*.parquet"))
                if matches:
                    local = str(matches[0])
        if not local:
            raise FileNotFoundError(
                "Local AMC23 parquet not found. Expected at "
                f"{(data_root or '<data_root>')}/math-ai/amc23/{split}-00000-of-00001.parquet"
            )
        ds = _read_parquet(local)
        rows = []
        for i, ex in enumerate(ds):
            q = str(ex.get("question", ""))
            a = str(ex.get("answer", ""))
            rows.append(
                TaskExample(
                    task="amc23",
                    id=str(ex.get("id", i)),
                    question=q,
                    answer=a,
                    meta={"url": ex.get("url")},
                )
            )
            if limit is not None and len(rows) >= limit:
                break
        return _maybe_truncate(rows, max_examples)

    if task in {"math", "competition_math"}:
        # HF canonical name is usually "competition_math" (Hendrycks MATH), but it can be gated.
        # We also support an open fallback "HuggingFaceH4/MATH".
        #
        # For quick iteration, prefer streaming when max_examples is set, and stop early.
        load_dataset = _try_import_datasets()
        use_streaming = limit is not None and limit > 0
        ds = None
        last_err: Exception | None = None
        for name in ("competition_math", "hendrycks/competition_math", "HuggingFaceH4/MATH"):
            try:
                ds = load_dataset(name, split=split, streaming=use_streaming)
                break
            except Exception as e:
                last_err = e
                ds = None
        if ds is None:
            raise RuntimeError(
                "Unable to load MATH from HuggingFace (competition_math may be gated). "
                "Check network/HF token or use the open fallback: HuggingFaceH4/MATH."
            ) from last_err
        rows = []
        for i, ex in enumerate(ds):
            q = ex.get("problem", "")
            a = ex.get("solution", ex.get("answer", ""))
            rows.append(
                TaskExample(
                    task="math",
                    id=str(i),
                    question=q,
                    answer=a,
                    meta={"level": ex.get("level"), "type": ex.get("type")},
                )
            )
            if limit is not None and len(rows) >= limit:
                break
        return _maybe_truncate(rows, max_examples)

    if task in {"arc-c", "arc_challenge", "arc"}:
        local = _maybe_local_path(f"allenai/ai2_arc/ARC-Challenge/{split}-00000-of-00001.parquet")
        if local:
            ds = _read_parquet(local)
        else:
            load_dataset = _try_import_datasets()
            ds = load_dataset("ai2_arc", "ARC-Challenge", split=split)
        rows = []
        for i, ex in enumerate(ds):
            q = ex["question"]
            choices = ex["choices"]
            # HF format: {"label":[...], "text":[...]}
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            options = {lab: txt for lab, txt in zip(labels, texts)}
            ans = ex.get("answerKey", "")
            rows.append(
                TaskExample(
                    task="arc-c",
                    id=str(ex.get("id", i)),
                    question=q,
                    answer=ans,
                    meta={"options": options},
                )
            )
            if limit is not None and len(rows) >= limit:
                break
        return _maybe_truncate(rows, max_examples)

    if task in {"openbookqa", "openbook_qa"}:
        local = _maybe_local_path(f"allenai/openbookqa/main/{split}-00000-of-00001.parquet")
        if local:
            ds = _read_parquet(local)
        else:
            # Optional HF fallback
            load_dataset = _try_import_datasets()
            ds = load_dataset("openbookqa", "main", split=split)
        rows = []
        for i, ex in enumerate(ds):
            q = ex.get("question_stem", ex.get("question", ""))
            choices = ex["choices"]
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            options = {lab: txt for lab, txt in zip(labels, texts)}
            ans = ex.get("answerKey", "")
            rows.append(
                TaskExample(
                    task="openbookqa",
                    id=str(ex.get("id", i)),
                    question=q,
                    answer=ans,
                    meta={"options": options},
                )
            )
            if limit is not None and len(rows) >= limit:
                break
        return _maybe_truncate(rows, max_examples)

    if task in {"commonsense_qa", "commonsenseqa"}:
        local = _maybe_local_path(f"tau/commonsense_qa/data/{split}-00000-of-00001.parquet")
        if local:
            ds = _read_parquet(local)
        else:
            load_dataset = _try_import_datasets()
            ds = load_dataset("commonsense_qa", split=split)
        rows = []
        for i, ex in enumerate(ds):
            q = ex["question"]
            choices = ex["choices"]
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            options = {lab: txt for lab, txt in zip(labels, texts)}
            ans = ex.get("answerKey", "")
            rows.append(
                TaskExample(
                    task="commonsense_qa",
                    id=str(ex.get("id", i)),
                    question=q,
                    answer=ans,
                    meta={"options": options},
                )
            )
            if limit is not None and len(rows) >= limit:
                break
        return _maybe_truncate(rows, max_examples)

    if task in {"strategyqa", "strategy_qa"}:
        # NOTE: hub may be unavailable. If you have a local StrategyQA, please put it under:
        #   {data_root}/strategyqa/{split}.jsonl  (fields: question, answer[bool or str])
        # or
        #   {data_root}/ChilleD/StrategyQA/{split}.jsonl
        local = _maybe_local_path(f"strategyqa/{split}.jsonl") or _maybe_local_path(
            f"ChilleD/StrategyQA/{split}.jsonl"
        )
        if local:
            ds = _read_jsonl(local)
        else:
            load_dataset = _try_import_datasets()
            ds = load_dataset("ChilleD/StrategyQA", split=split)
        rows = []
        for i, ex in enumerate(ds):
            q = ex["question"]
            a = ex.get("answer")
            if isinstance(a, bool):
                a = "Yes" if a else "No"
            else:
                a = str(a)
            rows.append(TaskExample(task="strategyqa", id=str(ex.get("id", i)), question=q, answer=a, meta={}))
            if limit is not None and len(rows) >= limit:
                break
        return _maybe_truncate(rows, max_examples)

    raise ValueError(f"Unknown task/dataset: {task}")


def _maybe_truncate(rows: list[TaskExample], max_examples: int | None) -> list[TaskExample]:
    if max_examples is None:
        return rows
    return rows[: int(max_examples)]
