from __future__ import annotations

from pathlib import Path

from stir.data.loaders import load_task_dataset
from stir.utils.io import read_jsonl, write_text


def _index_by_id(rows: list[dict]) -> dict[str, dict]:
    return {str(r["example_id"]): r for r in rows}


def write_case_markdown(
    *,
    run_dir: str | Path,
    dataset: str,
    split: str,
    max_examples: int | None,
    seed: int,
    data_root: str | None,
    T_max: int,
    greedy_tag: str,
    stir_tag: str,
    top_n: int = 5,
) -> str:
    run_dir = Path(run_dir)
    greedy_path = run_dir / "eval" / greedy_tag / "per_example.jsonl"
    stir_path = run_dir / "eval" / stir_tag / "per_example.jsonl"
    if not greedy_path.exists() or not stir_path.exists():
        return ""

    greedy = _index_by_id(read_jsonl(greedy_path))
    stir_rows = _index_by_id(read_jsonl(stir_path))

    exs = load_task_dataset(dataset, split, max_examples, seed, data_root=data_root)
    q_by_id = {str(e.id): e.question for e in exs}

    improve = []
    regress = []
    for ex_id in stir_rows.keys():
        if ex_id not in greedy:
            continue
        g = greedy[ex_id]
        e = stir_rows[ex_id]
        if bool(e.get("correct")) and not bool(g.get("correct")):
            improve.append(ex_id)
        if not bool(e.get("correct")) and bool(g.get("correct")):
            regress.append(ex_id)

    improve = improve[:top_n]
    regress = regress[:top_n]

    lines = []
    lines.append(f"# Case studies (T_max={T_max})")
    lines.append("")
    lines.append("## STIR improves over Greedy")
    lines.append("")
    for idx, ex_id in enumerate(improve, 1):
        g = greedy[ex_id]
        e = stir_rows[ex_id]
        q = q_by_id.get(ex_id, "")
        lines.append(f"### Improve-{idx}: id={ex_id}")
        lines.append("")
        lines.append("**Question**")
        lines.append("")
        lines.append(q)
        lines.append("")
        lines.append(f"**Gold**: {e.get('gold')}")
        lines.append("")
        lines.append(f"**Greedy pred**: {g.get('pred')}  | correct={g.get('correct')}")
        lines.append("")
        lines.append("**Greedy output**")
        lines.append("")
        lines.append("```")
        lines.append(str(g.get("text", ""))[:2000])
        lines.append("```")
        lines.append("")
        lines.append(f"**STIR pred**: {e.get('pred')}  | correct={e.get('correct')}")
        lines.append("")
        lines.append("**STIR output**")
        lines.append("")
        lines.append("```")
        lines.append(str(e.get("text", ""))[:2000])
        lines.append("```")
        lines.append("")

    lines.append("## STIR regresses vs Greedy")
    lines.append("")
    for idx, ex_id in enumerate(regress, 1):
        g = greedy[ex_id]
        e = stir_rows[ex_id]
        q = q_by_id.get(ex_id, "")
        lines.append(f"### Regress-{idx}: id={ex_id}")
        lines.append("")
        lines.append("**Question**")
        lines.append("")
        lines.append(q)
        lines.append("")
        lines.append(f"**Gold**: {e.get('gold')}")
        lines.append("")
        lines.append(f"**Greedy pred**: {g.get('pred')}  | correct={g.get('correct')}")
        lines.append("")
        lines.append("```")
        lines.append(str(g.get("text", ""))[:2000])
        lines.append("```")
        lines.append("")
        lines.append(f"**STIR pred**: {e.get('pred')}  | correct={e.get('correct')}")
        lines.append("")
        lines.append("```")
        lines.append(str(e.get("text", ""))[:2000])
        lines.append("```")
        lines.append("")

    out_dir = run_dir / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cases_T{T_max}.md"
    write_text(out_path, "\n".join(lines) + "\n")
    return str(out_path)


