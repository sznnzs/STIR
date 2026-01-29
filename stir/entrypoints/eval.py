from __future__ import annotations

import logging

from stir.config import STIRConfig

logger = logging.getLogger(__name__)


def run_eval(cfg: STIRConfig) -> None:
    from stir.analysis.cases import write_case_markdown
    from stir.analysis.compare_methods import write_method_comparison
    from stir.analysis.diagnostics import write_eval_diagnostics
    from stir.online.stir import run_stir_dataset
    from stir.online.greedy import run_greedy_dataset

    methods = [str(m).lower() for m in cfg.eval.methods]

    T = int(cfg.decode.max_new_tokens)
    logger.info("Eval: T_max=%d (decode.max_new_tokens)", int(T))

    from vllm import LLM  # type: ignore

    need_steer_vector = ("stir" in methods) or bool(getattr(cfg.eval, "ablations", None))
    llm_kwargs = dict(
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
    if need_steer_vector:
        llm_kwargs.update(enable_steer_vector=True, max_steer_vectors=max(8, int(cfg.model.max_num_seqs)))

    llm = LLM(**llm_kwargs)
    try:
        acc_at_T: dict[str, float] = {}
        tags: dict[str, str] = {}
        if "greedy" in methods:
            tags["greedy"] = f"greedy_T{T}"
            acc_at_T["Greedy-CoT"] = float(
                run_greedy_dataset(cfg, max_new_tokens=T, out_tag=tags["greedy"], llm=llm)
            )
        if "stir" in methods:
            tags["stir"] = f"STIR_T{T}"
            acc_at_T["STIR"] = float(run_stir_dataset(cfg, max_new_tokens=T, out_tag=tags["stir"], llm=llm))

        # Main-results row for this dataset (single-budget helper).
        import csv
        from pathlib import Path

        out_main = Path(f"{cfg.outputs.run_dir}/tables/main_results_single.csv")
        out_main.parent.mkdir(parents=True, exist_ok=True)
        with out_main.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "split", "T_max"] + list(acc_at_T.keys()))
            w.writerow([cfg.task.dataset, cfg.task.eval_split, int(T)] + [acc_at_T[k] for k in acc_at_T.keys()])
        logger.info("Wrote single-dataset main results: %s", out_main)

        # Case studies at the largest budget (if both methods were run)
        if "greedy" in methods and "stir" in methods:
            md = write_case_markdown(
                run_dir=cfg.outputs.run_dir,
                dataset=cfg.task.dataset,
                split=cfg.task.eval_split,
                max_examples=cfg.task.max_eval_examples,
                seed=cfg.seed,
                data_root=cfg.task.data_root,
                T_max=int(T),
                greedy_tag=str(tags["greedy"]),
                stir_tag=str(tags["stir"]),
                top_n=int(getattr(cfg.eval, "case_top_n", 5)),
            )
            if md:
                logger.info("Wrote case studies: %s", md)

            # Lightweight diagnostics for debugging and paper analysis.
            write_eval_diagnostics(
                run_dir=cfg.outputs.run_dir,
                T_max=int(T),
                greedy_tag=str(tags["greedy"]),
                stir_tag=str(tags["stir"]),
            )
            write_method_comparison(
                run_dir=cfg.outputs.run_dir,
                T_max=int(T),
                greedy_tag=str(tags["greedy"]),
                stir_tag=str(tags["stir"]),
            )

        # Ablations (run at decode.max_new_tokens)
        if getattr(cfg.eval, "ablations", None):
            import csv
            from pathlib import Path

            ab_rows = []
            original_variant = cfg.online.variant
            for variant in cfg.eval.ablations:
                cfg.online.variant = str(variant)
                acc = run_stir_dataset(cfg, max_new_tokens=int(T), out_tag=f"{variant}_T{int(T)}", llm=llm)
                ab_rows.append({"variant": str(variant), "T_max": int(T), "acc": float(acc)})
            cfg.online.variant = original_variant

            out_ab = Path(f"{cfg.outputs.run_dir}/tables/ablation.csv")
            out_ab.parent.mkdir(parents=True, exist_ok=True)
            with out_ab.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["variant", "T_max", "acc"])
                for r in ab_rows:
                    w.writerow([r["variant"], r["T_max"], r["acc"]])
            logger.info("Wrote ablation results: %s", out_ab)
    finally:
        try:
            eng = getattr(llm, "llm_engine", None)
            engine_core = getattr(eng, "engine_core", None) if eng is not None else None
            if engine_core is not None and hasattr(engine_core, "shutdown"):
                engine_core.shutdown()
        except Exception:
            pass
