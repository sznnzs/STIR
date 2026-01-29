#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single entry-point CLI for the STIR paper codebase.

Design goals:
1) One command surface, minimal scripts
2) Reproducible outputs under outputs/
3) Config-first (YAML) with CLI overrides
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

import yaml

from stir.config import STIRConfig, load_config
from stir.utils.logging import setup_logging


def _dump_config(cfg: STIRConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config_resolved.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if getattr(args, "run_id", None) is not None:
        cfg.outputs.set_run_id(args.run_id)
    setup_logging(cfg.outputs.log_dir)
    _dump_config(cfg, Path(cfg.outputs.run_dir))
    from stir.entrypoints.eval import run_eval

    run_eval(cfg)


def cmd_mine(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if getattr(args, "run_id", None) is not None:
        cfg.outputs.set_run_id(args.run_id)
    setup_logging(cfg.outputs.log_dir)
    _dump_config(cfg, Path(cfg.outputs.run_dir))
    from stir.entrypoints.mine import run_mine

    run_mine(cfg)


def cmd_select(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if getattr(args, "run_id", None) is not None:
        cfg.outputs.set_run_id(args.run_id)
    setup_logging(cfg.outputs.log_dir)
    _dump_config(cfg, Path(cfg.outputs.run_dir))
    from stir.entrypoints.select import run_select

    run_select(cfg)


def cmd_memory(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if getattr(args, "run_id", None) is not None:
        cfg.outputs.set_run_id(args.run_id)
    setup_logging(cfg.outputs.log_dir)
    _dump_config(cfg, Path(cfg.outputs.run_dir))
    from stir.entrypoints.memory import run_memory

    run_memory(cfg)


def cmd_case(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if getattr(args, "run_id", None) is not None:
        cfg.outputs.set_run_id(args.run_id)
    setup_logging(cfg.outputs.log_dir)
    _dump_config(cfg, Path(cfg.outputs.run_dir))
    from stir.entrypoints.case import run_case

    ids_s = getattr(args, "example_ids", None) or getattr(args, "example_id", None)
    if ids_s is None:
        raise ValueError("--example-id/--example-ids is required.")
    example_ids = [s.strip() for s in str(ids_s).split(",") if s.strip() != ""]
    methods = [s.strip() for s in str(getattr(args, "methods", "greedy,stir")).split(",") if s.strip() != ""]
    out_md = run_case(
        cfg,
        example_ids=example_ids,
        max_new_tokens=getattr(args, "max_new_tokens", None),
        methods=methods,
        tag=str(getattr(args, "tag", "case")),
    )
    print(out_md)


def _sanitize_run_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "run"


def _write_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def _aggregate_suite(
    *,
    outputs_root: Path,
    suite_id: str,
    run_names: list[str],
    dataset_labels: list[str],
) -> Path:
    """
    Aggregate per-dataset outputs into one suite folder:
    - tables/suite_main_results.csv (concatenated main_results_single.csv)
    - tables/suite_ablation_long.csv
    """
    import csv

    suite_dir = outputs_root / "_suite" / suite_id
    (suite_dir / "tables").mkdir(parents=True, exist_ok=True)

    # 1) main results
    main_rows = []
    main_header = None
    for run_name, ds_label in zip(run_names, dataset_labels):
        run_dir = outputs_root / run_name / suite_id
        p = run_dir / "tables" / "main_results_single.csv"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            row = next(r, None)
        if not header or not row:
            continue
        # overwrite dataset field for safety
        row = list(row)
        if len(row) >= 1:
            row[0] = ds_label
        if main_header is None:
            main_header = header
        main_rows.append(row)
    out_main = suite_dir / "tables" / "suite_main_results.csv"
    if main_header is not None:
        with out_main.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(main_header)
            for row in main_rows:
                w.writerow(row)

    # 2) ablation
    abl_rows = []
    for run_name, ds_label in zip(run_names, dataset_labels):
        run_dir = outputs_root / run_name / suite_id
        p = run_dir / "tables" / "ablation.csv"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            if not header:
                continue
            for row in r:
                if not row:
                    continue
                abl_rows.append([ds_label] + row)
    out_abl = suite_dir / "tables" / "suite_ablation_long.csv"
    with out_abl.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "T_max", "acc"])
        for row in abl_rows:
            w.writerow(row)

    return suite_dir


def cmd_suite(args: argparse.Namespace) -> None:
    """
    Offline-friendly multi-dataset runner.

    The YAML config should contain a top-level `suite` section:

    suite:
      run_name_prefix: "suite_small"
      stages: ["mine", "select", "memory", "eval"]
      datasets:
        - name: "gsm8k"
          prompt_template: "gsm8k_0shot"
          train_split: "train"
          eval_split: "test"
          max_train_examples: 200
          max_eval_examples: 200
        - name: "arc-c"
          prompt_template: "arc_0shot"
          train_split: "train"
          eval_split: "validation"
          max_train_examples: 200
          max_eval_examples: 200

    Use --gpus "0,1" to run in parallel on multiple GPUs (one job per GPU).
    """
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "suite" not in raw:
        raise ValueError("suite mode requires a top-level suite: {...} block in the YAML file.")
    suite = raw.get("suite") or {}
    datasets = suite.get("datasets") or []
    if not datasets:
        raise ValueError("suite.datasets is empty.")

    # suite_id = shared outputs.run_id across all datasets (makes aggregation easy)
    suite_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    # GPUs / stages
    gpu_list = []
    if getattr(args, "gpus", None):
        gpu_list = [g.strip() for g in str(args.gpus).split(",") if g.strip() != ""]
    if not gpu_list:
        gpu_list = [str(g) for g in (suite.get("gpus") or ["0"])]
    stages = suite.get("stages") or ["mine", "select", "memory", "eval"]
    if getattr(args, "stages", None):
        stages = [s.strip() for s in str(args.stages).split(",") if s.strip()]

    outputs_root = Path((raw.get("outputs") or {}).get("root_dir", str(Path(__file__).parent / "outputs")))
    run_prefix = suite.get("run_name_prefix") or (raw.get("outputs") or {}).get("run_name") or "suite"

    suite_dir = outputs_root / "_suite" / suite_id
    suite_cfg_dir = suite_dir / "configs"
    suite_log_dir = suite_dir / "logs"
    suite_cfg_dir.mkdir(parents=True, exist_ok=True)
    suite_log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    run_names: list[str] = []
    ds_labels: list[str] = []

    for ds in datasets:
        if not isinstance(ds, dict) or "name" not in ds:
            raise ValueError("Each suite.datasets entry must be a dict containing name.")
        ds_name = str(ds["name"])
        ds_label = ds_name
        ds_key = _sanitize_run_name(ds_name)
        run_name = f"{_sanitize_run_name(run_prefix)}_{ds_key}"

        # Derived config = base raw without suite + per-dataset overrides.
        d = copy.deepcopy(raw)
        d.pop("suite", None)
        d.setdefault("task", {})
        d.setdefault("prompt", {})
        d.setdefault("outputs", {})
        d["task"]["dataset"] = ds_name
        for k in ["data_root", "train_split", "eval_split", "max_train_examples", "max_eval_examples"]:
            if k in ds:
                d["task"][k] = ds[k]
        if "prompt_template" in ds:
            d["prompt"]["template"] = ds["prompt_template"]
        d["outputs"]["run_name"] = run_name
        d["outputs"]["run_id"] = suite_id

        out_cfg = suite_cfg_dir / f"{run_name}.yaml"
        _write_yaml(d, out_cfg)

        # Command: run selected stages sequentially in separate Python processes.
        py = sys.executable
        repo_root = str(Path(__file__).parent)
        stage_cmds = []
        for st in stages:
            st = st.strip()
            stage_cmds.append(f"cd {repo_root} && {py} run.py --config {out_cfg} --run-id {suite_id} {st}")
        cmd = " && ".join(stage_cmds)
        log_path = suite_log_dir / f"{run_name}.log"
        jobs.append((ds_label, run_name, str(out_cfg), cmd, log_path))
        run_names.append(run_name)
        ds_labels.append(ds_label)

    # Simple GPU scheduler: one job per GPU
    queue = list(jobs)
    running: list[tuple[str, str, str, subprocess.Popen]] = []  # (gpu, run_name, ds_label, proc)
    available = list(gpu_list)

    while queue or running:
        # launch while possible
        while queue and available:
            ds_label, run_name, _cfgp, cmd, log_path = queue.pop(0)
            gpu = available.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            with log_path.open("w", encoding="utf-8") as lf:
                proc = subprocess.Popen(
                    ["bash", "-lc", cmd],
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
            running.append((gpu, run_name, ds_label, proc))

        # poll
        still = []
        for gpu, run_name, ds_label, proc in running:
            rc = proc.poll()
            if rc is None:
                still.append((gpu, run_name, ds_label, proc))
            else:
                available.append(gpu)
                if rc != 0:
                    raise RuntimeError(f"Suite subtask failed: dataset={ds_label} run_name={run_name} rc={rc}")
        running = still
        if running:
            time.sleep(5.0)

    # Aggregate outputs
    _aggregate_suite(outputs_root=outputs_root, suite_id=suite_id, run_names=run_names, dataset_labels=ds_labels)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="STIR paper pipeline: data/mine/select/memory/eval/analyze",
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "debug.yaml"),
        help="Path to YAML config file.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override outputs.run_id. Use 'latest' to resume the latest run under outputs/<run_name>/LATEST.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_eval = sub.add_parser("eval", help="Run evaluation (baselines / STIR online).")
    sp_eval.set_defaults(func=cmd_eval)

    sp_mine = sub.add_parser("mine", help="Stage I: mine candidate steering tools.")
    sp_mine.set_defaults(func=cmd_mine)

    sp_select = sub.add_parser("select", help="Stage II: select diverse tool library.")
    sp_select.set_defaults(func=cmd_select)

    sp_mem = sub.add_parser("memory", help="Stage III: build episodic memory index.")
    sp_mem.set_defaults(func=cmd_memory)

    sp_case = sub.add_parser("case", help="Case-level debug run (fast iteration).")
    sp_case.add_argument("--example-id", type=str, default=None, help="Single example id (e.g. '0').")
    sp_case.add_argument(
        "--example-ids",
        type=str,
        default=None,
        help="Comma-separated example ids (e.g. '0,1,2'). Overrides --example-id.",
    )
    sp_case.add_argument(
        "--max-new-tokens",
        "--budget",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="Override decode.max_new_tokens (T_max) for this case run.",
    )
    sp_case.add_argument(
        "--methods",
        type=str,
        default="greedy,stir",
        help="Comma-separated methods: greedy,stir.",
    )
    sp_case.add_argument(
        "--tag",
        type=str,
        default="case",
        help="Tag prefix for output folders/files under outputs/<run>/eval and outputs/<run>/cases.",
    )
    sp_case.set_defaults(func=cmd_case)

    sp_suite = sub.add_parser("suite", help="Run multi-dataset suite (offline, multi-GPU).")
    sp_suite.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids, e.g. '0,1'. One dataset job per GPU.",
    )
    sp_suite.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated stages to run, e.g. 'mine,select,memory,eval'. Default from suite.stages.",
    )
    sp_suite.set_defaults(func=cmd_suite)

    return p


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.chdir(Path(__file__).resolve().parent)
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
