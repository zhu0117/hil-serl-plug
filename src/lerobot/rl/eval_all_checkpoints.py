#!/usr/bin/env python

"""Batch-evaluate all checkpoints from a training run.

This script scans a run directory such as:

    output/franka_unplug_sac/checkpoints/
        0005000/pretrained_model/
        0010000/pretrained_model/

and evaluates each checkpoint sequentially by reusing:

    python -m lerobot.rl.eval_hil_sim

It patches `policy.pretrained_path` in a temporary eval config for each checkpoint,
runs evaluation, and prints a concise summary table.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate all checkpoints in one run.")
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Training run directory, e.g. output/franka_unplug_sac",
    )
    parser.add_argument(
        "--eval_config",
        default="src/lerobot/rl/hil_sim_eval_config.json",
        help="Base eval config JSON. policy.pretrained_path will be overridden per checkpoint.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch eval subprocesses.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if one checkpoint evaluation fails.",
    )
    parser.add_argument(
        "--quiet_eval",
        action="store_true",
        help="Do not stream inner eval logs. Only print checkpoint-level progress.",
    )
    return parser.parse_args()


def find_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints_root = run_dir / "checkpoints"
    if not checkpoints_root.is_dir():
        output_root = run_dir.parent
        candidates: list[str] = []
        if output_root.is_dir():
            for child in sorted(output_root.iterdir()):
                if not child.is_dir():
                    continue
                if (child / "checkpoints").is_dir():
                    candidates.append(str(child))

        hint = ""
        if candidates:
            hint = "\nPossible run_dir values:\n  - " + "\n  - ".join(candidates)
        else:
            hint = "\nNo sibling run directory with checkpoints was found."

        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_root}{hint}")

    items: list[tuple[int, Path]] = []
    for child in checkpoints_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue

        step = int(child.name)
        pretrained_model = child / "pretrained_model"
        if pretrained_model.is_dir():
            items.append((step, pretrained_model.resolve()))

    items.sort(key=lambda x: x[0])
    return items


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_temp_eval_config(base_cfg: dict, pretrained_path: str) -> Path:
    cfg = dict(base_cfg)
    policy = dict(cfg.get("policy", {}))
    policy["pretrained_path"] = pretrained_path
    cfg["policy"] = policy

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    with tmp:
        json.dump(cfg, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
    return Path(tmp.name)


def extract_summary(stdout: str) -> tuple[str, str, str]:
    success_rate = "N/A"
    avg_reward = "N/A"
    avg_steps = "N/A"

    m = re.search(r"\|\s*Successes\s+(\d+\s*\([^\)]*\))\s*\|", stdout)
    if m:
        success_rate = m.group(1).strip()

    m = re.search(r"\|\s*Avg reward\s+([^|]+?)\s*\|", stdout)
    if m:
        avg_reward = m.group(1).strip()

    m = re.search(r"\|\s*Avg steps/episode\s+([^|]+?)\s*\|", stdout)
    if m:
        avg_steps = m.group(1).strip()

    return success_rate, avg_reward, avg_steps


def run_one(python_exe: str, temp_cfg_path: Path, quiet_eval: bool) -> tuple[int, str]:
    cmd = [
        python_exe,
        "-m",
        "lerobot.rl.eval_hil_sim",
        "--config_path",
        str(temp_cfg_path),
    ]

    if quiet_eval:
        result = subprocess.run(cmd, text=True, capture_output=True)
        return result.returncode, result.stdout + "\n" + (result.stderr or "")

    process = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    collected: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(f"    {line}", end="")
        collected.append(line)

    process.wait()
    return process.returncode, "".join(collected)


def print_table(rows: list[dict[str, str]]) -> None:
    headers = ["Step", "Status", "Successes", "AvgReward", "AvgSteps", "Checkpoint"]
    widths = [8, 10, 18, 12, 10, 42]

    def line(ch: str = "-") -> str:
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(f"{v:<{w}}"[:w] for v, w in zip(values, widths)) + " |"

    print("\n" + line("="))
    print(fmt(headers))
    print(line("="))
    for r in rows:
        print(
            fmt(
                [
                    r["step"],
                    r["status"],
                    r["successes"],
                    r["avg_reward"],
                    r["avg_steps"],
                    r["checkpoint"],
                ]
            )
        )
        print(line("-"))


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    eval_config_path = Path(args.eval_config).expanduser().resolve()

    if not eval_config_path.is_file():
        raise FileNotFoundError(f"Eval config not found: {eval_config_path}")

    ckpts = find_checkpoints(run_dir)
    if not ckpts:
        raise RuntimeError(f"No valid checkpoints found under: {run_dir / 'checkpoints'}")

    base_cfg = load_json(eval_config_path)

    print(f"Found {len(ckpts)} checkpoints in: {run_dir}")
    print(f"Using eval config: {eval_config_path}")

    rows: list[dict[str, str]] = []
    n_ok = 0
    n_fail = 0

    for step, ckpt_path in ckpts:
        temp_cfg = write_temp_eval_config(base_cfg, str(ckpt_path))
        print(f"\n[{step}] Evaluating: {ckpt_path}")
        t0 = time.time()

        try:
            returncode, merged_output = run_one(args.python, temp_cfg, args.quiet_eval)
        finally:
            temp_cfg.unlink(missing_ok=True)

        dt = time.time() - t0

        if returncode == 0:
            success_rate, avg_reward, avg_steps = extract_summary(merged_output)
            rows.append(
                {
                    "step": str(step),
                    "status": "OK",
                    "successes": success_rate,
                    "avg_reward": avg_reward,
                    "avg_steps": avg_steps,
                    "checkpoint": str(ckpt_path),
                }
            )
            n_ok += 1
            print(f"[{step}] Done ({dt:.1f}s)")
        else:
            first_error = ""
            for line in merged_output.splitlines():
                if line.strip():
                    first_error = line.strip()
                    break
            if not first_error:
                first_error = f"exit code {returncode}"

            rows.append(
                {
                    "step": str(step),
                    "status": "FAILED",
                    "successes": "-",
                    "avg_reward": "-",
                    "avg_steps": "-",
                    "checkpoint": str(ckpt_path),
                }
            )
            n_fail += 1
            print(f"[{step}] FAILED after {dt:.1f}s: {first_error}")

            if args.stop_on_error:
                break

    print_table(rows)
    print(f"Summary: total={len(rows)}, ok={n_ok}, failed={n_fail}")


if __name__ == "__main__":
    main()
