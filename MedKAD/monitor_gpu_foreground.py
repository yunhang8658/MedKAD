"""Run a training command in the foreground while sampling one GPU with nvidia-smi.

The training process is a child of this foreground command; no nohup or
detached training job is created. A CSV and JSON summary are written beside
the normal training log. Sampling stops when training exits or is interrupted.
"""

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FIELDS = (
    "sampled_at_utc",
    "gpu_index",
    "utilization_gpu_pct",
    "memory_used_mib",
    "memory_total_mib",
)


def parse_query_line(line: str) -> tuple[str, float | None, float, float]:
    parts = [part.strip() for part in next(csv.reader([line]))]
    if len(parts) != 4:
        raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
    gpu_index, utilization, used, total = parts
    util = None if utilization == "N/A" else float(utilization)
    used_mib = float(used)
    total_mib = float(total)
    if not math.isfinite(used_mib) or not math.isfinite(total_mib) or total_mib <= 0:
        raise ValueError(f"Invalid GPU memory values: {line!r}")
    return gpu_index, util, used_mib, total_mib


def query_gpu(gpu_id: str) -> tuple[str, float | None, float, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu_id,
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected one GPU row for {gpu_id!r}, got {len(lines)}")
    return parse_query_line(lines[0])


def summarize(rows: list[dict], return_code: int, sampling_errors: int) -> dict:
    memory = [row["memory_used_mib"] for row in rows]
    totals = [row["memory_total_mib"] for row in rows]
    utilization = [row["utilization_gpu_pct"] for row in rows if row["utilization_gpu_pct"] is not None]
    active = [value for value in utilization if value >= 5]
    return {
        "return_code": return_code,
        "samples": len(rows),
        "sampling_errors": sampling_errors,
        "gpu_index": rows[0]["gpu_index"] if rows else None,
        "initial_memory_used_mib": memory[0] if memory else None,
        "peak_memory_used_mib": max(memory) if memory else None,
        "memory_total_mib": totals[0] if totals else None,
        "peak_memory_fraction": max(memory) / totals[0] if memory else None,
        "mean_gpu_utilization_pct": statistics.mean(utilization) if utilization else None,
        "mean_active_gpu_utilization_pct": statistics.mean(active) if active else None,
        "active_samples": len(active),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="GPU sample CSV path")
    parser.add_argument("--gpu", required=True, help="Physical GPU index for nvidia-smi -i")
    parser.add_argument("--interval", type=float, default=2.0, help="Sampling interval in seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Training command after --")
    args = parser.parse_args()
    if args.interval < 0.5:
        parser.error("--interval must be at least 0.5 seconds")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a training command after --")
    if shutil.which("nvidia-smi") is None:
        parser.error("nvidia-smi is unavailable; refusing to start unmonitored training")
    try:
        first = query_gpu(args.gpu)
    except (subprocess.SubprocessError, ValueError) as exc:
        parser.error(f"GPU preflight failed; training was not started: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(".summary.json")
    rows = []
    sampling_errors = 0
    print(f"GPU samples: {args.output}", flush=True)
    print(f"GPU summary: {summary_path}", flush=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()

        def record(sample):
            gpu_index, util, used, total = sample
            row = {
                "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_index": gpu_index,
                "utilization_gpu_pct": util,
                "memory_used_mib": used,
                "memory_total_mib": total,
            }
            rows.append(row)
            writer.writerow(row)
            output.flush()

        record(first)
        process = subprocess.Popen(command)
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=args.interval)
                except subprocess.TimeoutExpired:
                    try:
                        record(query_gpu(args.gpu))
                    except (subprocess.SubprocessError, ValueError) as exc:
                        sampling_errors += 1
                        print(f"[gpu-monitor] sample failed: {exc}", file=sys.stderr, flush=True)
        except KeyboardInterrupt:
            # Ctrl-C reaches the foreground training process too. If not, ask
            # its exact PID to stop and wait for its own cleanup trap.
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("[gpu-monitor] training cleanup is still running", file=sys.stderr, flush=True)
                process.wait()
            return_code = 130
        else:
            return_code = process.returncode

    summary = summarize(rows, return_code, sampling_errors)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[gpu-monitor] {json.dumps(summary)}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
