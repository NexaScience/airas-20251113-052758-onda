import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------------- #
#                           Helper utilities for logging                            #
# ---------------------------------------------------------------------------------- #

def tee_stream(stream, log_file):
    """Forward a stream to both stdout/stderr and a log file."""

    def _forward():
        for line in iter(stream.readline, ""):
            if line:
                log_file.write(line)
                log_file.flush()
                print(line, end="")
        stream.close()

    t = threading.Thread(target=_forward)
    t.daemon = True
    t.start()
    return t


# ---------------------------------------------------------------------------------- #
#                                  Main logic                                      #
# ---------------------------------------------------------------------------------- #

def run_experiments(config_path: Path, results_root: Path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    run_variations = cfg["run_variations"]

    for run_cfg in run_variations:
        run_id = run_cfg["run_id"]
        print(f"\n========== Running {run_id} ==========")
        run_dir = results_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "images").mkdir(exist_ok=True)

        # Save human-readable config
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(run_cfg, f)
        # Save JSON config for train subprocess
        json_cfg_path = run_dir / "config.json"
        with open(json_cfg_path, "w") as f:
            json.dump(run_cfg, f)

        # Build subprocess command
        cmd = [sys.executable, "-m", "src.train", "--run-config", str(json_cfg_path), "--results-dir", str(run_dir)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Tee stdout/stderr
        stdout_log = open(run_dir / "stdout.log", "w")
        stderr_log = open(run_dir / "stderr.log", "w")
        threads = [
            tee_stream(proc.stdout, stdout_log),
            tee_stream(proc.stderr, stderr_log),
        ]

        proc.wait()
        for t in threads:
            t.join()
        stdout_log.close()
        stderr_log.close()

        if proc.returncode != 0:
            raise RuntimeError(f"Subprocess for {run_id} failed with exit code {proc.returncode}.")

    # After all runs, aggregate
    print("\n========== Aggregating results ==========")
    subprocess.run([sys.executable, "-m", "src.evaluate", "--results-dir", str(results_root)], check=True)


# ---------------------------------------------------------------------------------- #
#                                      CLI                                         #
# ---------------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Master orchestrator for experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run lightweight smoke test variations.")
    group.add_argument("--full-experiment", action="store_true", help="Run full experiment variations.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to save all results & figures.")
    args = parser.parse_args()

    root = Path(args.results_dir)
    root.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        config_path = Path("config") / "smoke_test.yaml"
    else:
        config_path = Path("config") / "full_experiment.yaml"

    run_experiments(config_path, root)


if __name__ == "__main__":
    main()