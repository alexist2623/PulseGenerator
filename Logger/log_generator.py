#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TempLog generator
    python log_generator.py --base-dir "C:\Logs" --interval 5.0 --lines 50
    python log_generator.py --base-dir "C:\Logs"
    python log_generator.py --config config.gen.json

"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

def load_config(path: Optional[Path]) -> dict:
    cfg = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    # defaults
    cfg.setdefault("log_filename", "TempLog.txt")
    cfg.setdefault("interval_seconds", 1.0)
    cfg.setdefault("encoding", "utf-8")
    cfg.setdefault("append", True)
    cfg.setdefault("mean", 0.01)
    cfg.setdefault("std", 0.002)
    cfg.setdefault("spike_prob", 0.1)
    cfg.setdefault("spike_mean", 0.08)
    cfg.setdefault("spike_std", 0.01)
    return cfg

def compute_path(base_dir: Optional[str], log_file: Optional[str], log_filename: str) -> Path:
    if log_file:
        return Path(log_file)
    if not base_dir:
        raise ValueError("Either --base-dir or --log-file must be provided (or via config).")
    date_folder = datetime.now().strftime("%Y%m%d")
    return Path(base_dir) / date_folder / log_filename

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def open_target(path: Path, append: bool, encoding: str):
    ensure_parent(path)
    mode = "a" if append else "w"
    return open(path, mode, encoding=encoding, newline="")

def sample_value(mean: float, std: float, spike_prob: float, spike_mean: float, spike_std: float) -> float:
    if random.random() < spike_prob:
        v = random.gauss(spike_mean, spike_std)
    else:
        v = random.gauss(mean, std)
    return max(0.0, v)

def format_line(ts: datetime, value: float) -> str:
    # DD-MM-YY, HH:MM:SS, 1.2345e-02
    d = ts.strftime("%d-%m-%y")
    t = ts.strftime("%H:%M:%S")
    return f"{d}, {t}, {value:.5e}\n"

def main(argv=None):
    p = argparse.ArgumentParser(description="Generate TempLog-style lines into YYYYMMDD/TempLog.txt")
    p.add_argument("--config", help="JSON Config file path")
    p.add_argument("--base-dir", help="YYYYMMDD folder will be created under this path")
    p.add_argument("--log-file", help="Full log file path (overrides base-dir)")
    p.add_argument("--log-filename", help="Log filename (default TempLog.txt)")
    p.add_argument("--interval", type=float, help="line write interval in seconds")
    p.add_argument("--lines", type=int, default=0, help="number of lines to write before stopping. 0 means unlimited")
    p.add_argument("--duration", type=float, default=0.0, help="maximum duration in seconds before stopping. 0 means unlimited")
    p.add_argument("--append", action="store_true", help="Append to existing file (default)")
    p.add_argument("--truncate", action="store_true", help="Make new file instead of appending")
    p.add_argument("--mean", type=float, help="mean value (e.g. 0.01)")
    p.add_argument("--std", type=float, help="standard deviation (e.g. 0.002)")
    p.add_argument("--spike-prob", type=float, help="spike probability (0.0 to 1.0, e.g. 0.1)")
    p.add_argument("--spike-mean", type=float, help="spike mean value (e.g. 0.08)")
    p.add_argument("--spike-std", type=float, help="spike std (e.g. 0.01)")

    args = p.parse_args(argv)

    cfg = load_config(Path(args.config) if args.config else None)

    # override config with command line args if provided
    if args.base_dir: cfg["base_dir"] = args.base_dir
    if args.log_file: cfg["log_file"] = args.log_file
    if args.log_filename: cfg["log_filename"] = args.log_filename
    if args.interval is not None: cfg["interval_seconds"] = args.interval
    if args.append: cfg["append"] = True
    if args.truncate: cfg["append"] = False
    if args.mean is not None: cfg["mean"] = args.mean
    if args.std is not None: cfg["std"] = args.std
    if args.spike_prob is not None: cfg["spike_prob"] = args.spike_prob
    if args.spike_mean is not None: cfg["spike_mean"] = args.spike_mean
    if args.spike_std is not None: cfg["spike_std"] = args.spike_std

    base_dir = cfg.get("base_dir")
    log_file = cfg.get("log_file")
    log_filename = cfg.get("log_filename", "TempLog.txt")
    interval = float(cfg.get("interval_seconds", 1.0))
    encoding = cfg.get("encoding", "utf-8")
    append = bool(cfg.get("append", True))

    mean = float(cfg.get("mean", 0.01))
    std = float(cfg.get("std", 0.002))
    spike_prob = float(cfg.get("spike_prob", 0.1))
    spike_mean = float(cfg.get("spike_mean", 0.08))
    spike_std = float(cfg.get("spike_std", 0.01))

    target_lines = int(args.lines or 0)
    max_duration = float(args.duration or 0.0)
    start_time = time.time()

    current_path = compute_path(base_dir, log_file, log_filename)
    f = open_target(current_path, append=append, encoding=encoding)
    print(f"[generator] writing to: {current_path} (append={append})")

    wrote = 0
    try:
        while True:
            ts = datetime.now()

            new_path = compute_path(base_dir, log_file, log_filename)
            if new_path != current_path:
                f.close()
                current_path = new_path
                f = open_target(current_path, append=True, encoding=encoding)
                print(f"[generator] rolled over to: {current_path}")

            value = sample_value(mean, std, spike_prob, spike_mean, spike_std)
            line = format_line(ts, value)
            f.write(line)
            f.flush()
            sys.stdout.write(line)
            sys.stdout.flush()

            wrote += 1
            if target_lines and wrote >= target_lines:
                break
            if max_duration and (time.time() - start_time) >= max_duration:
                break

            time.sleep(max(0.0, interval))
    except KeyboardInterrupt:
        print("\n[generator] stopped by user.")
    finally:
        try:
            f.close()
        except Exception:
            pass

if __name__ == "__main__":
    sys.exit(main())
