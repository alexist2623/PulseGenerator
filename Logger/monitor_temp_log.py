#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TempLog monitor
---------------
Monitors a dated TempLog.txt file and sends an email alert when the rightmost numeric
value in a new log line exceeds a configured threshold.

Usage:
    python monitor_temp_log.py --config config.json

Config JSON (minimal example):
{
  "base_dir": "D:\\\\Logs",              // parent folder that contains YYYYMMDD subfolders
  "log_filename": "TempLog.txt",         // (optional) default: TempLog.txt
  "threshold": 0.05,                     // numeric threshold
  "recipients": ["you@example.com"],     // list of recipient emails
  "email": {
    "subject": "[ALERT] Temp threshold exceeded: {value}",
    "sender": "Temp Monitor <bot@example.com>",
    "smtp": {
      "host": "smtp.gmail.com",
      "port": 587,
      "username": "your@gmail.com",
      "password": "SMTP_PASS"    // google app password
    }
  },
  "poll_interval_seconds": 1.0,          // (optional) tail poll interval
  "start_from_beginning": false,         // (optional) default: false (start at end-of-file)
  "resend_cooldown_seconds": 300         // (optional) minimum seconds between alert emails
}

You may alternatively provide a fixed absolute path to a single file via:
  "log_file": "D:\\\\Logs\\\\20250908\\\\TempLog.txt"
If "log_file" is present, the program watches exactly that file and does not rotate by date.

The program assumes a dated folder scheme:
  <base_dir>\\YYYYMMDD\\TempLog.txt
and will automatically roll over at midnight to the new day's file if "log_file" is not given.
"""
import argparse
import json
import logging
import os
import re
import smtplib
import sys
import time
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Optional, Tuple

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def now_local_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_config(path: Path) -> dict:
    r"""Setup configuration from JSON file.
    Args:
        path: Path to JSON config file.
    Returns:
        Configuration dictionary.
    """
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    # minimal validation
    if "threshold" not in cfg:
        raise ValueError("Config missing required field: 'threshold'")
    if "recipients" not in cfg or not isinstance(cfg["recipients"], list) or not cfg["recipients"]:
        raise ValueError("Config must include non-empty 'recipients' list")
    if "email" not in cfg or "smtp" not in cfg["email"]:
        raise ValueError("Config must include 'email.smtp' settings")
    # defaults
    cfg.setdefault("log_filename", "TempLog.txt")
    cfg.setdefault("poll_interval_seconds", 1.0)
    cfg.setdefault("start_from_beginning", False)
    cfg.setdefault("resend_cooldown_seconds", 300)
    cfg.setdefault("encoding", "utf-8")
    return cfg

def open_mailer(smtp_cfg: dict) -> smtplib.SMTP:
    r"""
    Open and login to an SMTP server based on config.
    Args:
        smtp_cfg: SMTP configuration dictionary.
    Returns:
        An smtplib.SMTP or smtplib.SMTP_SSL instance.
    """
    host = smtp_cfg.get("host")
    port = int(smtp_cfg.get("port", 587))
    username = smtp_cfg.get("username")
    password = smtp_cfg.get("password")

    if not host:
        raise ValueError("email.smtp.host is required in config")
    if not username:
        raise ValueError("email.smtp.username is required in config")
    if not "password" in smtp_cfg:
        raise ValueError("email.smtp.password or password_env_var is required in config")
    server = smtplib.SMTP(host, port, timeout=30)
    server.starttls()
    try:
        server.login(username, password or "")
    except smtplib.SMTPException as e:
        logging.error("SMTP login failed: %s", e)
        server.quit()
        raise
    return server

def send_email_alert(cfg: dict, value: float, line: str, log_path: Path) -> None:
    r"""
    Send an email alert about threshold exceedance.
    Args:
        cfg: Configuration dictionary.
        value: The numeric value that exceeded the threshold.
        line: The full log line containing the value.
        log_path: Path to the log file being monitored.
    """
    email_cfg = cfg["email"]
    smtp_cfg = email_cfg["smtp"]
    sender = email_cfg.get("sender", "Temp Monitor <no-reply@example.com>")
    subject_tpl = email_cfg.get("subject", "[ALERT] Threshold exceeded: {value}")
    subject = subject_tpl.format(value=value, threshold=cfg["threshold"], file=str(log_path))

    body = (
        f"Fridge1 threshold exceeded at {now_local_str()}.\n\n"
        f"Threshold: {cfg['threshold']} K\n"
        f"Value:     {value} K\n"
        f"File:      {log_path}\n"
        f"Line:      {line.strip()}\n"
    )

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(cfg["recipients"])
    msg["Subject"] = subject
    msg.set_content(body)

    server = None
    try:
        server = open_mailer(smtp_cfg)
        server.send_message(msg)
        logging.info("Alert email sent to %s", cfg["recipients"])
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass

_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.?\d+|\d+\.)(?:[eE][-+]?\d+)?")

def extract_rightmost_float(line: str) -> Optional[float]:
    r"""
    Try to parse the rightmost numeric token in a comma-separated line.
    Example line: "08-09-25, 00:00:02, 1.1613e-02"
    We split by comma and search from the end for a valid float string.
    Args:
        line: A single line of text.
    Returns:
        The parsed float value, or None if no valid float found.
    """
    if not line:
        return None

    # Prefer comma split; fall back to regex search if needed.
    parts = [p.strip() for p in line.strip().split(",")]
    for token in reversed(parts):
        # Quick check: does it look like a float? If not, try regex to find a number inside.
        try:
            return float(token)
        except ValueError:
            m = _FLOAT_RE.search(token)
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    pass

    # As a last resort, look anywhere in the line (right-to-left) for a float.
    matches = list(_FLOAT_RE.finditer(line))
    if matches:
        return float(matches[-1].group(0))
    return None

# ----------------------------- File handling ------------------------------

def compute_log_path(cfg: dict) -> Path:
    r"""
    Compute the current log file path based on config and current date.
    Args:
        cfg: Configuration dictionary.
    Returns:
        Path to the log file to monitor.
    """
    # If a specific file is provided, use it.
    if "log_file" in cfg and cfg["log_file"]:
        return Path(cfg["log_file"])

    base_dir = cfg.get("base_dir")
    if not base_dir:
        raise ValueError("Either 'log_file' or 'base_dir' must be specified in config")

    date_folder = datetime.now().strftime("%Y%m%d")
    return Path(base_dir) / date_folder / cfg.get("log_filename", "TempLog.txt")

def wait_for_file(path: Path, poll_interval: float) -> None:
    r"""
    Wait until the specified file exists.
    Args:
        path: Path to the file.
        poll_interval: Seconds between existence checks.
    """
    while not path.exists():
        logging.info("Waiting for log file to appear: %s", path)
        time.sleep(poll_interval)

def tail_file(f, poll_interval: float):
    """
    Generator that yields new lines appended to file f, similar to 'tail -f'.
    """
    while True:
        where = f.tell()
        line = f.readline()
        if not line:
            time.sleep(poll_interval)
            f.seek(where)
        else:
            yield line

def monitor(cfg: dict) -> None:
    r"""
    Main monitoring loop.
    Args:
        cfg: Configuration dictionary.
    """
    threshold = float(cfg["threshold"])
    poll_interval = float(cfg.get("poll_interval_seconds", 1.0))
    start_from_beginning = bool(cfg.get("start_from_beginning", False))
    cooldown = float(cfg.get("resend_cooldown_seconds", 300))
    encoding = cfg.get("encoding", "utf-8")

    last_alert_ts = 0.0
    current_path = compute_log_path(cfg)
    logging.info("Initial target file: %s", current_path)

    while True:
        # Handle date rollover if using base_dir
        target_path = compute_log_path(cfg)
        if target_path != current_path:
            logging.info("Date rollover detected. Switching file to: %s", target_path)
            current_path = target_path

        wait_for_file(current_path, poll_interval)

        try:
            with current_path.open("r", encoding=encoding, errors="replace") as f:
                if not start_from_beginning:
                    f.seek(0, os.SEEK_END)
                logging.info(
                    "Monitoring file: %s (start_from_beginning=%s)",
                    current_path,
                    start_from_beginning
                )

                for line in tail_file(f, poll_interval):
                    # If date changed while tailing, break to reopen new file
                    new_target = compute_log_path(cfg)
                    if new_target != current_path:
                        logging.info("Date rollover while tailing. Switching to: %s", new_target)
                        current_path = new_target
                        break

                    value = extract_rightmost_float(line)
                    if value is None:
                        logging.debug("No numeric value found in line (skipped): %s", line.strip())
                        continue

                    logging.debug("Parsed value: %s | line: %s", value, line.strip())

                    if value > threshold:
                        now_ts = time.time()
                        if now_ts - last_alert_ts >= cooldown:
                            logging.warning("Threshold exceeded: value=%s > %s", value, threshold)
                            try:
                                send_email_alert(cfg, value, line, current_path)
                                last_alert_ts = now_ts
                            except Exception as e:
                                logging.error("Failed to send alert email: %s", e)
                        else:
                            logging.info(
                                "Threshold exceeded but within cooldown (%.1fs remaining).",
                                cooldown - (now_ts - last_alert_ts)
                            )
        except FileNotFoundError:
            # If file vanished (rotation, cleanup), loop will try again
            logging.info("File not found (may be rotating). Will retry: %s", current_path)
            time.sleep(poll_interval)
        except PermissionError as e:
            logging.warning("Permission error opening file (locked?). Retrying. %s", e)
            time.sleep(poll_interval)
        except Exception as e:
            logging.error("Unexpected error while monitoring: %s", e)
            time.sleep(poll_interval)

def main(argv=None) -> int:
    r"""
    Main entry point.
    Args:
        argv: List of command line arguments (defaults to sys.argv).
    Returns:
        Exit code (0=success, 2=error).
    """
    parser = argparse.ArgumentParser(
        description="Monitor TempLog and email on threshold exceedance."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        return 2

    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        return 2

    logging.info("Starting monitor with config: %s", cfg_path)
    try:
        monitor(cfg)
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
        return 0
    return 0

if __name__ == "__main__":
    sys.exit(main())
