"""The worker's log: stderr, because stdout is the protocol with the Rust side (worker.py)."""

import sys


def log(msg, level="info"):
    print(f"[{level}] {msg}", file=sys.stderr, flush=True)
