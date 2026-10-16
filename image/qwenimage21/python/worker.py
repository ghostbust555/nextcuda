#!/usr/bin/env python3
"""The bridge between nextcuda's Rust image engine and the platform's CUDA engine for Qwen-Image 2.1.

One JSON object per line on stdin, one per line on stdout (stderr is the log):

  {"op": "load", "models": {sdcpp_bin, qwen21_gguf, qwen21_vae, qwen21_llm, qwen21_llm_vision?, qwen21_turbo_lora?},
   "settings": {...}}                                             -> {"event": "loaded", "seconds": s}
  {"op": "generate", "id": n, "prompt", "negative", "width", "height", "steps", "cfg", "seed", "references": [png],
   "mask": png|null, "transparent": bool, "turbo": bool, "init_image": png|null, "strength": f|null, "out": png}
      -> {"event": "progress", "id": n, "step", "total", "phase"} ...
      -> {"event": "done", "id": n, "path": out, "info": {...}}  or  {"event": "error", "id": n, "message": m}
  {"op": "quit"}

A request runs to its end (sd-cli is one process a picture); the Rust side serialises them and cancels between
pictures by not sending the next.
"""

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_image21 import QwenImage21Engine  # noqa: E402


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    engine = QwenImage21Engine()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as e:
            send({"event": "error", "id": None, "message": f"bad request line: {e}"})
            continue
        op = req.get("op")
        if op == "quit":
            return
        if op == "load":
            t0 = time.time()
            try:
                engine.load(req.get("models") or {}, req.get("settings") or {})
                send({"event": "loaded", "seconds": round(time.time() - t0, 2)})
            except Exception as e:  # the files are the operator's to fix
                send({"event": "error", "id": None, "message": str(e)})
            continue
        if op == "generate":
            rid = req.get("id")
            try:
                def progress(step, total, phase, rid=rid):
                    send({"event": "progress", "id": rid, "step": step, "total": total, "phase": phase})

                img, info = engine.generate(
                    req.get("prompt") or "", req.get("negative") or "", req.get("width") or 1024, req.get("height") or 1024,
                    steps=req.get("steps") or 0, cfg=req.get("cfg") or 0.0, seed=req.get("seed") or 0, progress=progress,
                    references=req.get("references") or [], transparent=bool(req.get("transparent")),
                    mask=req.get("mask"), turbo=bool(req.get("turbo")),
                    init_image=req.get("init_image"), strength=req.get("strength"))
                out = req["out"]
                img.save(out)
                send({"event": "done", "id": rid, "path": out, "info": info})
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                send({"event": "error", "id": rid, "message": str(e)})
            continue
        send({"event": "error", "id": req.get("id"), "message": f"unknown op {op!r}"})


if __name__ == "__main__":
    main()
