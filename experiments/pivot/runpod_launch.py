"""Run the pivot study on one RunPod RTX 4090 pod, mirroring vast_launch.py.

    set RUNPOD_API_KEY=...            (never commit or print it)
    python experiments/pivot/runpod_launch.py start --budget 15 [--spot]
    python experiments/pivot/runpod_launch.py watch --id <pod> --budget 15
    python experiments/pivot/runpod_launch.py stop  --id <pod>

The pod receives only the runner, the prompts and your SSH public key. The
same benchmark-then-run job and the same cost guard as on Vast are used.
With --spot the pod is interruptible; if it is reclaimed, `watch` reports it,
keeps the results synced so far and deletes the pod.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vast_launch as vl  # noqa: E402  (reuses ssh, scp, sync and the remote job)

API = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU = "NVIDIA GeForce RTX 4090"


def api(method, path, body=None):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    req = urllib.request.Request(API + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def pod(pid):
    try:
        p = api("GET", f"/pods/{pid}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    ports = p.get("portMappings") or {}
    p["ssh_host"], p["ssh_port"] = p.get("publicIp"), ports.get("22")
    return p


def destroy(pid):
    try:
        api("DELETE", f"/pods/{pid}")
    except Exception as exc:
        print("delete failed, remove the pod in the console:", exc, flush=True)
    print("deleted", pid, flush=True)


def watch(pid, budget):
    started, checked = time.time(), False
    while True:
        p = pod(pid)
        if p is None:
            sys.exit("pod is gone")
        if p.get("desiredStatus") != "RUNNING":
            print("pod is no longer running (spot reclaim?); results so far are in output/pivot/results", flush=True)
            destroy(pid)
            return
        rate = float(p.get("costPerHr") or 0)
        spent = (time.time() - started) / 3600 * rate
        try:
            names = vl.sync(p)
        except Exception as exc:
            print("sync failed:", exc, flush=True)
            names = []
        print(f"{time.strftime('%H:%M')} files={len(names)} est_spend=${spent:.2f}", flush=True)
        if "ALL_DONE" in names:
            vl.sync(p)
            destroy(pid)
            return
        bench = vl.LOCAL_RESULTS / "bench.json"
        if bench.exists() and not checked:
            checked = True
            proj = json.loads(bench.read_text(encoding="utf-8"))["proj_total_h"] * rate
            print(f"projected total ${proj:.2f} at ${rate:.3f}/h", flush=True)
            if proj > budget:
                print("PROJECTED COST OVER BUDGET; lower --pairs or raise --budget", flush=True)
                destroy(pid)
                return
        if spent >= budget:
            print("BUDGET REACHED", flush=True)
            destroy(pid)
            return
        time.sleep(600)


def start(budget, pairs, spot):
    keyfile = next((k for k in (Path.home() / ".ssh/id_ed25519.pub", Path.home() / ".ssh/id_rsa.pub") if k.exists()), None)
    if keyfile is None:
        sys.exit("no SSH public key in ~/.ssh")
    created = api("POST", "/pods", {
        "name": "pivot-study", "imageName": IMAGE, "gpuTypeIds": [GPU], "gpuCount": 1,
        "cloudType": "COMMUNITY", "interruptible": spot, "containerDiskInGb": 150,
        "ports": ["22/tcp"], "supportPublicIp": True,
        "env": {"PUBLIC_KEY": keyfile.read_text().strip()},
    })
    pid = created["id"]
    print("pod", pid, "spot" if spot else "on-demand", flush=True)
    try:
        deadline = time.time() + 1800
        while True:
            p = pod(pid)
            if p and p.get("ssh_host") and p.get("ssh_port"):
                try:
                    if vl.ssh(p, "nvidia-smi -L").returncode == 0:
                        break
                except subprocess.TimeoutExpired:
                    pass
            if time.time() > deadline:
                raise TimeoutError("pod did not start")
            time.sleep(20)
        here = Path(__file__).resolve().parent
        vl.ssh(p, "mkdir -p /workspace")
        vl.scp_up(p, [here / "pivot_runner.py", here / "prompts_revision_v1.json"])
        vl.ssh(p, vl.REMOTE_SETUP.format(pairs=pairs), timeout=900)
    except Exception:
        destroy(pid)
        raise
    watch(pid, budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "watch", "stop"])
    ap.add_argument("--id")
    ap.add_argument("--budget", type=float, default=15.0)
    ap.add_argument("--pairs", type=int, default=60_000)
    ap.add_argument("--spot", action="store_true", help="interruptible pod (cheaper, may be reclaimed)")
    a = ap.parse_args()
    if a.cmd == "start":
        start(a.budget, a.pairs, a.spot)
    elif a.cmd == "watch":
        watch(a.id, a.budget)
    else:
        destroy(a.id)


if __name__ == "__main__":
    main()
