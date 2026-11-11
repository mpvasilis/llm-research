"""Run the pivot study on one rented Vast.ai RTX 4090 and bring the results home.

    python experiments/pivot/vast_launch.py start --budget 15
    python experiments/pivot/vast_launch.py watch --id <instance> --budget 15
    python experiments/pivot/vast_launch.py stop  --id <instance>

The remote job first runs `bench` (real DPO steps and generation on the rented
GPU). `watch` syncs results every 10 minutes, stops at once if the benchmark
projects a total over the budget, and destroys the instance when the run
finishes or when the estimated spend reaches the budget. Only the runner and
prompts are uploaded; no credentials are sent, since all models are public.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_RESULTS = ROOT / "output" / "pivot" / "results"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
QUERY = ("gpu_name=RTX_4090 num_gpus=1 reliability>0.99 disk_space>150 inet_down>500 "
         "direct_port_count>4 cuda_vers>=12.4 rentable=true")
SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=30"]
REMOTE_SETUP = (
    "cd /workspace && pip install -q 'transformers>=4.57,<6' accelerate pyarrow huggingface_hub bitsandbytes "
    "&& PIVOT_N_PAIRS={pairs} nohup sh -c 'python pivot_runner.py bench --out results && python pivot_runner.py all --out results' "
    "> run.log 2>&1 &"
)


def vast(*args, timeout=120):
    out = subprocess.run(["vastai", *args, "--raw"], capture_output=True, text=True, timeout=timeout).stdout
    return json.loads(out[min(i for i in (out.find("["), out.find("{")) if i >= 0):])


def instance(iid):
    return next((i for i in vast("show", "instances") if i["id"] == iid), None)


def ssh_args(inst):
    return ["-p", str(inst["ssh_port"]), f"root@{inst['ssh_host']}"]


def ssh(inst, cmd, timeout=120):
    return subprocess.run(["ssh", *SSH_OPTS, *ssh_args(inst), cmd], capture_output=True, text=True, timeout=timeout)


def scp_up(inst, paths):
    subprocess.run(["scp", *SSH_OPTS, "-P", str(inst["ssh_port"]), *map(str, paths),
                    f"root@{inst['ssh_host']}:/workspace/"], check=True, timeout=300)


def sync(inst):
    """Copy finished result files; partial files have a .tmp suffix and are skipped."""
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    listing = ssh(inst, "cd /workspace/results && ls").stdout.split()
    for name in listing:
        if name.endswith(".tmp") or (LOCAL_RESULTS / name).exists():
            continue
        subprocess.run(["scp", *SSH_OPTS, "-P", str(inst["ssh_port"]),
                        f"root@{inst['ssh_host']}:/workspace/results/{name}", str(LOCAL_RESULTS / name)],
                       check=True, timeout=600)
    log = ssh(inst, "tail -c 4000 /workspace/run.log").stdout
    (LOCAL_RESULTS.parent / "run.log.tail").write_text(log, encoding="utf-8")
    return listing


def destroy(iid):
    subprocess.run(["vastai", "destroy", "instance", str(iid)], input="y\n", text=True, timeout=120, check=False)
    print("destroyed", iid, flush=True)


def watch(iid, budget):
    started, checked = time.time(), False
    while True:
        inst = instance(iid)
        if inst is None:
            sys.exit("instance is gone")
        rate = inst["dph_total"]
        spent = (time.time() - started) / 3600 * rate
        try:
            names = sync(inst)
        except Exception as exc:  # a flaky connection must not stop the budget guard
            print("sync failed:", exc, flush=True)
            names = []
        print(f"{time.strftime('%H:%M')} files={len(names)} est_spend=${spent:.2f}", flush=True)
        if "ALL_DONE" in names:
            sync(inst)
            destroy(iid)
            return
        bench = LOCAL_RESULTS / "bench.json"
        if bench.exists() and not checked:
            checked = True
            proj = json.loads(bench.read_text(encoding="utf-8"))["proj_total_h"] * rate
            print(f"projected total ${proj:.2f} at ${rate:.3f}/h", flush=True)
            if proj > budget:
                print("PROJECTED COST OVER BUDGET; lower PIVOT_N_PAIRS or raise --budget", flush=True)
                destroy(iid)
                return
        if spent >= budget:
            print("BUDGET REACHED", flush=True)
            destroy(iid)
            return
        time.sleep(600)


def start(budget, pairs):
    offers = vast("search", "offers", QUERY, "-o", "dph_total")
    offer = offers[0]
    print("offer", offer["id"], offer["dph_total"], flush=True)
    created = vast("create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", "150",
                   "--ssh", "--direct", "--label", "pivot-study")
    if not created.get("success"):
        sys.exit(f"could not rent: {created}")
    iid = created["new_contract"]
    print("instance", iid, flush=True)
    try:
        deadline = time.time() + 1800
        while True:
            inst = instance(iid)
            if inst and inst.get("actual_status") == "running" and inst.get("ssh_host"):
                try:
                    if ssh(inst, "nvidia-smi -L").returncode == 0:
                        break
                except subprocess.TimeoutExpired:
                    pass
            if time.time() > deadline:
                raise TimeoutError("instance did not start")
            time.sleep(20)
        here = Path(__file__).resolve().parent
        scp_up(inst, [here / "pivot_runner.py", here / "prompts_revision_v1.json"])
        ssh(inst, REMOTE_SETUP.format(pairs=pairs), timeout=900)
    except Exception:
        destroy(iid)
        raise
    watch(iid, budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "watch", "stop"])
    ap.add_argument("--id", type=int)
    ap.add_argument("--budget", type=float, default=15.0)
    ap.add_argument("--pairs", type=int, default=60_000, help="pairs per DPO run (protocol: 60000)")
    a = ap.parse_args()
    if a.cmd == "start":
        start(a.budget, a.pairs)
    elif a.cmd == "watch":
        watch(a.id, a.budget)
    else:
        destroy(a.id)


if __name__ == "__main__":
    main()
