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
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
QUERY = ("gpu_name in [RTX_3090,RTX_4090] num_gpus=1 gpu_ram>=23 reliability>0.98 disk_space>150 inet_down>1500 "
         "direct_port_count>4 cuda_vers>=12.4 rentable=true dph_total<0.22 "
         "inet_down_cost<0.005 storage_cost<0.2")
SPOT_QUERY = ("gpu_name in [RTX_3090,RTX_4090,RTX_A5000,RTX_A6000,A40] num_gpus=1 gpu_ram>=23 "
              "reliability>0.98 disk_space>150 inet_down>300 direct_port_count>4 cuda_vers>=12.4 rentable=true")
SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=30"]
REMOTE_SETUP = (
    "cd /workspace && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PIVOT_N_PAIRS={pairs} PIVOT_ARMS=baseline,random_drop,drop_chosen_only,flip_chosen_only setsid nohup sh -c '"
    ": PIVOT_JOB; pip uninstall -y -q torchvision torchaudio; "
    "pip install -q \"torch>=2.6,<2.7\" \"transformers>=4.57,<6\" accelerate \"jinja2>=3.1.6\" pyarrow huggingface_hub bitsandbytes "
    "&& python pivot_runner.py bench --out results && python pivot_runner.py all --out results' "
    ">> run.log 2>&1 < /dev/null &"
)


def vast(*args, timeout=120, tries=5):
    for k in range(tries):  # the CLI sometimes returns nothing; retry before failing
        try:
            r = subprocess.run(["vastai", *args, "--raw"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            print("vastai timed out, retrying", flush=True)
            time.sleep(15 * (k + 1))
            continue
        starts = [i for i in (r.stdout.find("["), r.stdout.find("{")) if i >= 0]
        if starts:
            return json.loads(r.stdout[min(starts):])
        print("vastai returned no JSON, retrying:", r.stderr.strip()[-200:], flush=True)
        time.sleep(15 * (k + 1))
    raise RuntimeError(f"vastai {' '.join(args[:2])} failed")


def instance(iid):
    return next((i for i in vast("show", "instances") if i["id"] == iid), None)


def ssh_args(inst):
    return ["-p", str(inst["ssh_port"]), f"root@{inst['ssh_host']}"]


def start_job(inst, pairs):
    """Start the detached job; ssh may not return even though the job is running."""
    try:
        ssh(inst, REMOTE_SETUP.format(pairs=pairs), timeout=60)
    except subprocess.TimeoutExpired:
        pass


def ssh(inst, cmd, timeout=120):
    return subprocess.run(["ssh", *SSH_OPTS, *ssh_args(inst), cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


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
        if inst.get("actual_status") != "running":
            print(f"{time.strftime('%H:%M')} instance {inst.get('actual_status')} (spot outbid?); waiting", flush=True)
            time.sleep(300)
            continue
        try:
            alive = ssh(inst, "pgrep -f '[P]IVOT_JOB' >/dev/null && echo up || ls /workspace/results/ALL_DONE").stdout
            if "up" not in alive and "ALL_DONE" not in alive and checked is not None:
                print("job not running; restarting (resumes from finished files)", flush=True)
                start_job(inst, PAIRS)
        except Exception as exc:
            print("liveness check failed:", exc, flush=True)
        spent = (time.time() - started) / 3600 * rate
        try:
            names = sync(inst)
        except Exception as exc:  # a flaky connection must not stop the budget guard
            print("sync failed:", exc, flush=True)
            names = []
        print(f"{time.strftime('%H:%M')} files={len(names)} est_spend=${spent:.2f}", flush=True)
        if "ALL_DONE" not in names and sum(n.startswith("gen_cf_") and n[-8:-6] in ("s0", "s1") for n in names) >= N_TRAIN_RUNS:
            # seeds 0 and 1 complete (amended scope); stop before seed 2 starts
            ssh(inst, "pkill -f '[P]IVOT_JOB'; pkill -f pivot_runner; touch /workspace/results/ALL_DONE; true")
            names = sync(inst)
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


PAIRS = 60_000
N_TRAIN_RUNS = 8  # 4 arms x seeds 0 and 1


def start(budget, pairs, spot=False):
    global PAIRS
    stale = LOCAL_RESULTS / "bench.json"
    if stale.exists():  # a benchmark from an earlier instance must not drive the cost guard
        stale.rename(stale.with_name(f"bench_old_{int(time.time())}.json"))
    PAIRS = pairs
    extra = []
    if spot:
        offers = vast("search", "offers", SPOT_QUERY, "--type", "bid", "-o", "min_bid")
        offer = offers[0]
        bid = round(offer["min_bid"] * 1.5 + 0.01, 3)
        extra = ["--bid_price", str(bid)]
        print("spot offer", offer["id"], offer["gpu_name"], "bid", bid, flush=True)
    else:
        offers = vast("search", "offers", QUERY, "-o", "dph_total")
        offer = offers[0]
        print("offer", offer["id"], offer["dph_total"], flush=True)
    created = vast("create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", "150",
                   "--ssh", "--direct", "--label", "pivot-study", *extra)
    if not created.get("success") and not created.get("new_contract"):
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
        start_job(inst, pairs)
    except Exception:
        destroy(iid)
        raise
    watch(iid, budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "watch", "stop"])
    ap.add_argument("--id", type=int)
    ap.add_argument("--budget", type=float, default=15.0)
    ap.add_argument("--spot", action="store_true", help="interruptible bid instance; job resumes after outbid")
    ap.add_argument("--pairs", type=int, default=60_000, help="pairs per DPO run (protocol: 60000)")
    ap.add_argument("--study", choices=["main", "strong", "dose", "dose12"], default="main",
                    help="strong: enriched baseline vs flip at 50%% share; dose: same at 25%% share")
    a = ap.parse_args()
    if a.study in ("strong", "dose", "dose12"):
        global REMOTE_SETUP, LOCAL_RESULTS, N_TRAIN_RUNS
        share = {"strong": "0.5", "dose": "0.25", "dose12": "0.12"}[a.study]
        REMOTE_SETUP = REMOTE_SETUP.replace(
            "PIVOT_ARMS=baseline,random_drop,drop_chosen_only,flip_chosen_only",
            "PIVOT_ARMS=enriched_baseline,enriched_flip PIVOT_SEEDS=0,1,2 PIVOT_LR=2.5e-6 "
            f"PIVOT_SKIP_ATLAS=1 PIVOT_ENRICH_SHARE={share}")
        LOCAL_RESULTS = ROOT / "output" / "pivot" / {"strong": "results_strong", "dose": "results_dose25", "dose12": "results_dose12"}[a.study]
        N_TRAIN_RUNS = 10 ** 9  # run_all writes ALL_DONE after all six runs
    if a.cmd == "start":
        start(a.budget, a.pairs, a.spot)
    elif a.cmd == "watch":
        global PAIRS
        PAIRS = a.pairs
        watch(a.id, a.budget)
    else:
        destroy(a.id)


if __name__ == "__main__":
    main()
