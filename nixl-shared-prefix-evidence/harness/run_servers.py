"""Launch dedicated TP1 1P+1D servers, collect cold groups, reap only our PIDs."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from summarize_trace import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--warmup-groups", type=int, default=0)
    parser.add_argument("--siblings", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--prefill-mode", choices=["parallel", "serial"], default="parallel"
    )
    parser.add_argument("--prefix-tokens", type=int, nargs="+", default=[4096, 4097])
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--kv-extra", type=json.loads, default={})
    parser.add_argument(
        "--attention-config",
        type=json.loads,
        default={"backend": "FLASH_ATTN", "flash_attn_version": 2},
    )
    parser.add_argument("--startup-timeout", type=int, default=600)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    baseline = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    # Avoid fixed-port collisions with other jobs on a shared node. Hold all
    # reservations together so none of our six endpoints gets the same port.
    reserved = []
    try:
        for _ in range(6):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            reserved.append(sock)
        ports = [sock.getsockname()[1] for sock in reserved]
    finally:
        for sock in reserved:
            sock.close()
    api_ports, side_ports, internal_ports = ports[:2], ports[2:4], ports[4:]
    endpoints = [f"http://127.0.0.1:{port}" for port in api_ports]
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(["git", "-C", str(source), "diff"], text=True)
    (output / "source.patch").write_text(diff)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source), str(baseline), env.get("PYTHONPATH", "")]
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["VLLM_KV_CACHE_LAYOUT"] = "HND"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_USE_RUST_FRONTEND"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_LOG_STATS_INTERVAL"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # Record runtime provenance; this does not assert compiled ABI compatibility.
    provenance = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import json, torch, vllm; print(json.dumps(dict(torch=torch.__version__, "
                "vllm=vllm.__version__, vllm_file=vllm.__file__)))"
            ),
        ],
        env=env,
        text=True,
    )
    (output / "runtime.txt").write_text(provenance)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha": revision,
                "source_dirty": bool(diff),
                "model": str(args.model.resolve()),
                "trace": args.trace,
                "measurement_groups_per_case": args.groups,
                "warmup_groups_per_case": args.warmup_groups,
                "prefill_mode": args.prefill_mode,
                "kv_extra": args.kv_extra,
                "attention_config": args.attention_config,
                "flashinfer_sampler": False,
                "python": sys.executable,
                "slurm_job_id": env.get("SLURM_JOB_ID"),
                "ports": {
                    "api": api_ports,
                    "nixl_side_channel": side_ports,
                    "internal": internal_ports,
                },
            },
            indent=2,
        )
    )
    children = []
    logs = []
    try:
        for index, (role, port) in enumerate(
            zip(["kv_producer", "kv_consumer"], api_ports)
        ):
            worker_env = dict(env)
            worker_env["CUDA_VISIBLE_DEVICES"] = str(index)
            worker_env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(side_ports[index])
            worker_env["VLLM_PORT"] = str(internal_ports[index])
            config = {
                "kv_connector": "NixlConnector",
                "kv_role": role,
                "kv_connector_extra_config": args.kv_extra,
            }
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                str(args.model.resolve()),
                "--served-model-name",
                "rl-probe",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--tensor-parallel-size",
                "1",
                "--dtype",
                "bfloat16",
                "--max-model-len",
                str(max(args.prefix_tokens) + 256),
                "--max-num-seqs",
                "64",
                "--gpu-memory-utilization",
                "0.3",
                "--enable-prefix-caching",
                "--block-size",
                "128",
                "--enforce-eager",
                "--attention-config",
                json.dumps(args.attention_config),
                "--kv-transfer-config",
                json.dumps(config),
            ]
            if args.trace and index == 1:
                worker_env["RL_NIXL_TRACE_DIR"] = str(output / "traces")
                command += ["--worker-cls", "trace_worker.TraceWorker"]
                command += ["--scheduler-cls", "trace_scheduler.TraceAsyncScheduler"]
            log = (output / f"{role}.log").open("w")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=worker_env,
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + args.startup_timeout
        with httpx.Client(timeout=2, trust_env=False) as client:
            for port in api_ports:
                while time.monotonic() < deadline:
                    if any(child.poll() is not None for child in children):
                        raise RuntimeError("A server exited; inspect per-role logs")
                    try:
                        response = client.get(f"http://127.0.0.1:{port}/health")
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError(f"Server {port} did not become healthy")
        for tokens in args.prefix_tokens:
            subprocess.run(
                [
                    sys.executable,
                    str(baseline / "validate_outputs.py"),
                    "--prefill",
                    endpoints[0],
                    "--decode",
                    endpoints[1],
                    "--prefix-tokens",
                    str(tokens),
                    "--siblings",
                    str(max(args.siblings)),
                    "--output",
                    str(output / f"correctness-{tokens}.json"),
                ],
                env=env,
                check=True,
            )
            # Keep validation's delayed stats out of the next measured group.
            time.sleep(12)
            for siblings in args.siblings:
                subprocess.run(
                    [
                        sys.executable,
                        str(baseline / "cold_grpo.py"),
                        "--prefill",
                        endpoints[0],
                        "--decode",
                        endpoints[1],
                        "--model",
                        "rl-probe",
                        "--model-revision",
                        args.model.name,
                        "--vllm-revision",
                        revision,
                        "--prefix-tokens",
                        str(tokens),
                        "--siblings",
                        str(siblings),
                        "--prefill-mode",
                        args.prefill_mode,
                        "--groups",
                        str(args.groups),
                        "--warmup-groups",
                        str(args.warmup_groups),
                        "--output",
                        str(output / f"tokens-{tokens}-n-{siblings}"),
                    ],
                    env=env,
                    check=True,
                )
        if args.trace:
            rows = [
                json.loads(line)
                for path in (output / "traces").glob("worker-*.jsonl")
                for line in path.read_text().splitlines()
            ]
            allocations = [
                json.loads(line)
                for path in (output / "traces").glob("scheduler-*.jsonl")
                for line in path.read_text().splitlines()
            ]
            for result_path in output.glob("tokens-*/grpo-*/result.json"):
                result = json.loads(result_path.read_text())
                summary = summarize(rows, result["group_id"], allocations)
                result_path.with_name("trace-summary.json").write_text(
                    json.dumps(summary, indent=2)
                )
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
