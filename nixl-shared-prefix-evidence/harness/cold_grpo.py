"""Measure a synchronized, cold-D-prefix GRPO group on dedicated P/D servers.

Uses the upstream toy proxy's P -> kv_transfer_params -> D protocol directly.
No transfer-count reduction is assumed. Run against otherwise idle servers.
"""

import argparse
import asyncio
import json
import math
import re
import statistics
import time
import uuid
from pathlib import Path

import httpx

METRICS = (
    "vllm:nixl_bytes_transferred_count",
    "vllm:nixl_bytes_transferred_sum",
    "vllm:nixl_num_failed_transfers_total",
    "vllm:nixl_num_kv_expired_reqs_total",
)


def metric_values(text):
    result = {}
    for name in METRICS:
        pattern = re.compile(r"^" + re.escape(name) + r"(?:\{.*\})?\s+(\S+)")
        values = [
            float(m.group(1))
            for line in text.splitlines()
            if (m := pattern.match(line))
        ]
        result[name] = sum(values) if values else None
    return result


def metric_delta(before, after):
    result = {}
    for name in METRICS:
        a, b = before.get(name), after.get(name)
        # Missing telemetry and counter resets must not look like zero traffic.
        result[name] = (
            b - a
            if a is not None and b is not None and math.isfinite(b - a) and b >= a
            else None
        )
    return result


async def snapshot(client, base, path):
    response = await client.get(base + "/metrics")
    response.raise_for_status()
    path.write_text(response.text)
    return metric_values(response.text)


async def decode(client, args, body, request_id, started):
    sent = time.perf_counter()
    first = None
    chunks = []
    usage = None
    finished = False
    async with client.stream(
        "POST",
        args.decode + "/v1/completions",
        json=body,
        headers={"X-Request-Id": request_id},
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("text", "")
                if text and first is None:
                    first = time.perf_counter()
                chunks.append(text)
                finished |= choice.get("finish_reason") is not None
    if first is None or not finished:
        raise RuntimeError(f"Incomplete generation for {request_id}")
    return {
        "request_id": request_id,
        "decode_ttft_s": first - sent,
        "end_to_end_ttft_s": first - started,
        "decode_elapsed_s": time.perf_counter() - sent,
        "usage": usage,
        "text": "".join(chunks),
    }


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    headers = {}
    # Local experiment endpoints only; no credentials are persisted.
    async with httpx.AsyncClient(
        timeout=args.timeout,
        trust_env=False,
        headers=headers,
        limits=httpx.Limits(max_connections=max(20, args.siblings * 2)),
    ) as client:
        response = await client.post(
            args.prefill + "/tokenize",
            json={
                "model": args.model,
                "prompt": "Discuss the evidence carefully. ",
                "add_special_tokens": False,
            },
        )
        response.raise_for_status()
        unit = response.json()["tokens"]
        if not unit or not all(isinstance(token, int) for token in unit):
            raise ValueError("Tokenize endpoint did not return integer token IDs")
        prompt = (unit * math.ceil(args.prefix_tokens / len(unit)))[
            : args.prefix_tokens
        ]
        all_results = []
        warmup_results = []
        for iteration in range(args.warmup_groups + args.groups):
            is_warmup = iteration < args.warmup_groups
            group = iteration if is_warmup else iteration - args.warmup_groups
            phase = "warmup" if is_warmup else "measurement"
            group_id = f"{'warmup' if is_warmup else 'grpo'}-{uuid.uuid4().hex}"
            group_dir = args.output / group_id
            group_dir.mkdir()
            before = await snapshot(
                client, args.decode, group_dir / "decode-before.prom"
            )
            p_before = await snapshot(
                client, args.prefill, group_dir / "prefill-before.prom"
            )
            base = {
                "model": args.model,
                "prompt": prompt,
                "n": 1,
                "max_tokens": args.output_tokens,
                "ignore_eos": True,
                "temperature": 0.8,
                "cache_salt": group_id,
            }
            started = {}

            async def prefill(index, group_id=group_id, started=started, base=base):
                request_id = f"{group_id}-{index}"
                started[request_id] = time.perf_counter()
                body = dict(base, stream=False, max_tokens=1, seed=index)
                body["kv_transfer_params"] = {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_engine_id": None,
                    "remote_block_ids": None,
                    "remote_host": None,
                    "remote_port": None,
                }
                reply = await client.post(
                    args.prefill + "/v1/completions",
                    json=body,
                    headers={"X-Request-Id": request_id},
                )
                reply.raise_for_status()
                params = reply.json().get("kv_transfer_params")
                if not params or not params.get("remote_block_ids"):
                    raise RuntimeError("P returned no remote blocks; not a NIXL run")
                return request_id, params

            # Explicit D-admission barrier: all P replies arrive before any D send.
            if args.prefill_mode == "serial":
                prepared = [await prefill(i) for i in range(args.siblings)]
            else:
                prepared = await asyncio.gather(
                    *(prefill(i) for i in range(args.siblings))
                )
            (group_dir / "producer-replies.json").write_text(
                json.dumps(prepared, indent=2)
            )
            replies = await asyncio.gather(
                *(
                    decode(
                        client,
                        args,
                        dict(
                            base,
                            stream=True,
                            stream_options={"include_usage": True},
                            seed=index,
                            kv_transfer_params=params,
                        ),
                        request_id,
                        started[request_id],
                    )
                    for index, (request_id, params) in enumerate(prepared)
                ),
                return_exceptions=True,
            )
            # Engine stats are published periodically, after transfer completion.
            await asyncio.sleep(args.metrics_settle_seconds)
            after = await snapshot(client, args.decode, group_dir / "decode-after.prom")
            p_after = await snapshot(
                client, args.prefill, group_dir / "prefill-after.prom"
            )
            successes = [r for r in replies if not isinstance(r, BaseException)]
            errors = [str(r) for r in replies if isinstance(r, BaseException)]
            delta = metric_delta(before, after)
            entry = {
                "group_id": group_id,
                "group_index": group,
                "phase": phase,
                "successful_requests": len(successes),
                "errors": errors,
                "requests": successes,
                "decode_metrics_delta": delta,
                "prefill_metrics_delta": metric_delta(p_before, p_after),
                "worker_transfers": delta[METRICS[0]],
                "payload_bytes": delta[METRICS[1]],
                "decode_ttft_median_s": statistics.median(
                    [r["decode_ttft_s"] for r in successes]
                )
                if successes
                else None,
                "duplicate_transfers": None,
                "unique_destination_gpu_blocks": None,
                "trace_note": "Use per-rank READ traces for block/duplicate attribution; "
                "telemetry counts alone do not prove duplicate prefixes.",
            }
            (group_dir / "result.json").write_text(json.dumps(entry, indent=2))
            (warmup_results if is_warmup else all_results).append(entry)
            print(
                json.dumps({k: v for k, v in entry.items() if k != "requests"}),
                flush=True,
            )
        config = {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        }
        result = {
            "configuration": config,
            "workload": "one independent request per sibling, shared salted prompt; "
            "P-prefill barrier before simultaneous D admission",
            "cold_definition": "fresh group cache salt on D; process/JIT may be warm",
            "bytes_definition": "successful NIXL payload bytes, not physical wire bytes",
            "warmups": warmup_results,
            "groups": all_results,
        }
        (args.output / "results.json").write_text(json.dumps(result, indent=2))
        if any(r["errors"] for r in [*warmup_results, *all_results]):
            raise RuntimeError("Some requests failed; see recorded results")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", default="http://127.0.0.1:18100")
    parser.add_argument("--decode", default="http://127.0.0.1:18200")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="record-in-server-manifest")
    parser.add_argument("--vllm-revision", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--siblings", type=int, default=4)
    parser.add_argument(
        "--prefill-mode", choices=["parallel", "serial"], default="parallel"
    )
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument(
        "--warmup-groups",
        type=int,
        default=0,
        help="Fresh-salt groups recorded separately and excluded from measurements",
    )
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--metrics-settle-seconds", type=float, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.prefix_tokens, args.siblings, args.groups, args.output_tokens, args.tp)
        < 1
    ):
        parser.error("Sizes must be positive")
    if args.metrics_settle_seconds < 0:
        parser.error("Metrics settlement must be nonnegative")
    if args.warmup_groups < 0:
        parser.error("Warmup group count must be nonnegative")
    args.prefill = args.prefill.rstrip("/")
    args.decode = args.decode.rstrip("/")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
