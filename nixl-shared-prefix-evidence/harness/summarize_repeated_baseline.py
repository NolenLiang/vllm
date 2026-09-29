"""Summarize independent cold-D groups, excluding separately recorded warmups.

Untraced producer metadata describes offered source blocks, not observed READs.
This script does not infer unique destination allocations without worker traces.
"""

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from cold_grpo import METRICS, metric_delta, metric_values


def prefix_hit_delta(before, after):
    pattern = re.compile(r"^vllm:prefix_cache_hits_total(?:\{.*\})?\s+(\S+)")

    def value(text):
        values = [
            float(match.group(1))
            for line in text.splitlines()
            if (match := pattern.match(line))
        ]
        assert values, "Missing local prefix-hit telemetry"
        return sum(values)

    delta = value(after) - value(before)
    assert delta >= 0, "Local prefix-hit counter reset"
    return delta


def distribution(values):
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def producer_overlap(replies, prompt_tokens, block_size=128):
    params = [row[1] for row in replies]
    identity_fields = ("remote_engine_id", "remote_host", "remote_port")
    endpoints = {tuple(p[name] for name in identity_fields) for p in params}
    geometries = {
        tuple(p.get(name, 1) for name in ("tp_size", "dcp_size", "pp_size"))
        for p in params
    }
    source_blocks = [
        (p["remote_engine_id"], group_id, block)
        for p in params
        for group_id, group in enumerate(p["remote_block_ids"])
        for block in group
    ]
    common_prefix = None
    if len(params) > 1 and len(endpoints) == 1 and geometries == {(1, 1, 1)}:
        assert all(len(p["remote_block_ids"]) == 1 for p in params)
        groups = [p["remote_block_ids"][0] for p in params]
        prefix_limit = min((prompt_tokens - 1) // block_size, *(len(g) for g in groups))
        common_prefix = 0
        for block_index in range(prefix_limit):
            if len({group[block_index] for group in groups}) != 1:
                break
            common_prefix += 1
    return {
        "same_producer_endpoint": len(endpoints) == 1,
        "source_geometry": [list(g) for g in sorted(geometries)],
        "distinct_producer_requests": len({p["remote_request_id"] for p in params}),
        "all_siblings_common_full_prefix_blocks": common_prefix,
        "offered_source_block_occurrences": len(source_blocks),
        "unique_offered_source_blocks": len(set(source_blocks)),
        "repeated_offered_source_blocks": len(source_blocks) - len(set(source_blocks)),
        "producer_cached_tokens": [
            p.get("remote_prefill_cached_tokens") for p in params
        ],
        "remote_block_size_fields": [p.get("remote_block_size") for p in params],
        "note": "Producer metadata, not a trace of completed READs; n=1 overlap is null.",
    }


def summarize_case(case_file, expected_groups=10, expected_warmups=2):
    case = json.loads(case_file.read_text())
    config = case["configuration"]
    groups = case["groups"]
    warmups = case["warmups"]
    assert len(groups) == expected_groups, (case_file, len(groups))
    assert len(warmups) == expected_warmups, (case_file, len(warmups))
    assert [g["group_index"] for g in groups] == list(range(expected_groups))
    assert all(g["phase"] == "measurement" for g in groups)
    assert all(g["phase"] == "warmup" for g in warmups)
    all_ids = [g["group_id"] for g in [*warmups, *groups]]
    assert len(set(all_ids)) == len(all_ids), "Warmup/measurement salts overlap"
    assert all(not g["errors"] for g in [*warmups, *groups])
    siblings = config["siblings"]
    records = []
    for group in groups:
        group_dir = case_file.parent / group["group_id"]
        assert json.loads((group_dir / "result.json").read_text()) == group
        requests = group["requests"]
        assert group["successful_requests"] == len(requests) == siblings
        decode_before = (group_dir / "decode-before.prom").read_text()
        decode_after = (group_dir / "decode-after.prom").read_text()
        deltas = metric_delta(metric_values(decode_before), metric_values(decode_after))
        assert deltas == group["decode_metrics_delta"]
        assert all(v is not None for v in deltas.values()), "Telemetry missing/reset"
        assert deltas[METRICS[2]] == deltas[METRICS[3]] == 0
        p_deltas = metric_delta(
            metric_values((group_dir / "prefill-before.prom").read_text()),
            metric_values((group_dir / "prefill-after.prom").read_text()),
        )
        assert p_deltas[METRICS[2]] == p_deltas[METRICS[3]] == 0
        replies = json.loads((group_dir / "producer-replies.json").read_text())
        assert len(replies) == siblings
        assert all(r[0].startswith(group["group_id"]) for r in replies)
        records.append(
            {
                "group_id": group["group_id"],
                "successful_requests": len(requests),
                "transfers": deltas[METRICS[0]],
                "payload_bytes": deltas[METRICS[1]],
                "payload_MiB": deltas[METRICS[1]] / 2**20,
                "decode_local_prefix_hit_tokens": prefix_hit_delta(
                    decode_before, decode_after
                ),
                "d_ttft_median_ms": 1000
                * statistics.median(r["decode_ttft_s"] for r in requests),
                "e2e_ttft_median_ms": 1000
                * statistics.median(r["end_to_end_ttft_s"] for r in requests),
                "d_ttft_per_request_ms": [1000 * r["decode_ttft_s"] for r in requests],
                "producer_overlap": producer_overlap(replies, config["prefix_tokens"]),
            }
        )
    return {
        "case": case_file.parent.name,
        "prompt_tokens": config["prefix_tokens"],
        "siblings": siblings,
        "measured_groups": len(groups),
        "excluded_warmup_groups": len(warmups),
        "successful_measured_requests": sum(r["successful_requests"] for r in records),
        "transfers_per_group": dict(Counter(str(r["transfers"]) for r in records)),
        "decode_local_prefix_hit_tokens_per_group": dict(
            Counter(str(r["decode_local_prefix_hit_tokens"]) for r in records)
        ),
        "payload_MiB_per_group": distribution([r["payload_MiB"] for r in records]),
        "group_median_d_ttft_ms": distribution(
            [r["d_ttft_median_ms"] for r in records]
        ),
        "group_median_e2e_ttft_ms": distribution(
            [r["e2e_ttft_median_ms"] for r in records]
        ),
        "producer_common_prefix_blocks_per_group": dict(
            Counter(
                str(r["producer_overlap"]["all_siblings_common_full_prefix_blocks"])
                for r in records
            )
        ),
        "unique_destination_blocks": None,
        "peak_live_destination_blocks": None,
        "groups": records,
        "source_file": str(case_file),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    assert manifest["trace"] is False
    assert manifest["source_dirty"] is False
    assert manifest["source_sha"] == "0af34418e99b972029b53130c9a9665b4e692536"
    cases = [
        summarize_case(p) for p in sorted(args.input.glob("tokens-*/results.json"))
    ]
    assert {(c["prompt_tokens"], c["siblings"]) for c in cases} == {
        (tokens, siblings) for tokens in (4096, 4097) for siblings in (1, 4, 8)
    }
    ids = [g["group_id"] for case in cases for g in case["groups"]]
    assert len(set(ids)) == len(ids), "Measurement salts overlap across cases"
    summary = {
        "status": "passed",
        "statistical_unit": "one independently salted sibling group",
        "ttft_aggregation": "median and range across ten per-group request medians",
        "destination_block_note": "Not measured without traces; do not infer from bytes",
        "manifest": manifest,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    for case in cases:
        print(json.dumps({k: v for k, v in case.items() if k != "groups"}))


if __name__ == "__main__":
    main()
