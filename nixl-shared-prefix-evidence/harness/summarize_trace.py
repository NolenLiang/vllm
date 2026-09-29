"""Summarize actual submitted READ destinations for one isolated GRPO group."""

import argparse
import collections
import json
from pathlib import Path


def summarize(rows, group_id, allocation_rows=()):
    rows = [row for row in rows if group_id in row["request_id"]]
    usable = [
        row
        for row in rows
        if row["effective_block_ids_supported"]
        and row["read_returned_success"]
        and row["new_tracked_handles"] > 0
    ]
    destinations = set()
    sources = collections.Counter()
    plans = collections.Counter()
    for row in usable:
        for group, blocks in enumerate(row["local_block_ids"]):
            destinations.update(
                (row["engine_id"], row["rank"], group, b) for b in blocks
            )
        for group, blocks in enumerate(row["remote_block_ids"]):
            sources.update(
                (row["source_engine_id"], row["source_rank"], group, b) for b in blocks
            )
        plans[
            (
                row["source_engine_id"],
                row["source_rank"],
                tuple(tuple(b) for b in row["remote_block_ids"]),
            )
        ] += 1
    complete = bool(rows) and all(row["effective_block_ids_supported"] for row in rows)
    allocation_peaks = []
    for row in allocation_rows:
        blocks = {
            (group, block)
            for request_id, groups in row["allocations"].items()
            if group_id in request_id
            for group, ids in enumerate(groups)
            for block in ids
        }
        if blocks:
            allocation_peaks.append(len(blocks))
    return {
        "group_id": group_id,
        "complete_block_attribution": complete,
        "read_calls_with_new_handles": len(usable),
        "unique_destination_gpu_blocks": len(destinations) if complete else None,
        "peak_live_request_gpu_blocks": max(allocation_peaks)
        if allocation_peaks
        else None,
        "redundant_source_block_reads": sum(v - 1 for v in sources.values())
        if complete
        else None,
        "duplicate_full_read_plans": sum(v - 1 for v in plans.values())
        if complete
        else None,
        "note": "Submission trace, not completion proof. Correlate successful transfer "
        "telemetry and request outcomes; failures invalidate performance claims. "
        "Source physical identity detects overlap, not all token-equivalent data.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--group-id", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for path in args.trace_dir.glob("worker-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    allocations = [
        json.loads(line)
        for path in args.trace_dir.glob("scheduler-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    print(json.dumps(summarize(rows, args.group_id, allocations), indent=2))
