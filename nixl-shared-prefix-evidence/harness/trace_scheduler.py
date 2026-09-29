"""Diagnostic snapshots of live request allocations, including private tails."""

import json
import os
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class TraceAsyncScheduler(AsyncScheduler):
    def schedule(self, *args, **kwargs):
        result = super().schedule(*args, **kwargs)
        ids = set(self.requests)
        for entry in getattr(self, "_shared_prefix_loads", {}).values():
            ids.add(entry.owner.request_id)
        allocations = {}
        for request_id in sorted(ids):
            groups = self.kv_cache_manager.get_blocks(request_id).blocks
            allocations[request_id] = [
                [block.block_id for block in group if not block.is_null]
                for group in groups
            ]
        if allocations != getattr(self, "_rl_previous_allocations", None):
            row = {
                "event": "live_allocations",
                "pid": os.getpid(),
                "dp_rank": self.parallel_config.data_parallel_rank,
                "allocations": allocations,
            }
            directory = Path(os.environ["RL_NIXL_TRACE_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"scheduler-{os.getpid()}.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            self._rl_previous_allocations = allocations
        return result
