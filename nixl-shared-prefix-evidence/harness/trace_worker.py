"""Diagnostic worker; use separately from uninstrumented latency measurements.

Pass --worker-cls trace_worker.TraceWorker and set RL_NIXL_TRACE_DIR on D.
Restricts block attribution to matching physical block sizes and no DCP.
"""

import functools
import inspect
import json
import os
import time
from pathlib import Path

import torch.distributed as dist
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.v1.worker.gpu_worker import Worker


def install_trace():
    original = NixlPullConnectorWorker._read_blocks
    if getattr(original, "_rl_trace", False):
        return
    directory = Path(os.environ["RL_NIXL_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    signature = inspect.signature(original)

    @functools.wraps(original)
    def traced(self, *args, **kwargs):
        params = signature.bind(self, *args, **kwargs).arguments
        spec = params["read_spec"]
        req_id = params["request_id"]
        local = [list(group) for group in spec.local_block_ids]
        remote = [list(group) for group in spec.remote_block_ids]
        info = self.transfer_topo.get_engine_info(params["dst_engine_id"])
        supported = (
            self.transfer_topo.block_size_ratio(info.remote_block_size) == 1
            and self.world_size == 1
            and info.remote_tp_size == 1
            and not spec.block_ids_by_region
            and self.dcp_size == 1
            and info.remote_dcp_size == 1
            and self._physical_blocks_per_logical_kv_block
            == info.remote_physical_blocks_per_logical
        )
        if supported:
            local, remote = self._apply_prefix_caching(
                decode_block_ids=local,
                prefill_block_ids=remote,
                decode_physical_per_logical=(
                    self._physical_blocks_per_logical_kv_block
                ),
                prefill_physical_per_logical=info.remote_physical_blocks_per_logical,
            )
        count_before = len(self._recving_transfers.get(req_id, []))
        started = time.perf_counter()
        success = original(self, *args, **kwargs)
        row = {
            "event": "read_submission",
            "request_id": req_id,
            "engine_id": self.engine_id,
            "pid": os.getpid(),
            "rank": dist.get_rank() if dist.is_initialized() else None,
            "source_engine_id": params["dst_engine_id"],
            "source_rank": spec.remote_rank,
            "producer_request_id": params["remote_request_id"],
            "effective_block_ids_supported": supported,
            "local_block_ids": local,
            "remote_block_ids": remote,
            "read_returned_success": bool(success),
            "new_tracked_handles": len(self._recving_transfers.get(req_id, []))
            - count_before,
            "submission_elapsed_s": time.perf_counter() - started,
        }
        with (directory / f"worker-{os.getpid()}.jsonl").open("a") as output:
            output.write(json.dumps(row) + "\n")
        return success

    traced._rl_trace = True
    NixlPullConnectorWorker._read_blocks = traced


class TraceWorker(Worker):
    def __init__(self, *args, **kwargs):
        install_trace()
        super().__init__(*args, **kwargs)
