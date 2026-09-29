"""Protect measurement claims against missing telemetry and rank aliasing."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from cold_grpo import METRICS, decode, metric_delta, metric_values, run
from summarize_repeated_baseline import distribution, prefix_hit_delta, producer_overlap
from summarize_trace import summarize


class MeasurementTests(unittest.TestCase):
    def test_repeated_summary_keeps_group_units_and_actual_source_overlap(self):
        self.assertEqual(distribution([10, 20, 90])["median"], 20)
        self.assertEqual(
            prefix_hit_delta(
                'vllm:prefix_cache_hits_total{engine="0"} 16\n',
                'vllm:prefix_cache_hits_total{engine="0"} 48\n'
                'vllm:prefix_cache_hits_created{engine="0"} 1000\n',
            ),
            32,
        )
        replies = []
        for i, blocks in enumerate(([1, 2, 30], [1, 3, 31])):
            replies.append(
                [
                    f"sibling-{i}",
                    {
                        "remote_engine_id": "producer",
                        "remote_host": "host",
                        "remote_port": 123,
                        "remote_request_id": f"lease-{i}",
                        "remote_block_ids": [blocks],
                        "tp_size": 1,
                        "pp_size": 1,
                        "dcp_size": 1,
                    },
                ]
            )
        result = producer_overlap(replies, prompt_tokens=257)
        self.assertEqual(result["all_siblings_common_full_prefix_blocks"], 1)
        self.assertEqual(result["repeated_offered_source_blocks"], 1)
        self.assertEqual(result["distinct_producer_requests"], 2)
        replies[1][1]["remote_port"] = 456
        self.assertIsNone(
            producer_overlap(replies, 257)["all_siblings_common_full_prefix_blocks"]
        )

    def test_missing_and_reset_counters_are_unknown(self):
        empty = metric_values("# NIXL telemetry disabled\n")
        current = metric_values(f'{METRICS[0]}{{engine="0"}} 7\n')
        self.assertIsNone(metric_delta(empty, current)[METRICS[0]])
        self.assertIsNone(metric_delta(current, {METRICS[0]: 2})[METRICS[0]])

    def test_aggregate_worker_counters_without_counting_buckets(self):
        text = (
            'vllm:nixl_bytes_transferred_bucket{le="100"} 9\n'
            'vllm:nixl_bytes_transferred_count{engine="0"} 3\n'
            'vllm:nixl_bytes_transferred_count{engine="1"} 4\n'
        )
        self.assertEqual(metric_values(text)[METRICS[0]], 7)

    def test_same_block_number_on_two_ranks_is_two_destinations(self):
        row = {
            "request_id": "group-a-0",
            "engine_id": "D",
            "rank": 0,
            "source_engine_id": "P",
            "source_rank": 0,
            "local_block_ids": [[2]],
            "remote_block_ids": [[7]],
            "effective_block_ids_supported": True,
            "read_returned_success": True,
            "new_tracked_handles": 1,
        }
        result = summarize(
            [
                row,
                dict(row, rank=1),
                dict(row, request_id="group-a-1", local_block_ids=[[3]]),
            ],
            "group-a",
        )
        self.assertEqual(result["unique_destination_gpu_blocks"], 3)
        self.assertEqual(result["redundant_source_block_reads"], 2)
        self.assertIsNone(summarize([], "missing")["unique_destination_gpu_blocks"])

    def test_live_allocations_count_shared_prefix_once_and_private_tails(self):
        snapshots = [
            {"allocations": {"group-a-0": [[2, 3, 4]], "unrelated": [[20]]}},
            {
                "allocations": {
                    "group-a-0": [[2, 3, 4]],
                    "group-a-1": [[2, 3, 5]],
                    "unrelated": [[20]],
                }
            },
            {"allocations": {"group-a-1": [[2, 3, 5]]}},
        ]
        result = summarize([], "group-a", snapshots)
        self.assertEqual(result["peak_live_request_gpu_blocks"], 4)
        self.assertIsNone(result["unique_destination_gpu_blocks"])


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmups_use_fresh_salts_and_are_excluded_from_measurement(self):
        prefill_salts = []

        def respond(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1, 2]})
            if request.url.path == "/metrics":
                return httpx.Response(
                    200, text="\n".join(f"{name} 0" for name in METRICS)
                )
            body = json.loads(request.content)
            if request.url.host == "p":
                prefill_salts.append(body["cache_salt"])
                return httpx.Response(
                    200, json={"kv_transfer_params": {"remote_block_ids": [[1]]}}
                )
            return httpx.Response(
                200,
                text='data: {"choices":[{"text":"ok","finish_reason":"length"}]}'
                "\n\ndata: [DONE]\n\n",
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=Path(directory),
                timeout=10,
                siblings=2,
                prefill="http://p",
                decode="http://d",
                model="test-model",
                prefix_tokens=8,
                output_tokens=1,
                prefill_mode="parallel",
                groups=2,
                warmup_groups=2,
                metrics_settle_seconds=12,
            )
            with (
                patch("cold_grpo.httpx.AsyncClient", return_value=client),
                patch("cold_grpo.asyncio.sleep", new_callable=AsyncMock) as settle,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                await run(args)
            result = json.loads((args.output / "results.json").read_text())
        self.assertEqual(len(result["warmups"]), 2)
        self.assertEqual(len(result["groups"]), 2)
        self.assertEqual([g["group_index"] for g in result["groups"]], [0, 1])
        self.assertTrue(all(g["phase"] == "warmup" for g in result["warmups"]))
        self.assertTrue(all(g["phase"] == "measurement" for g in result["groups"]))
        self.assertEqual(len(set(prefill_salts)), 4)
        self.assertEqual(prefill_salts[::2], prefill_salts[1::2])
        self.assertEqual(settle.await_count, 4)
        self.assertTrue(all(call.args == (12,) for call in settle.await_args_list))

    async def test_ttft_ignores_empty_stream_metadata(self):
        text = (
            'data: {"choices":[{"text":"","finish_reason":null}]}\n\n'
            'data: {"choices":[{"text":"hello","finish_reason":null}]}\n\n'
            'data: {"choices":[{"text":"","finish_reason":"length"}]}\n\n'
            "data: [DONE]\n\n"
        )
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=text))
        async with httpx.AsyncClient(transport=transport) as client:
            clock = SimpleNamespace(perf_counter=Mock(side_effect=[1.0, 1.5, 2.0]))
            with patch("cold_grpo.time", clock):
                result = await decode(
                    client, SimpleNamespace(decode="http://D"), {}, "request", 0.0
                )
        self.assertEqual(result["decode_ttft_s"], 0.5)
        self.assertEqual(result["end_to_end_ttft_s"], 1.5)
        self.assertEqual(result["text"], "hello")

    async def test_truncated_stream_is_not_a_successful_sample(self):
        text = 'data: {"choices":[{"text":"partial","finish_reason":null}]}\n\n'
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=text))
        async with httpx.AsyncClient(transport=transport) as client:
            with self.assertRaisesRegex(RuntimeError, "Incomplete generation"):
                await decode(
                    client, SimpleNamespace(decode="http://D"), {}, "request", 0.0
                )


if __name__ == "__main__":
    unittest.main()
