# October 11 integration validation

Published source: `268d488c9b1d04dfbea1128b8101f20079e488d2`.
Source tree: `dbd1674bf724e9fc40290c9875034d3964fac576`.
Parent: #57418 at `acc247941a7bd7eb41cebde9a97eb89dbddc3a52`.
Main: `26a488ad93d12c658449a19dea2eb78f1be38f83`.
Review base: `58980d253b403ec68300560dcec2c1da25ae973c` (parent plus main).
[NIXL-only diff](https://github.com/NolenLiang/vllm/compare/58980d253b403ec68300560dcec2c1da25ae973c...268d488c9b1d04dfbea1128b8101f20079e488d2).

## What changed

The NIXL attach callback remains before the parent's waiting transition.
The parent's `skip_request` performs the queue removal and places readers
in the KV-holding queue; a second pop would remove a different request or
access an empty queue. The callback still releases the follower's independent
unused producer lease.

The existing NIXL lifecycle test now checks queue placement and all KV-fetch
stage counts through cancellation, completion and scheduling. The receive
failure test covers both recompute and fail policies, including an aborted
owner. Other integration conflicts combine existing fixture fields and imports.

## Results

| Selection | Passed |
| --- | ---: |
| Remote-prefill lifecycle and NIXL heartbeat | 109 |
| CPU-offload scheduler, prefix caching, KV-load recovery, admission control | 649 |
| Selected NIXL worker receive/polling/failure/handshake cases | 30 |
| Two selected HMA scheduler cases | 2 |
| Total distinct tests | 790 |

All applicable pre-commit hooks passed on 19 changed files, including the
current Python 3.11-target mypy hook, ruff, test tethering and SPDX checks.
Whitespace checking against the recorded main passed.

## Reproduction

Use a configured Linux CUDA vLLM test runtime with the matching native
extensions, Ray, model-config cache and test dependencies. Checkout the
published source above and use `uv` to manage its `.venv`.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py \
  tests/v1/kv_connector/unit/test_nixl_heartbeat.py \
  -q --maxfail=3 --confcutdir=tests/v1

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
  tests/v1/core/test_prefix_caching.py \
  tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py \
  tests/v1/engine/test_admission_control.py \
  -q --maxfail=3 --confcutdir=tests/v1

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_nixl_connector.py \
  -k 'recv_failure or transfer_failure_logging or failed_request_skips_kv_postprocessing or mixed_memory_read or poll_exception or kv_transfer_handshake or empty_recv_is_reported_only_when_awaited' \
  -q --maxfail=3

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_nixl_connector_hma.py::test_update_state_after_alloc_tracks_cached_blocks_per_group \
  tests/v1/kv_connector/unit/test_nixl_connector_hma.py::test_full_local_hit_is_not_awaited_by_the_scheduler \
  -q --maxfail=3

.venv/bin/python -m pre_commit run --files <19 changed files>
git diff 26a488ad93d12c658449a19dea2eb78f1be38f83 --check
```

The worker selections load the normal root fixtures; the two larger selections
use the recorded confcutdir. Initial attempts with missing local extensions,
Ray or fixture scope did not complete and are not counted as passes.

## Environment and evidence boundary

Execution used WSL2 x86_64, Python 3.12.3 and Torch 2.13.0+cu132 with an RTX
4060 Laptop 8 GiB visible. Existing installed native extensions were reused;
their build provenance was not re-established. These are focused Python
regressions, using mocked transport boundaries, not actual NIXL peer-transfer
or model-output acceptance tests.

Pre-commit ran in a Linux-filesystem checkout of the same source to avoid slow
cross-filesystem Git scans. Its formatting hooks made no changes. The published
tree was compared to the tested source and matches byte-for-byte for tracked
files. Logs are included alongside this report; local workspace/runtime paths
are replaced by `<SOURCE>` and `<RUNTIME>`.

Earlier dual-GB200 transfer, output and failure results remain tied to their
original commits. That environment is no longer accessible. Final-source
dual-GPU lifecycle/model-output validation, default-progress heartbeat-crossing
evidence, human review/personal test execution of the adaptation and upstream
CI remain outstanding. No new latency or hardware performance claim is made.

AI assistance: Codex prepared the adaptation, ran these checks and wrote this
report. These executions do not attest to personal human test execution.
