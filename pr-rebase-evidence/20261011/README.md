# October 11, 2026 PR rebase validation

All eight existing NolenLiang vLLM PR branches were rebased onto main snapshot `fcae5c98870ff2c20659731dce7ba3a3ff0d48aa`, captured when this maintenance started. No new upstream PR was created. Existing Draft states were preserved; no new reviewer requests were sent. The Mamba2 series remains subject to the maintainer's RFC/hold discussion.

Main subsequently advanced to `2339bd39d661965b821bfbe652db8f536e0b3dcd`; that later main is not the tested base. Mergeability is checked separately from test acceptance.

## Final revisions and checks

| PR | Area | Published head | Passed focused cases | Logs |
| --- | --- | --- | ---: | --- |
| [#57539](https://github.com/vllm-project/vllm/pull/57539) | Exact-replay SSD | [`366fd6ef`](https://github.com/NolenLiang/vllm/commit/366fd6efbd04836097ed9a2313e976c4d3505cbd) | 3 | [tests](pr-57539-tests.log), [pre-commit](pr-57539-pre-commit.log) |
| [#55895](https://github.com/vllm-project/vllm/pull/55895) | Pinned SSD configurations | [`eecbd16e`](https://github.com/NolenLiang/vllm/commit/eecbd16ec3bc4999be7b4fa318d19782dfe58699) | 101 | [tests](pr-55895-tests.log), [pre-commit](pr-55895-pre-commit.log) |
| [#57123](https://github.com/vllm-project/vllm/pull/57123) | RMSNorm fp32 epsilon | [`8d82730e`](https://github.com/NolenLiang/vllm/commit/8d82730e2e77afad1cbe200cfe22f596636a2826) | 487 | [tests](pr-57123-tests.log), [pre-commit](pr-57123-pre-commit.log) |
| [#55627](https://github.com/vllm-project/vllm/pull/55627) | Mamba2 replay wiring | [`7e2ba182`](https://github.com/NolenLiang/vllm/commit/7e2ba182d75d1543d248d2adbede36e668868059) | 10 | [tests](pr-55627-tests.log), [pre-commit](pr-55627-pre-commit.log) |
| [#55905](https://github.com/vllm-project/vllm/pull/55905) | Mamba2 decode CUDA graphs | [`f2d74c8f`](https://github.com/NolenLiang/vllm/commit/f2d74c8f52727acd7ab5ad32988cb3b4ee9a0861) | 34 | [tests](pr-55905-tests.log), [pre-commit](pr-55905-pre-commit.log) |
| [#56244](https://github.com/vllm-project/vllm/pull/56244) | Mamba2 gate placement | [`17879116`](https://github.com/NolenLiang/vllm/commit/17879116581f6d4dd14d44a48320b06768cc6b42) | 55 (48 + 7) | [tests](pr-56244-tests.log), [pre-commit](pr-56244-pre-commit.log) |
| [#59243](https://github.com/vllm-project/vllm/pull/59243) | Dynamic TP / IPC refit | [`9405670b`](https://github.com/NolenLiang/vllm/commit/9405670b0515f4d1fa8af5444081cbd386fe57cc) | 43 + 5 subtests | [tests](pr-59243-tests.log), [pre-commit](pr-59243-pre-commit.log) |
| [#59343](https://github.com/vllm-project/vllm/pull/59343) | NIXL shared prefix | [`376f6836`](https://github.com/NolenLiang/vllm/commit/376f6836b497b59ce556b24c568683a9773ce235) | 790 (758 + 30 + 2) | [tests](pr-59343-tests.log), [pre-commit](pr-59343-pre-commit.log) |

Every branch passed all applicable changed-file pre-commit hooks, including mypy and test tethering. Whitespace checks passed. Counts are per PR and overlap across stacked PRs; they must not be summed as distinct tests. Only successful final-source logs are attached. Earlier failed setup/validation attempts are not counted as passes.

## Integration decisions

- #57539, #55895 and #57123 retain their original patches; range-diff reports patch equivalence.
- #55627 retains upstream prefill checkpoint storage and speculative-state geometry in default mode, uses the current decode state-index variable, and registers its determinism test in the existing H100 batch-invariance job. Logprob assertions now narrow optional values required by current typing.
- #55905 combines checkpoint block regathering with the per-group persistent decode-position buffer update, preserving both in one `update_block_table` method. #56244 is restacked on that updated decode branch.
- #59243 retains upstream checksum/config behavior and reuses the current `_make_model_runner` factory during a drained transition. Its existing loader-preservation regression checks that returned runner ownership is retained even on injected loading failure.
- #59343 linearizes the prior integration merges into a rebased parent layer plus the NIXL layer. It retains the tested queue/lifecycle adaptation and original parent authorship credit. The 790 focused regressions were rerun on this rebased tree.

[NIXL-only review comparison](https://github.com/NolenLiang/vllm/compare/3ede47677ab1c43a6f79d1a0926e1b9b6a876378...376f6836b497b59ce556b24c568683a9773ce235).

## Reproduce

Use a configured Linux CUDA runtime, a uv-managed `.venv`, matching native extensions and relevant cached model configurations. The run used WSL2 x86_64, Python 3.12.3, Torch 2.13.0+cu132 and an RTX 4060 Laptop 8 GiB. Existing native extensions were reused; their build provenance was not re-established. Tests ran with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` after retrieving missing real model configurations.

Checkout each published head above, then run its selection:

### #57539

```sh
.venv/bin/python -m pytest tests/kernels/mamba/test_mamba2_exact_replay.py -q --maxfail=3
```

### #55895

```sh
VLLM_BATCH_INVARIANT=1 .venv/bin/python -m pytest tests/kernels/mamba/test_mamba2_ssd_batch_invariance.py tests/kernels/mamba/test_mamba_ssm_ssd.py -q --maxfail=3
```

### #57123

```sh
.venv/bin/python -m pytest tests/kernels/mamba/test_layernorm_gated_compile.py tests/v1/determinism/test_rms_norm_batch_invariant.py -q --maxfail=3
```

### #55627

```sh
VLLM_BATCH_INVARIANT=1 .venv/bin/python -m pytest tests/v1/attention/test_mamba2_batch_invariant.py tests/kernels/mamba/test_mamba2_exact_replay.py -q --maxfail=3
```

### #55905

```sh
VLLM_BATCH_INVARIANT=1 .venv/bin/python -m pytest tests/kernels/mamba/test_mamba2_exact_replay.py tests/kernels/mamba/test_mamba2_ssd_emit.py tests/kernels/mamba/test_mamba2_ssd_batch_invariance.py tests/v1/attention/test_mamba2_batch_invariant.py -q --maxfail=3
```

### #56244

```sh
VLLM_BATCH_INVARIANT=1 .venv/bin/python -m pytest tests/kernels/mamba/test_mamba2_exact_replay.py tests/kernels/mamba/test_mamba2_ssd_emit.py tests/kernels/mamba/test_mamba2_ssd_batch_invariance.py tests/v1/attention/test_mamba2_batch_invariant.py -q --maxfail=3
.venv/bin/python -m pytest tests/v1/attention/test_mamba2_prefill_checkpoints.py tests/v1/attention/test_mamba_metadata_builder.py -q --maxfail=3
```

### #59243

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --noconftest tests/v1/engine/test_layout_transition.py tests/v1/engine/test_layout_checkpoint.py tests/v1/worker/test_gpu_worker_weight_transfer.py -q --maxfail=3
```

### #59343

```sh
.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py tests/v1/kv_connector/unit/test_nixl_heartbeat.py tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py tests/v1/core/test_prefix_caching.py tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py tests/v1/engine/test_admission_control.py --confcutdir=tests/v1 -q --maxfail=3
.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_nixl_connector.py -k 'recv_failure or transfer_failure_logging or failed_request_skips_kv_postprocessing or mixed_memory_read or poll_exception or kv_transfer_handshake or empty_recv_is_reported_only_when_awaited' -q --maxfail=3
.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_nixl_connector_hma.py::test_update_state_after_alloc_tracks_cached_blocks_per_group tests/v1/kv_connector/unit/test_nixl_connector_hma.py::test_full_local_hit_is_not_awaited_by_the_scheduler -q --maxfail=3
```

For every branch, the applicable hooks and whitespace check were:

```sh
mapfile -t changed_files < <(git diff --name-only fcae5c98870ff2c20659731dce7ba3a3ff0d48aa HEAD)
.venv/bin/python -m pre_commit run --files "${changed_files[@]}"
git diff fcae5c98870ff2c20659731dce7ba3a3ff0d48aa HEAD --check
```

## Evidence and review boundary

Each published Git tree was verified equal to its tested local tree. [Source mapping](source-map.json) records original heads, tested local commits, published heads and tree IDs; publication preserves commit order and coauthor credit, with Codex attribution. Commit IDs differ because publication adds attribution metadata. Logs replace local source/runtime paths with placeholders.

Mamba kernel selections execute on the local RTX 4060. NIXL transport boundaries and dynamic-TP collective/unit fixtures do not establish actual two-GPU NIXL lifecycle or DP/TP reconfiguration acceptance. No new model-level end-to-end, multi-GPU, throughput, training-equivalence or GB200 result is claimed. Historical GPU/model measurements remain tied to their original revisions and environments. The previous dual-GB200 environment is no longer accessible.

The human submitter must review conflict resolutions and personally run the relevant tests under AGENTS.md. Earlier first-person human review statements in PR bodies apply to pre-rebase revisions; this maintenance does not attest to new human execution. Maintainer review, RFC alignment and upstream CI remain separate from these local checks. No upstream CI success is claimed.

AI assistance: Codex performed the rebase, conflict resolution, local checks and evidence update.
