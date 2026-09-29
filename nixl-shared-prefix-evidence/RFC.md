# [RFC][KV Connector] NIXL support for shared in-flight prefix loads

Draft proposal with an implemented, locally tested follow-up to #57418. The parent author confirmed that NIXL is outside that PR's implementation scope and welcomed a separate follow-up. This proposal depends on its core sharing framework; the parent owns the shared allocation, reference counting and shared receive-failure propagation.

## Problem and measured workload

Concurrent rollout siblings can pull the same physical producer KV prefix into separate decode allocations before the first load becomes a decode prefix-cache hit. A controlled 1P+1D GRPO-shaped workload on Qwen3-0.6B BF16 observes this pattern. Each sibling has an independent producer request and lease. This is a synthetic serving workload, independent of any training framework.

The diagnostic uses fresh cache salts per group and an explicit barrier between concurrent P requests and concurrent D admission. “Cold” means no prior D prefix-cache hit for the group. P can share prefix blocks within the group. It does not represent a cold process/JIT, every possible P batching pattern, or a full RL training run.

The [reproducibility bundle](https://github.com/NolenLiang/vllm/tree/rl/nixl-shared-prefix-evidence-20260929/nixl-shared-prefix-evidence) contains exact revisions, configuration, harness and group-level evidence. Its [baseline report](https://github.com/NolenLiang/vllm/tree/rl/nixl-shared-prefix-evidence-20260929/nixl-shared-prefix-evidence/REPORT.md) covers 60 untraced groups / 260 successful measured requests (52 warmup requests excluded): n=4 transfers 1792/1848 MiB per group for 4096/4097-token prompts, versus 448/462 MiB at n=1. Traced block attribution and repeated untraced timing are separate measurements.

The repeated untraced main baseline also exposes a limit: eight-sibling groups can already reuse much of the prefix through ordinary decode-side caching. Eight completed transfers therefore do not imply eight complete-prefix reads. This feature targets requests overlapping an unfinished compatible load; savings depend on arrival and completion timing rather than sibling count alone.

## Proposed behavior

- Opt in through `kv_connector_extra_config={"enable_shared_prefix_loads": true}`. Initial support is NIXL pull, direct GPU buffers, TP1/PP1/CP1 and one full-attention cache group, with prefix caching. Host staging, bidirectional transfer, speculation and encoder inputs remain outside this scope.
- Keep the owner's actual full-prompt READ unchanged. Share only complete immutable blocks before the last request token. Followers compute their own writable tails locally.
- Require both core prefix-hash isolation and matching producer engine/endpoint, block geometry and leased physical prefix block IDs. Each sibling must supply a distinct producer lease. Older producer metadata without `remote_block_size`, or incompatible configurations, uses the existing independent-load path.
- Notify the producer that each follower's unused lease can be released, without creating a second READ. Keep an aborted owner's READ destination and heartbeat alive until worker completion. This reuses the parent's deferred-reference lifecycle and adds NIXL lease coordination.
- On a shared read failure, reuse the parent's failure propagation and detach/recompute policy. Followers must not reread through leases they already released.

The proposed connector interface adds a slicing capability, a metadata-only compatibility check and a successful-attachment notification. Existing connector defaults preserve the parent's exact partial-prefix behavior. The interface is subject to the parent's maintainer review.

## Implementation and validation

The follow-up is based on #57418 at `244daac75b3311df5c5367cd671d21070ca1f309`. Its focused diff changes ten files: four production files, existing tests and NIXL usage documentation. The [implementation delta](https://github.com/NolenLiang/vllm/compare/244daac75b3311df5c5367cd671d21070ca1f309...529d59760ff7eed788d31225dc85a800a005db13) is available for review; it adds no evidence files to the product tree.

- 93 distinct relevant unit tests passed: 87 lifecycle/heartbeat cases and six worker/HMA cases. The 22-case focused run is a subset of the 87 and is not counted again. These exercise mocked transport boundaries.
- Real NIXL 1P+1D normal-path output checks passed for aligned/unaligned prompts and private follower tails. Deterministic text/finish reasons matched local generation; token logprob absolute error remained below the preset 0.01 tolerance.
- In the traced, same-source off/on diagnostic with four siblings, transfers changed from four to one, payload from 1792 to 448 MiB / 1848 to 462 MiB, and sampled live-request block peaks from 132 to 39 / 132 to 36 for 4096/4097-token prompts. These are measured single-group observations. They do not establish stable latency benefits, whole-device memory savings, or untraced sharing hit rates; tracing can change the overlap window.
- Codex and independent Claude Code reviews are recorded in the reproducibility bundle. Automated review does not attest human review.

Before marking a code PR ready for merge:

- [ ] Exercise real pending READs across owner/all-reader cancellation, heartbeat intervals, transport failure and final producer/destination reclamation.
- [ ] Directly test unsupported local configuration guards, including TP/PP/CP, host staging and bidirectional mode.
- [ ] Measure candidate sharing off/on without tracing, with balanced order and repeated independent groups, before making performance claims.
- [ ] Complete human line-by-line review, relevant personal test execution and upstream CI, as required by the repository's contribution policy.

## Feedback requested

Is this connector capability/compatibility/attachment boundary suitable for #57418's final contract? Is direct-buffer TP1 pull with independent producer leases an appropriate first scope?

Duplicate checks include open PRs referencing #57418 and related NIXL sharing work. #57418 provides the dependency; #58245 concerns CPU-offload sharing across local DP replicas. Neither supplies this NIXL follow-up. This draft does not claim exclusive ownership or maintainer acceptance.

AI assistance: Codex prepared the implementation, tests and measurements; Claude Code independently reviewed the source and evidence. Human review remains pending. This is a design/implementation draft, not a merge-readiness attestation.
