# Cold 1P+1D GRPO-shaped NIXL baseline

Clean main Python source `0af34418e99b972029b53130c9a9665b4e692536`, two GB200 GPUs on one NVLink-connected node, Qwen3-0.6B BF16. Full configuration, runtime provenance and reproduction commands are in [README.md](README.md) and [runtime.json](runtime.json).

Each cell contains ten fresh-salt measurement groups after two excluded warmup groups. All 260 measured D requests in 60 groups succeeded; 52 warmup requests and separate deterministic correctness requests are excluded. No measured P/D transfer-failure or lease-expiry counter increments were observed. This run has no hot-path tracing.

| Prompt | Siblings | Transfers/group | Payload MiB/group (groups) | D TTFT ms | E2E TTFT ms |
|---:|---:|---:|---|---|---|
| 4096 | 1 | 1 | 448 (10/10) | 69.49 [66.69, 71.72] | 115.39 [111.82, 123.95] |
| 4096 | 4 | 4 | 1792 (10/10) | 105.83 [102.24, 109.33] | 192.72 [159.33, 200.06] |
| 4096 | 8 | 8 | 546 (10/10) | 142.00 [139.52, 143.83] | 213.57 [209.95, 217.18] |
| 4097 | 1 | 1 | 462 (10/10) | 60.20 [55.89, 90.81] | 96.05 [87.74, 415.63] |
| 4097 | 4 | 4 | 1848 (10/10) | 77.60 [75.66, 87.73] | 143.78 [140.79, 158.69] |
| 4097 | 8 | 8 | 560 (6/10); 1008 (4/10) | 141.59 [118.64, 146.39] | 213.35 [188.49, 219.03] |

TTFT entries are median [min, max] across ten per-group request medians. D TTFT starts at D HTTP submission and ends at the first nonempty text chunk; E2E includes P and the group barrier. The 12-second telemetry settling wait is outside TTFT. These are baseline observations in this controlled setup, not performance gains or cross-machine estimates.

## Existing cache reuse changes the opportunity

For n=4, all ten groups at each prompt length transferred four complete payloads and recorded zero D prefix-hit tokens. For n=8, all groups completed eight transfers, but payload dropped because decode prefix caching was already active within each group. The 4096-token cases recorded 27,776 hit tokens per group (7 × 3968). The 4097-token cases recorded 28,672 hit tokens in six groups and 24,576 in four. Their payloads were respectively 560 and 1008 MiB. These counters establish ordinary local reuse; per-READ sizes were not traced in this run. Eight transfers do not mean eight full-prefix READs. Cold at group start does not guarantee cold admissions throughout the group.

Producer metadata shows the same physical 31/32 full prefix blocks offered to each 4096/4097-token sibling, with independent leases. Offered blocks are not evidence that every offered block was actually transferred. Destination-block unions and active-request block peaks were not collected in this untraced run.

## Separate traced diagnostics

The earlier single-group clean-main diagnostic recorded n=4 READ destination unions of 128/132 blocks and peak live-request allocations of 132/132 blocks for 4096/4097-token prompts; repeated physical source-prefix reads were 93/96. Full READ plans differed in private tails. See [traced-main-audit.json](traced-main-audit.json).

A separate single-group, same-source candidate off/on diagnostic recorded n=4 transfers 4→1, payload 1792→448 / 1848→462 MiB, and live-request block peaks 132→39 / 132→36. See [traced-candidate.json](traced-candidate.json). The 75% payload reduction applies to those traced groups only. Synchronous tracing can change overlap as well as latency. Candidate untraced off/on comparisons remain pending; the repeated main-only baseline does not establish candidate hit rates or speedups.

## Recheck the published evidence

[untraced-summary.json](untraced-summary.json) was regenerated from this public subset and compared with the private full-artifact audit; all case measurements match exactly. Group result JSON, producer metadata and the relevant original Prometheus lines are included, with private paths/hosts replaced consistently.

```bash
python harness/summarize_repeated_baseline.py --input untraced --output summary-recomputed.json
```

See [review-summary.en.md](review-summary.en.md) for independent review and remaining validation gaps.
