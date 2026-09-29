Thanks. We now have an **untraced cold-D baseline**, with 10 separately salted groups per case after 2 excluded warmup groups.

Clean main `0af34418e99b972029b53130c9a9665b4e692536`; Qwen3-0.6B BF16 (snapshot `c1899de289a04d12100db370d81485cdf75e47ca`); one P and one D on two same-node GB200 GPUs, TP1/PP1 each; NixlConnector pull, 128-token blocks, APC, FA2/eager, 32 output tokens. Each sibling is an independent request. P requests run concurrently; an explicit barrier waits for all P responses before parallel D submission.

| Prompt tokens | Siblings | Transfers/group | Payload MiB/group | D TTFT ms, median [range] | E2E TTFT ms, median [range] |
|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 1 | 448, all 10 groups | 69.49 [66.69, 71.72] | 115.39 [111.82, 123.95] |
| 4096 | 4 | 4 | 1792, all 10 | 105.83 [102.24, 109.33] | 192.72 [159.33, 200.06] |
| 4096 | 8 | 8 | 546, all 10 | 142.00 [139.52, 143.83] | 213.57 [209.95, 217.18] |
| 4097 | 1 | 1 | 462, all 10 | 60.20 [55.89, 90.81] | 96.05 [87.74, 415.63] |
| 4097 | 4 | 4 | 1848, all 10 | 77.60 [75.66, 87.73] | 143.78 [140.79, 158.69] |
| 4097 | 8 | 8 | 560 in 6 groups; 1008 in 4 | 141.59 [118.64, 146.39] | 213.35 [188.49, 219.03] |

TTFT statistics aggregate the 10 **group-level request medians**, not siblings as independent trials. D TTFT measures D HTTP start to first nonempty text SSE chunk; E2E starts at the corresponding P HTTP request and includes the barrier. All 260 measured D requests succeeded, with zero transfer-failure/lease-expiry counter deltas. MiB measures connector payload, not wire traffic.

**Eight transfers do not mean eight full-prefix copies.** D's local `prefix_cache_hits_total` increased by 27776 cached tokens in every 4096/n8 group, and by 24576 or 28672 in the 4097/n8 groups; it stayed at zero for n1/n4. Existing APC therefore already reduces n8 bytes substantially. Producer metadata confirms a common physical P prefix (31/32 blocks), but describes offered blocks, not completed READs. Benefit depends on arrival/transfer overlap, not simply sibling count.

For the requested destination-block attribution, a **separate single-group traced main diagnostic** found:

| Prompt / siblings | Repeated source-block READ occurrences | Unique D READ blocks | Peak live D KV blocks |
|---|---:|---:|---:|
| 4096 / 1 | 0 | 32 | 33 |
| 4096 / 4 | 93 | 128 | 132 |
| 4097 / 1 | 0 | 33 | 33 |
| 4097 / 4 | 96 | 132 | 132 |

The 93/96 repeats are three additional reads of the common 31/32-block prefix; complete READ plans differ in their private tails. These are diagnostic block counts, not measurements from the untraced TTFT run or total allocated VRAM.

Reproduction with the [published harness and group-level evidence](https://github.com/NolenLiang/vllm/tree/rl/nixl-shared-prefix-evidence-20260929/nixl-shared-prefix-evidence):

```bash
python harness/run_servers.py --model "$MODEL_SNAPSHOT" --source "$VLLM_0af344" \
  --output "$OUT" --groups 10 --warmup-groups 2 --siblings 1 4 8 \
  --prefix-tokens 4096 4097 --prefill-mode parallel
```

The harness sets max model length 4353, max sequences 64, GPU memory utilization 0.3, HND layout and the PyTorch sampler. Runtime: Torch 2.13.0+cu130, NIXL 1.4.1; native extensions from official wheel `70ae0b7435bcb1ceef051df28c50fb7ee7d1dfec`, with the main Python source above.

“Cold” means no matching D prefix at each group's start, not process/JIT cold start or prevention of within-group cache hits. This is a synthetic GRPO-shaped workload. It reproduces n4 prefix redundancy while exposing the n8 APC limitation; it does **not** establish a stable candidate speedup without repeated untraced off/on measurements.
