# NIXL shared-prefix baseline and prototype evidence

This is a reproducibility attachment for a draft follow-up to
[#57418](https://github.com/vllm-project/vllm/pull/57418). It is not an upstream
benchmark suite or a claim that the feature is merge-ready.

Implementation:
[follow-up commit](https://github.com/NolenLiang/vllm/commit/529d59760ff7eed788d31225dc85a800a005db13),
[delta from the parent](https://github.com/NolenLiang/vllm/compare/244daac75b3311df5c5367cd671d21070ca1f309...529d59760ff7eed788d31225dc85a800a005db13).
The evidence branch adds only supporting files to that same product tree.


Results: [repeated untraced baseline and interpretation](REPORT.md). The n=8 baseline already reuses prefixes through ordinary D-side caching; transfer counts alone overstate the optimization opportunity.

## Workload and measurement definitions

Qwen3-0.6B BF16, snapshot `c1899de289a04d12100db370d81485cdf75e47ca`, one P and
one D server on separate GPUs of one node, TP1/PP1, NIXL pull, 128-token blocks,
prefix caching, FlashAttention 2, eager execution and the PyTorch sampler.
The workload generates 32 tokens at temperature 0.8. Each sibling is an
independent request and producer lease; same-group requests share prompt and
cache salt. Salts differ across groups. P requests are concurrent, followed by
an explicit barrier before concurrent D admission.

“Cold” refers to a new logical prefix at D, not cold processes, kernels or all
producer caches. P prefix caching can share physical blocks inside a group.
This is a controlled GRPO-shaped serving workload, not a trainer integration.

Payload bytes come from successful NIXL transfer telemetry, not NIC wire-byte
measurement. Whole READ plans and overlapping physical source prefixes are
different quantities. READ destination-block unions and scheduler-sampled
active-request block peaks are separate; neither is total device memory.
D TTFT is HTTP submission to the first nonempty text chunk. E2E TTFT also
includes P and the group barrier.

## Reproduction

Use a working Python 3.12 vLLM/NIXL CUDA runtime and the pinned model snapshot.
The source argument selects Python code, so independently check compiled
extension compatibility. The recorded runtime used Torch 2.13.0+cu130,
NIXL 1.4.1 and the official native wheel from commit
`70ae0b7435bcb1ceef051df28c50fb7ee7d1dfec`; see `runtime.json`.
Main baseline Python source is `0af34418e99b972029b53130c9a9665b4e692536`.

```bash
mkdir -p "$OUT/tmp" .nixl-rpc
export TMPDIR="$OUT/tmp" VLLM_RPC_BASE_PATH=.nixl-rpc
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python harness/run_servers.py \
  --model "$MODEL_SNAPSHOT" --source "$VLLM_MAIN_CHECKOUT" --output "$OUT" \
  --groups 10 --warmup-groups 2 --siblings 1 4 8 \
  --prefix-tokens 4096 4097 --prefill-mode parallel
```

For a separate block-attribution diagnostic, add `--trace`, choose a different
output directory, and use `--groups 1 --warmup-groups 0 --siblings 1 4`.
Tracing writes synchronously on hot paths and can change both TTFT and the
in-flight overlap window. Do not combine traced and untraced timing samples.
For the experimental implementation, select its checkout and pass
`--kv-extra '{"enable_shared_prefix_loads":true}'` on the command line.

The harness reserves API/side-channel ports, records launch/source settings,
checks deterministic NIXL outputs against local generation, and retains each
request's producer metadata and timing. Warmup groups are separately labeled
and excluded from measured `groups`. The 12-second per-group telemetry wait
is outside request TTFT. Each case's statistical unit is a fresh-salt sibling
group, not each correlated sibling request.

## Validation and limits

The source follow-up passed 87 lifecycle/heartbeat cases and 6 selected
worker/HMA cases (93 distinct cases). The focused 22-case run is included in 87.
These tests use mocked transport boundaries. The real GPU experiment covers
normal-path generation with aligned/unaligned prompts and private tails;
greedy text/finish checks match and token logprob error stays below 0.01.

The single-group traced candidate observation of 4→1 transfers and 75% fewer
payload bytes does not establish untraced sharing hit rates, stable latency
benefits, or whole-device memory savings. A repeated main baseline does not
close that candidate-comparison gap. Real GPU cancellation/failure/reclamation,
unsupported-layout coverage, human review and CI remain pending.

See `review-summary.en.md` for the review record and attribution boundary.
Only selected measurement artifacts are published; private workspace paths
and node names are replaced by placeholders. Runtime credentials, full model
weights, private configuration, and Claude debug/session streams are excluded.
The `.prom` attachments preserve the original selected transfer, expiry and
prefix-hit metric lines, with private path labels replaced; unrelated metric
families are omitted. They suffice for the supplied group-level audit script.
