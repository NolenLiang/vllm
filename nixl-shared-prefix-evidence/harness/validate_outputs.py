"""Compare synchronized NIXL siblings with ordinary TP1 local generation."""

import argparse
import asyncio
import json
import math
import uuid
from pathlib import Path

import httpx


async def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        response = await client.post(
            args.prefill + "/tokenize",
            json={
                "model": args.model,
                "prompt": "Explain this mathematical argument. ",
                "add_special_tokens": False,
            },
        )
        response.raise_for_status()
        tokens = response.json()["tokens"]
        prompt = (tokens * math.ceil(args.prefix_tokens / len(tokens)))[
            : args.prefix_tokens
        ]
        base = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": 16,
            "temperature": 0,
            "seed": 7,
            "ignore_eos": True,
            "logprobs": 1,
        }

        async def post(url, body, request_id):
            response = await client.post(
                url + "/v1/completions", json=body, headers={"X-Request-Id": request_id}
            )
            response.raise_for_status()
            return response.json()

        group = "validation-" + uuid.uuid4().hex
        reference = await post(
            args.prefill, dict(base, cache_salt=group + "-ref"), group + "-reference"
        )
        prepared = await asyncio.gather(
            *(
                post(
                    args.prefill,
                    dict(
                        base,
                        max_tokens=1,
                        cache_salt=group,
                        kv_transfer_params={
                            "do_remote_decode": True,
                            "do_remote_prefill": False,
                        },
                    ),
                    f"{group}-{i}",
                )
                for i in range(args.siblings)
            )
        )
        params = [r.get("kv_transfer_params") for r in prepared]
        if not all(p and p.get("remote_block_ids") for p in params):
            raise RuntimeError("No usable producer transfer metadata")
        outputs = await asyncio.gather(
            *(
                post(
                    args.decode,
                    dict(base, cache_salt=group, kv_transfer_params=p),
                    f"{group}-{i}",
                )
                for i, p in enumerate(params)
            )
        )
        expected = reference["choices"][0]
        checks = []
        expected_lp = expected["logprobs"]["token_logprobs"]
        for output in outputs:
            actual = output["choices"][0]
            actual_lp = actual["logprobs"]["token_logprobs"]
            comparable = len(actual_lp) == len(expected_lp) and all(
                a is not None and b is not None for a, b in zip(actual_lp, expected_lp)
            )
            errors = (
                [abs(a - b) for a, b in zip(actual_lp, expected_lp)]
                if comparable
                else []
            )
            checks.append(
                {
                    "text_equal": actual["text"] == expected["text"],
                    "finish_equal": actual["finish_reason"]
                    == expected["finish_reason"],
                    "logprobs_comparable": comparable,
                    "max_abs_logprob_error": max(errors) if errors else None,
                    "within_tolerance": comparable
                    and bool(errors)
                    and all(
                        math.isfinite(e) and e <= args.logprob_atol for e in errors
                    ),
                }
            )
        passed = all(
            c["text_equal"] and c["finish_equal"] and c["within_tolerance"]
            for c in checks
        )
        result = {
            "passed": passed,
            "configuration": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "reference": reference,
            "producer_responses": prepared,
            "outputs": outputs,
            "checks": checks,
            "scope": "one fixed checkpoint, TP1 local vs NIXL deterministic generation; "
            "does not by itself prove cancellation/failure behavior",
        }
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({"passed": passed, "checks": checks}), flush=True)
        if not passed:
            raise RuntimeError("NIXL outputs differ from local reference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", default="http://127.0.0.1:18100")
    parser.add_argument("--decode", default="http://127.0.0.1:18200")
    parser.add_argument("--model", default="rl-probe")
    parser.add_argument("--prefix-tokens", type=int, default=4097)
    parser.add_argument("--siblings", type=int, default=4)
    parser.add_argument("--logprob-atol", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
