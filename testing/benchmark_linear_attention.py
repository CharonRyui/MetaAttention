"""Compare linear recurrences under the shared Stateful measurement contract.

FLA is a performance baseline, never Meta's oracle. Each JSON line reports real
CPU wall time, public-call CUDA events, Graph replay, preparation, allocator peak
memory, launches, and independent authored-order FP32 recurrence diagnostics.
Legacy --quick, --seqlen and --timing e2e/graph/both options remain supported.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.benchmark_stateful_plans import (  # noqa: E402
    CASES as STATEFUL_CASES,
    CONTRACTS,
    _print_record,
    measure,
)

CASES = tuple(case for case in STATEFUL_CASES if case.name in ("linear", "simple-gla", "gla"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="use one case")
    parser.add_argument("--seqlen", type=int, default=512, help="override sequence length")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--timing", choices=("e2e", "graph", "both"), default="both")
    parser.add_argument("--contract", choices=(*CONTRACTS, "all"), default="output-only")
    parser.add_argument("--json", type=Path, help="write the complete records as JSON")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    contracts = CONTRACTS if args.contract == "all" else (args.contract,)
    if args.seqlen < (2 if "continuation" in contracts else 1) or args.heads < 1 or args.warmup < 0 or args.repetitions < 1:
        parser.error("positive lengths/heads/repetitions and nonnegative warmup required; continuation needs >=2 tokens")
    torch.set_num_threads(1)
    records = []
    for case in CASES[:1] if args.quick else CASES:
        case = replace(case, heads=args.heads)
        for contract in contracts:
            torch.manual_seed(0)
            record = measure(
                case, args.seqlen, torch.device("cuda"), args.warmup,
                args.repetitions, fla=True, contract=contract, timing=args.timing,
            )
            records.append(record)
            _print_record(record)
    if args.json is not None:
        args.json.write_text(json.dumps(records, indent=2))
    if any(record["status"] != "OK" for record in records):
        raise SystemExit("Meta measurement failed; see records (FLA diagnostics do not determine this status)")


if __name__ == "__main__":
    main()
