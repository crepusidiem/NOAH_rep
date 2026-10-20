"""Validate an experiment before releasing a scientific claim; no model execution."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.contracts import ContractError, validate_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--claim", required=True, choices=["prospective", "retrospective", "reconstruction", "paper_faithful"])
    args = parser.parse_args()
    try:
        validate_release(json.loads(args.manifest.read_text()), args.claim, base=args.manifest.parent)
    except (ContractError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"RELEASE BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(f"Release evidence passed for requested claim: {args.claim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
