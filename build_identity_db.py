from __future__ import annotations

import argparse
from pathlib import Path

from src.identity_database import IdentityDatabase


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean canonical identity DB from verified references")
    parser.add_argument("--source", required=True, help="Existing SQLite DB; never modified")
    parser.add_argument("--output", required=True, help="New canonical SQLite DB")
    parser.add_argument("--max-frame", type=int, default=None, help="Use only enrollment observations before this frame")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing DB: {output}")
    with IdentityDatabase(args.source) as source:
        stats = source.export_canonical(output, max_frame=args.max_frame)
        with IdentityDatabase(output) as target:
            final = target.canonical_stats()
            print(f"source_observations={stats['observations']}")
            print(f"source_identities={stats['identities']}")
            print(f"canonical_observations={final['observations']}")
            print(f"canonical_identities={final['identities']}")
            print(f"model_signatures={final['model_signatures']}")
            print(f"integrity={target.integrity_check()}")


if __name__ == "__main__":
    main()
