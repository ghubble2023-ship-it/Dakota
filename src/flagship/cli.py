#!/usr/bin/env python3
"""python -m flagship.cli path/to/image.jpg"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .pipeline import run_flagship


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m flagship.cli IMAGE [IMAGE...]", file=sys.stderr)
        return 2
    code = 0
    for raw in argv:
        path = Path(raw)
        if not path.exists():
            print(json.dumps({"source": raw, "error": "missing"}))
            code = 1
            continue
        report = run_flagship(path)
        print(json.dumps(report.to_dict(), indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
