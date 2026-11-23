"""Pull all raw data from configured (keyless) sources.

Each source is pulled independently; a failure in one is logged and the
others still run. The script exits non-zero if any source failed.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def _sources(raw: Path) -> dict[str, Callable[[], object]]:
    from ngforecast.data.acled import fetch_acled
    from ngforecast.data.inec import fetch_results
    from ngforecast.data.macro import fetch_macro
    from ngforecast.data.poverty import fetch_poverty

    return {
        "inec": lambda: fetch_results(raw / "inec"),
        "poverty": lambda: fetch_poverty(raw / "poverty"),
        "macro": lambda: fetch_macro(raw / "macro"),
        "acled": lambda: fetch_acled(raw / "acled"),
    }


def main(only: list[str] | None = None) -> int:
    load_dotenv()
    raw = Path("data/raw")
    logger.info("Pulling raw data into {}", raw)

    failed = []
    for name, fetch in _sources(raw).items():
        if only and name not in only:
            continue
        try:
            fetch()
            logger.info("✓ {}", name)
        except Exception as exc:  # noqa: BLE001 — report every source, then fail
            logger.error("✗ {}: {}", name, exc)
            failed.append(name)

    if failed:
        logger.error("Failed sources: {}", ", ".join(failed))
        return 1
    logger.info("All raw data pulled successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
