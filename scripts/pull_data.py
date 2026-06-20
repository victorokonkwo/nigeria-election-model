"""Pull all raw data from configured sources."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def main() -> None:
    load_dotenv()
    raw = Path("data/raw")

    logger.info("Pulling raw data into {}", raw)

    from ngforecast.data.inec import fetch_register, fetch_results

    fetch_results(raw / "inec")
    fetch_register(raw / "inec")

    from ngforecast.data.nbs import fetch_nbs

    fetch_nbs(raw / "nbs")

    from ngforecast.data.cbn import fetch_cbn

    fetch_cbn(raw / "cbn")

    from ngforecast.data.oil import fetch_oil

    fetch_oil(raw / "oil")

    from ngforecast.data.acled import fetch_acled

    fetch_acled(raw / "acled")

    logger.info("All raw data pulled successfully")


if __name__ == "__main__":
    main()
