"""Internal data utilities — config loading, vintage stamping."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_source_config(source_name: str) -> dict:
    """Load a single source block from config/data_sources.yaml."""
    cfg_path = Path("config/data_sources.yaml")
    with open(cfg_path) as f:
        all_sources = yaml.safe_load(f)
    if source_name not in all_sources:
        raise KeyError(f"Source '{source_name}' not found in {cfg_path}")
    return all_sources[source_name]


def stamp_vintage(directory: Path, vintage: str) -> None:
    """Write a vintage metadata file into the given directory."""
    meta = {
        "vintage": vintage,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
    }
    (directory / "_vintage.json").write_text(json.dumps(meta, indent=2))


def download(url: str, dest: Path, timeout: float = 120.0) -> dict:
    """Download ``url`` to ``dest`` and return provenance (url, sha256, bytes)."""
    import hashlib

    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    dest.write_bytes(resp.content)
    return {
        "file": dest.name,
        "url": url,
        "sha256": hashlib.sha256(resp.content).hexdigest(),
        "bytes": len(resp.content),
    }


def hdx_resource_urls(dataset_id: str) -> list[dict]:
    """List the resources (name, url, last_modified) of an HDX dataset."""
    import httpx

    api = f"https://data.humdata.org/api/3/action/package_show?id={dataset_id}"
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        resp = client.get(api)
        resp.raise_for_status()
    return [
        {"name": r["name"], "url": r["url"], "last_modified": r.get("last_modified")}
        for r in resp.json()["result"]["resources"]
    ]


def write_provenance(directory: Path, source: str, files: list[dict]) -> None:
    """Record where and when each raw file came from (replaces a static vintage)."""
    meta = {
        "source": source,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (directory / "_provenance.json").write_text(json.dumps(meta, indent=2))


def load_election_dates(config_path: Path = Path("config/config.yaml")) -> dict[int, str]:
    """{election year: ISO date} from config/config.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {int(y): str(d) for y, d in cfg["election_dates"].items()}
