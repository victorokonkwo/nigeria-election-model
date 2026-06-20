# ──────────────────────────────────────────────
# Snakemake pipeline: ingest → features → fit → simulate → report
# Run: snakemake --cores 4 all
# ──────────────────────────────────────────────
import yaml
from pathlib import Path

configfile: "config/config.yaml"


DATA_RAW       = Path("data/raw")
DATA_INTERIM   = Path("data/interim")
DATA_PROCESSED = Path("data/processed")
DATA_EXTERNAL  = Path("data/external")
OUTPUTS        = Path("outputs")


rule all:
    input:
        OUTPUTS / "forecasts" / "latest.json",
        OUTPUTS / "figures" / "calibration.png",


# ── Stage 1: Ingest ──────────────────────────
rule pull_data:
    output:
        directory(DATA_RAW / "inec"),
        directory(DATA_RAW / "nbs"),
        directory(DATA_RAW / "cbn"),
        directory(DATA_RAW / "oil"),
        directory(DATA_RAW / "acled"),
        directory(DATA_RAW / "polls"),
    log:
        "logs/pull_data.log",
    shell:
        "python scripts/pull_data.py 2>&1 | tee {log}"


rule clean_data:
    input:
        rules.pull_data.output,
    output:
        directory(DATA_INTERIM),
    log:
        "logs/clean_data.log",
    shell:
        "python -m ngforecast.data.validate --interim {output} 2>&1 | tee {log}"


# ── Stage 2: Features ────────────────────────
rule build_panel:
    input:
        interim=DATA_INTERIM,
        external=DATA_EXTERNAL,
    output:
        panel=DATA_PROCESSED / "panel.parquet",
        W=DATA_PROCESSED / "W.npz",
        strata=DATA_PROCESSED / "strata.parquet",
    log:
        "logs/build_panel.log",
    shell:
        "python scripts/build_panel.py 2>&1 | tee {log}"


# ── Stage 3: Fit models ──────────────────────
rule fit_models:
    input:
        panel=rules.build_panel.output.panel,
        W=rules.build_panel.output.W,
        strata=rules.build_panel.output.strata,
    output:
        directory(OUTPUTS / "models"),
    log:
        "logs/fit_models.log",
    shell:
        "python scripts/fit_models.py 2>&1 | tee {log}"


# ── Stage 4: Simulate + Forecast ─────────────
rule run_forecast:
    input:
        models=rules.fit_models.output,
        panel=rules.build_panel.output.panel,
        W=rules.build_panel.output.W,
    output:
        forecast=OUTPUTS / "forecasts" / "latest.json",
    params:
        n_sims=config.get("n_sims", 20000),
        seed=config.get("seed", 2027),
    log:
        "logs/run_forecast.log",
    shell:
        "python scripts/run_forecast.py "
        "--n-sims {params.n_sims} --seed {params.seed} "
        "2>&1 | tee {log}"


# ── Stage 5: Reporting ───────────────────────
rule report:
    input:
        forecast=rules.run_forecast.output.forecast,
    output:
        OUTPUTS / "figures" / "calibration.png",
    log:
        "logs/report.log",
    shell:
        "python -m ngforecast.reporting.figures 2>&1 | tee {log}"
