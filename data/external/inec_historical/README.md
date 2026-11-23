# Nigerian presidential results by state, 1999–2023 (curated)

`results_long.csv` (year, state_name, party, votes) and `state_totals.csv`
(year, state_name, registered, accredited, valid, rejected) cover 1999, 2003,
2011, 2015, 2019 and 2023 for all 36 states + FCT. **2007 is absent: INEC never
published a state-level breakdown.**

Sources, archive snapshots, OCR corrections and known data-quality issues are in
`sources.md`; reconciliation against official national totals is in
`validation_log.txt` (every top-party national total matches exactly except
2011 ACN, −11,850, an inconsistency internal to INEC's own files).

`build_scripts/` regenerates the CSVs. The raw downloads they read (~93 MB:
archived INEC pages/CSVs, scanned PDFs, OCR output) are kept out of git under
`data/raw/inec_historical/` (`dl/`, `ocr/`); point `build.py`'s ROOT there.
`ocr.swift` is compiled with `swiftc ocr.swift -o ocr` (macOS Vision OCR).

Retrieved 2026-11-23.
