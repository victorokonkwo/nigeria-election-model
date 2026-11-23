# Sources: Nigerian presidential results by state, 1999–2023

Retrieved 2026-11-23. Each raw file is under `dl/<subfolder>/`. Parsing code is in `scripts/` (`build.py` builds the CSVs; `ocr.swift`, `tile_ocr.sh`, `grid2015.py` and `solve2015.py` handle the scanned tables; `crosscheck.py` compares against Wikipedia). The validation output is in `validation_log.txt`.

Most inecnigeria.org `wp-content` files now return 404 on the live site, which was rebuilt in Nuxt. They were fetched from the Internet Archive using raw `id_` snapshots. Live INEC URLs are given with the snapshot used.

## Outputs
- `results_long.csv`: year, state_name, party, votes
- `state_totals.csv`: year, state_name, registered, accredited, valid, rejected (blank = not available from the sources used)

## 1999 (27 Feb 1999)
- **Data:** AAEA/IFES observation report on the Dec 1998–Feb 1999 transitional elections, p. 76, table "RESULTS (Votes Cast Data from INEC – Analysis by IFES)". The page is a scanned image, read with macOS Vision OCR.
  - https://www.ifes.org/sites/default/files/migrate/r01779.pdf (the live URL returns 403)
  - Snapshot used: https://web.archive.org/web/20240802211336/https://www.ifes.org/sites/default/files/migrate/r01779.pdf
  - Local file: `dl/ifes1999/`
- **Fields:** registered, total valid, PDP (Obasanjo), APP/AD joint ticket (Falae). No accredited or rejected by state. Nationally the table reports 431,611 invalid votes and 30,280,052 votes cast.
- **OCR corrections:** four cells were misread (6/8 confusion), in Abia, Cross River, Ekiti and Niger. Each correction is the only value that satisfies PDP + APP = valid and reproduces the printed percentage column. After the corrections, every row and column total agrees.
- **Official national totals** (same table; also African Elections Database https://africanelections.tripod.com/ng.html): PDP 18,738,154; APP/AD 11,110,287; valid 29,848,441.

## 2003 (19 Apr 2003)
- **Data:** INEC's 2003 results website, one page per state plus a national summary, archived 29 Apr 2003.
  - URL pattern: `http://www.inecnigeria.org/2003elections/results/presidential/<state>.htm`, where `<state>` is e.g. `abia`, `akwa-ibom`, `cross_river` or `fct`. The national page is `summary.htm`.
  - Example snapshot: https://web.archive.org/web/20030429003305/http://www.inecnigeria.org:80/2003elections/results/presidential/abia.htm
  - Local files: `dl/inec2003pres/`
- **Fields:** votes for all 20 parties, total valid, rejected ballots, voters on register. No accredited.
- **Caveat:** the archived Imo page says "26 of 27 LGAs received". The national summary (last updated 22 Apr 2003 19:25) is built from the same state pages, so the state rows still sum exactly to it.
- **Official national totals** (`summary.htm`): PDP 24,456,140; ANPP 12,710,022; APGA 1,297,445; valid 39,480,489; rejected 2,538,246; register 60,823,022.

## 2007 (21 Apr 2007): not included
INEC never published a state-by-state breakdown of the 2007 presidential result.
- TheCable, "INCREDIBLE: INEC does NOT have full official results of 2007 presidential election" (24 Oct 2018): https://www.thecable.ng/incredible-inec-does-not-have-full-official-results-of-2007-presidential-election/
  - Snapshot used: https://web.archive.org/web/20260927132001/https://www.thecable.ng/incredible-inec-does-not-have-full-official-results-of-2007-presidential-election/
  - It says 2007 "is the first and only in Nigeria's history in which there is no state-by-state breakdown".
- National figures only:
  - INEC website (archived): http://web.archive.org/web/20070713082947/www.inecnigeria.org/election/show_index_result.php?ele_id=2804 gives PDP 24,784,227; ANPP 6,607,419; AC 2,567,798.
  - The figures announced on 23 Apr 2007 differ: PDP 24,638,063; ANPP 6,605,299; AC 2,637,848. See African Elections Database, https://africanelections.tripod.com/ng2007presidential.pdf

## 2011 (16 Apr 2011)
- **Data:** INEC results dashboard CSV, archived 25 Apr 2011.
  - http://www.inecnigeria.org/results/force.php?file=inec_presidential_results_37_states.csv
  - Snapshot used: https://web.archive.org/web/20110425031101/http://www.inecnigeria.org:80/results/force.php?file=inec_presidential_results_37_states.csv
  - Local file: `dl/inec2011csv/`
- **Fields:** votes for all 20 parties in all 37 units. The CSV has no registered, accredited or rejected columns, so `valid` is the sum of party votes.
- **National totals:** taken from the INEC dashboard page http://web.archive.org/web/20110501010957/http://www.inecnigeria.org:80/results/presidential/
- **Internal inconsistency in INEC's own data:** the state rows sum exactly to the national figures for PDP, CPC and ANPP. They do not for ACN (−11,850) or for the minor parties ADC, BNPP, PDC and SDMP. Psephos (source: INEC) has the same state figures.

## 2015 (28 Mar 2015)
- **Data:** INEC "2015 Presidential Election, March 28, 2015, Summary of Results", a one-page scanned sheet.
  - https://www.inecnigeria.org/wp-content/uploads/2019/02/summary-of-results.pdf
  - Snapshot used: https://web.archive.org/web/20260815143744/https://www.inecnigeria.org/wp-content/uploads/2019/02/summary-of-results.pdf
  - Local file: `dl/inec2015sum/`
- **Method:** read with tiled macOS Vision OCR. Values were then chosen so that every row passes two checks: the 14 party votes sum to valid votes, and valid + rejected = total votes cast.
- **Fields:** registered, accredited, votes for 14 parties, valid, rejected, total cast.
- **Result of the checks:** all 37 rows pass. Every column sums to the printed national total, with one exception: ADC, where the states sum to 29,666 against a printed 29,665. That is an arithmetic slip on the sheet itself.
- **Gombe:** the OCR value for valid votes was a fragment. It was replaced by the party sum, 460,599, which satisfies 460,599 + 12,845 = 473,444.

## 2019 (23 Feb 2019)
- **Party votes:** Stears Elections, parsed from the `__NEXT_DATA__` JSON embedded in https://www.stears.co/elections/2019/president/ (`partyDominanceData`). Local file: `dl/stears/`.
  - Fields: APC, PDP and total valid votes per state.
  - "Others" is computed as valid − APC − PDP, because Stears' own "Others" field (sum 856,641) disagrees with its state totals.
- **Registered voters:** INEC "Report of the 2023 General Election", Table 8.8, 2019 column (sums to 84,004,084).
  - https://inecnigeria.org/wp-content/uploads/2024/02/2023-GENERAL-ELECTION-REPORT-1.pdf
  - Snapshot used: https://web.archive.org/web/20260214132338/https://inecnigeria.org/wp-content/uploads/2024/02/2023-GENERAL-ELECTION-REPORT-1.pdf
  - Note: the national declaration gives 82,344,107 registered voters "as collated".
- **Official national totals:** INEC declaration of results.
  - https://www.inecnigeria.org/wp-content/uploads/2019/03/2019-GE-PRESIDENTIAL-ELECTION-RESULTS.pdf
  - Snapshot used: https://web.archive.org/web/20260312001430/https://www.inecnigeria.org/wp-content/uploads/2019/03/2019-GE-PRESIDENTIAL-ELECTION-RESULTS.pdf
  - Figures: APC 15,191,847; PDP 11,262,978; valid 27,324,583; rejected 1,289,607; accredited 29,364,209.
  - This PDF and INEC's 2019 General Election Report (Annexure 1) give national figures only. No INEC state table was found.
- **Not available by state:** accredited and rejected.

## 2023 (25 Feb 2023)
- **Party votes:** Stears Elections, https://www.stears.co/elections/2023/president/ (`__NEXT_DATA__`, `partyDominanceData`). Fields: APC, LP, PDP, NNPP, Others and total valid per state.
- **Registered voters:** INEC 2023 General Election Report, Table 8.8, 2023 column (sums to 93,469,008). Same URL as above.
- **Not available by state:** accredited and rejected.
  - INEC's IReV portal (inecelectionresults.ng) holds only images of result sheets, and its API could not be reached.
  - The 2023 INEC report has no state-level presidential vote table.
- **Official national totals:** APC 8,794,726; PDP 6,984,520; LP 6,101,533; NNPP 1,496,687 (as supplied in the brief). Total valid 24,025,940 is confirmed by Wikipedia and Stears, not by an INEC document downloaded here.

## Cross-check only: Wikipedia
Revision IDs retrieved via the API, stored in `dl/wiki/`:
- 1999 presidential: 1383204503
- 2003 presidential: 1377655800
- 2015 general: 1385023157
- 2019 general: 1385037235 (its table cites "BBC, This Day, Vanguard")
- 2023 presidential: 1385835335

Psephos text files (http://psephos.adam-carr.net/countries/n/nigeria/) are stored in `dl/psephos/`. They were used only for comparison.
