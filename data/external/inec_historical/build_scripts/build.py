"""Build results_long.csv and state_totals.csv from downloaded sources.
usage: python3 -I build.py <research_root>
"""
import sys, os, re, csv, json, html, collections

ROOT = sys.argv[1]
DL = os.path.join(ROOT, "dl")
OCR = os.path.join(ROOT, "ocr")

STATES = ["Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno",
          "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo", "Jigawa",
          "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger",
          "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe",
          "Zamfara", "FCT"]
ALIAS = {"nassarawa": "Nasarawa", "abuja": "FCT", "federal capital territory": "FCT", "federal capital territory fct": "FCT", "akwa-ibom": "Akwa Ibom",
         "akwaibom": "Akwa Ibom", "cross_river": "Cross River", "crossriver": "Cross River", "fct": "FCT",
         "sokto": "Sokoto"}


def canon(s):
    k = re.sub(r"[^a-z ]", "", s.lower().replace("-", " ").replace("_", " ")).strip()
    k = re.sub(r"\s+", " ", k)
    if k in ALIAS:
        return ALIAS[k]
    if k.replace(" ", "") in ALIAS:
        return ALIAS[k.replace(" ", "")]
    for st in STATES:
        if st.lower() == k:
            return st
    raise KeyError(s)


def num(s):
    return int(s.replace(",", ""))


long_rows = []   # (year, state, party, votes)
tot_rows = {}    # (year, state) -> dict
report = []


def log(*a):
    s = " ".join(str(x) for x in a)
    report.append(s)
    print(s)


# ---------------- 1999: IFES/AAEA report table (INEC votes-cast data), Vision OCR ----------------
FIX = str.maketrans({"฿": "8", "B": "8", "O": "0", "o": "0", "l": "1", "I": "1", "|": "", "S": "5"})
rows = collections.defaultdict(list)
for line in open(os.path.join(OCR, "ifes1999.tsv"), encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    x, y, w, h = map(float, p[:4])
    rows[round((y + h / 2) * 1000)].append((x, x + w, p[5]))
# group rows within 6/1000
keys = sorted(rows)
groups = []
for k in keys:
    if groups and k - groups[-1][0] <= 6:
        groups[-1][1].extend(rows[k])
    else:
        groups.append([k, list(rows[k])])
COLX = {"reg": 0.406, "valid": 0.471, "pdp": 0.616, "app": 0.718}  # right edges
OCR_STATE_ORDER = STATES  # table is in alphabetical order with FCT last, S/No 1..37
data99 = []
for k, toks in groups:
    toks.sort()
    vals = {}
    for x0, x1, t in toks:
        for c, xr in COLX.items():
            tt = t.translate(FIX).replace(".", ",").replace(" ", "")
            if abs(x1 - xr) < 0.006 and re.fullmatch(r"[\d,]+", tt):
                vals[c] = num(tt)
    if len(vals) == 4:
        data99.append(vals)
assert len(data99) == 38, len(data99)
tot99 = data99[-1]
data99 = data99[:-1]
log("== 1999 (IFES/AAEA report p.76, INEC votes-cast data, OCR)")
# OCR 6<->8 / 5<->6 confusions on a low-res scan. Each fix is the unique value that satisfies
# PDP+APP == valid AND reproduces the printed 2-dp vote-share columns; column totals then match.
FIX99 = {("Abia", "pdp"): (380823, 360823), ("Cross River", "app"): (283488, 283468),
         ("Ekiti", "pdp"): (191518, 191618), ("Niger", "app"): (140485, 140465)}
for st, v in zip(STATES, data99):
    for c in ("pdp", "app"):
        if (st, c) in FIX99 and v[c] == FIX99[(st, c)][0]:
            v[c] = FIX99[(st, c)][1]
            share = 100 * v[c] / v["valid"]
            log(f"  OCR fix {st} {c}: {FIX99[(st, c)][0]} -> {v[c]} (share {share:.2f}% matches printed column)")
bad = 0
for st, v in zip(STATES, data99):
    if v["pdp"] + v["app"] != v["valid"]:
        bad += 1
        log(f"  row check FAIL {st}: PDP {v['pdp']} + APP {v['app']} != valid {v['valid']}")
    long_rows += [(1999, st, "PDP", v["pdp"]), (1999, st, "APP/AD (Falae)", v["app"])]
    tot_rows[(1999, st)] = dict(registered=v["reg"], accredited="", valid=v["valid"], rejected="")
for c in COLX:
    s = sum(v[c] for v in data99)
    log(f"  col {c}: sum states {s:,} vs printed total {tot99[c]:,} {'OK' if s == tot99[c] else 'DIFF ' + str(s - tot99[c])}")
log(f"  row checks failed: {bad}")

# ---------------- 2003: INEC 2003 results website (Wayback) ----------------
log("== 2003 (INEC website 2003elections/results/presidential, archived 29 Apr 2003)")
d03 = os.path.join(DL, "inec2003pres")
for fn in sorted(os.listdir(d03)):
    if fn == "summary.htm":
        continue
    t = open(os.path.join(d03, fn), encoding="latin-1").read()
    t = re.sub(r"<script.*?</script>", "", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", html.unescape(t))
    st = canon(fn[:-4])
    lga = re.search(r"(\d+) of (\d+) LGAs received", t)
    if lga.group(1) != lga.group(2):
        log(f"  WARNING {st}: only {lga.group(0)}")
    parties = re.findall(r"([A-Z]{2,5}) Pres: .*? Vice: .*? ([\d,]+) ([\d.]+)", t)
    valid = num(re.search(r"Total Valid Votes ([\d,]+)", t).group(1))
    rej = num(re.search(r"Rejected Ballots ([\d,]+)", t).group(1))
    vr = num(re.search(r"Valid Votes \+ Rejected Ballots ([\d,]+)", t).group(1))
    reg = num(re.search(r"Number of Voters on Register ([\d,]+)", t).group(1))
    s = sum(num(v) for _, v, _ in parties)
    if s != valid or valid + rej != vr or len(parties) != 20:
        log(f"  row check FAIL {st}: partysum {s} valid {valid} rej {rej} v+r {vr} nparties {len(parties)}")
    for p, v, _ in parties:
        long_rows.append((2003, st, p, num(v)))
    tot_rows[(2003, st)] = dict(registered=reg, accredited="", valid=valid, rejected=rej)
assert len([k for k in tot_rows if k[0] == 2003]) == 37

# ---------------- 2011: INEC results dashboard CSV (Wayback) ----------------
log("== 2011 (INEC results dashboard CSV inec_presidential_results_37_states.csv)")
with open(os.path.join(DL, "inec2011csv", "inec_presidential_results_37_states.csv"), encoding="latin-1") as f:
    r = csv.DictReader(f)
    seen = set()
    for row in r:
        row = {k.strip(): v.strip() for k, v in row.items()}
        st = canon(row["State"])
        seen.add(st)
        long_rows.append((2011, st, row["Party"], int(row["Votes"])))
assert len(seen) == 37, seen
for st in STATES:
    v = sum(x[3] for x in long_rows if x[0] == 2011 and x[1] == st)
    tot_rows[(2011, st)] = dict(registered="", accredited="", valid=v, rejected="")
log("  note: valid votes for 2011 = sum of all 20 party rows in the INEC CSV (CSV has no valid/rejected/registered fields)")

# ---------------- 2015: INEC summary of results sheet (scan), Vision OCR ----------------
log("== 2015 (INEC 'Summary of Results' sheet, scanned PDF, OCR + arithmetic validation)")
pk = json.load(open(os.path.join(OCR, "pick2015.json")))
P15 = ["AA", "ACPN", "AD", "ADC", "APA", "APC", "CPP", "HOPE", "KOWA", "NCP", "PDP", "PPN", "UDP", "UPP"]
name15 = {s.upper(): s for s in STATES}
for k, d in pk.items():
    if k == "TOTAL":
        continue
    st = name15[k]
    s = sum(d[p] for p in P15)
    if s != d["VALID"]:
        if s + d["REJ"] == d["TOTAL"]:
            log(f"  {st}: OCR valid {d['VALID']} replaced by party sum {s} (party sum + rejected == total cast {d['TOTAL']})")
            d["VALID"] = s
        else:
            log(f"  row check FAIL {st}")
    if d["VALID"] + d["REJ"] != d["TOTAL"]:
        log(f"  valid+rejected != total cast in {st}")
    for p in P15:
        long_rows.append((2015, st, p, d[p]))
    tot_rows[(2015, st)] = dict(registered=d["REG"], accredited=d["ACC"], valid=d["VALID"], rejected=d["REJ"])
T = pk["TOTAL"]
for c in ["REG", "ACC"] + P15 + ["VALID", "REJ", "TOTAL"]:
    s = sum(pk[name15[k].upper()][c] for k in name15 if k != "TOTAL") if False else sum(pk[k][c] for k in pk if k != "TOTAL")
    if s != T[c]:
        log(f"  col {c}: sum states {s:,} vs printed total {T[c]:,} DIFF {s - T[c]}")
log("  all other columns: sum of states == printed national total")

# ---------------- 2019 / 2023: Stears embedded JSON + INEC 2023 report Table 8.8 (registered) ----------------
reg_tbl = {}
txt = open(os.path.join(DL, "inec2023report", "report.txt"), encoding="utf-8").read().splitlines()
start = next(i for i, l in enumerate(txt) if "Table 8.8: Comparative Analysis of State by State" in l and i > 1000)
for l in txt[start:start + 70]:
    m = re.match(r"\s*(\d+)\s+([A-Za-z \-]+?)\s+([\d,]{6,})\s+([\d,]{6,})\s+[\d.]+\s*$", l)
    if m:
        reg_tbl[canon(m.group(2))] = (num(m.group(3)), num(m.group(4)))
    if re.match(r"\s*Total", l) and "93,469,008" in l:
        break
assert len(reg_tbl) == 37, len(reg_tbl)
log(f"  INEC 2023 report Table 8.8 registered voters: 2019 sum {sum(v[0] for v in reg_tbl.values()):,}, 2023 sum {sum(v[1] for v in reg_tbl.values()):,}")
code2state = {}
for y in (2019, 2023):
    log(f"== {y} (Stears Elections __NEXT_DATA__ partyDominanceData)")
    f = os.path.join(DL, "stears", f"https___www.stears.co_elections_{y}_president_.html.json")
    pp = json.load(open(f))["props"]["pageProps"]
    for s in pp["states"]:
        code2state[s["code"]] = canon(s["name"])
    tv = {}
    others_raw = {}
    for rec in pp["partyDominanceData"]:
        st = code2state[rec["state"]]
        tv[st] = rec["total_votes"]
        if rec["party"] == "Others":
            others_raw[st] = rec["vote_count"]
            continue
        long_rows.append((y, st, rec["party"], rec["vote_count"]))
    # Stears' "Others" rows are not always consistent with its own state valid totals (2019);
    # write Others as valid - named parties, and log where it differs from Stears' figure.
    ndiff = 0
    for st in STATES:
        named = sum(x[3] for x in long_rows if x[0] == y and x[1] == st)
        oth = tv[st] - named
        if oth != others_raw[st]:
            ndiff += 1
        long_rows.append((y, st, "Others", oth))
    log(f"  'Others' derived as valid minus named parties; differs from Stears' own Others figure in {ndiff} states "
        f"(Stears Others national sum {sum(others_raw.values()):,} vs derived {sum(tv.values()) - sum(x[3] for x in long_rows if x[0] == y and x[2] != 'Others'):,})")
    for st in STATES:
        s = sum(x[3] for x in long_rows if x[0] == y and x[1] == st)
        if s != tv[st]:
            log(f"  row check FAIL {y} {st}: {s} vs {tv[st]}")
        tot_rows[(y, st)] = dict(registered=reg_tbl[st][0 if y == 2019 else 1], accredited="", valid=tv[st], rejected="")

# ---------------- national validation ----------------
OFFICIAL = {
    1999: {"PDP": 18738154, "APP/AD (Falae)": 11110287},
    2003: {"PDP": 24456140, "ANPP": 12710022, "APGA": 1297445},
    2011: {"PDP": 22495187, "CPC": 12214853, "ACN": 2079151, "ANPP": 917012},
    2015: {"APC": 15424921, "PDP": 12853162},
    2019: {"APC": 15191847, "PDP": 11262978},
    2023: {"APC": 8794726, "PDP": 6984520, "LP": 6101533, "NNPP": 1496687},
}
OFF_VALID = {1999: 29848441, 2003: 39480489, 2015: 28587564, 2019: 27324583, 2023: 24025940}
log("== national validation (sum of states vs official)")
for y, d in OFFICIAL.items():
    for p, off in d.items():
        s = sum(x[3] for x in long_rows if x[0] == y and x[2] == p)
        log(f"  {y} {p:15s} computed {s:>12,}  official {off:>12,}  {'MATCH' if s == off else 'DIFF ' + format(s - off, ',')}")
    v = sum(tot_rows[(y, st)]["valid"] for st in STATES)
    if y in OFF_VALID:
        log(f"  {y} {'valid':15s} computed {v:>12,}  official {OFF_VALID[y]:>12,}  {'MATCH' if v == OFF_VALID[y] else 'DIFF ' + format(v - OFF_VALID[y], ',')}")
    else:
        log(f"  {y} {'valid':15s} computed {v:>12,}")

# ---------------- write ----------------
long_rows.sort(key=lambda r: (r[0], r[1], -r[3]))
with open(os.path.join(ROOT, "results_long.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["year", "state_name", "party", "votes"])
    w.writerows(long_rows)
with open(os.path.join(ROOT, "state_totals.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["year", "state_name", "registered", "accredited", "valid", "rejected"])
    for (y, st) in sorted(tot_rows):
        d = tot_rows[(y, st)]
        w.writerow([y, st, d["registered"], d["accredited"], d["valid"], d["rejected"]])
open(os.path.join(ROOT, "validation_log.txt"), "w").write("\n".join(report) + "\n")
print("rows:", len(long_rows), "state-years:", len(tot_rows))
