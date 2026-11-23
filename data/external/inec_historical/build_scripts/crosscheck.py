"""Cross-check results_long.csv against Wikipedia state tables (secondary source).
usage: python3 -I crosscheck.py <root>
"""
import sys, os, re, json, html, csv, collections

ROOT = sys.argv[1]
ours = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(ROOT, "results_long.csv"))):
    ours[(int(r["year"]), r["state_name"])][r["party"]] = int(r["votes"])
valid = {}
for r in csv.DictReader(open(os.path.join(ROOT, "state_totals.csv"))):
    valid[(int(r["year"]), r["state_name"])] = int(r["valid"])

ALIAS = {"Nassarawa": "Nasarawa", "Federal Capital Territory": "FCT", "Abuja": "FCT", "FCT": "FCT", "Akwa-Ibom": "Akwa Ibom"}


def rows_of(year, marker):
    t = json.load(open(os.path.join(ROOT, "dl", "wiki", f"{year}.html")))["parse"]["text"]
    for tb in re.findall(r"<table.*?</table>", t, flags=re.S):
        txt = re.sub(r"<[^>]+>", " ", tb)
        if marker in txt and "Zamfara" in txt:
            out = []
            for tr in re.findall(r"<tr.*?</tr>", tb, flags=re.S):
                cells = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", c))).strip()
                         for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.S)]
                out.append(cells)
            return out
    return []


def n(s):
    s = re.sub(r"\[.*?\]", "", s).replace(",", "").replace("−", "-").strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def stname(c):
    c = re.sub(r"\[.*?\]", "", c).strip()
    return ALIAS.get(c, c)


# column positions (index of numeric cells in each row) per year: map party -> position among integer cells
SPEC = {
    1999: ("Obasanjo", {"PDP": 0, "APP/AD (Falae)": 1, "_valid": 2}),
    2003: ("Obasanjo", {"PDP": 0, "ANPP": 1, "APGA": 2, "_valid": 4}),
    2015: ("Jonathan", {"APC": 0, "PDP": 1}),
    2019: ("Mailafia", {"APC": 0, "PDP": 1, "_valid": -1}),
    2023: ("Turnout (%)", {"APC": 0, "PDP": 1, "LP": 2, "NNPP": 3, "_valid": 5}),
}
for y, (marker, spec) in SPEC.items():
    rows = rows_of(y, marker)
    checked = mism = 0
    for cells in rows:
        if not cells:
            continue
        st = stname(cells[0])
        if (y, st) not in ours:
            continue
        ints = [n(c) for c in cells[1:] if n(c) is not None and "%" not in c]
        # drop small 'states won' tallies (0/1) that follow vote cells in 2019/2023 tables
        if y in (2019, 2023):
            ints = [n(c) for c in cells[1:] if n(c) is not None and "%" not in c and c not in ("0", "1")]
        for p, i in spec.items():
            try:
                w = ints[i]
            except IndexError:
                continue
            o = valid[(y, st)] if p == "_valid" else ours[(y, st)].get(p)
            checked += 1
            if w != o:
                mism += 1
                print(f"  {y} {st} {p}: ours {o} wikipedia {w}")
    print(f"{y}: {checked} cells compared with Wikipedia, {mism} mismatches")
