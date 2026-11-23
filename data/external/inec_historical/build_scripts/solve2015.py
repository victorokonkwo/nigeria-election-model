"""Pick consistent values from OCR candidates for the 2015 INEC summary.
Constraints: sum(party votes) == VALID ; VALID + REJ == TOTAL ; column sums == TOTAL row.
usage: python3 -I solve2015.py grid.json out.json
"""
import sys, json, itertools

t = json.load(open(sys.argv[1]))
PARTIES = ["AA", "ACPN", "AD", "ADC", "APA", "APC", "CPP", "HOPE", "KOWA", "NCP", "PDP", "PPN", "UDP", "UPP"]
COLS = ["REG", "ACC"] + PARTIES + ["VALID", "REJ", "TOTAL"]


def cands(lst):
    vals = [(v, n) for v, n in lst if v]
    out = []
    for v, n in vals:
        frag = any(o != v and len(o) > len(v) and (o.endswith(v) or o.startswith(v)) for o, _ in vals)
        if not frag:
            out.append((v, n))
    out.sort(key=lambda a: (-a[1], -len(a[0])))
    return [int(v) for v, _ in out] or [None]


res, notes = {}, {}
for st, row in t.items():
    C = {c: cands(row[c]) for c in COLS}
    pick = {c: C[c][0] for c in COLS}
    ok = False
    # search: valid/rej/total consistent first
    for v, r, tt in itertools.product(C["VALID"][:4], C["REJ"][:4], C["TOTAL"][:4]):
        if None in (v, r, tt) or v + r != tt:
            continue
        # party combos: allow up to 2 cells deviating from top pick
        base = {p: C[p][0] for p in PARTIES}
        def s(d):
            return sum((d[p] or 0) for p in PARTIES)
        if None not in base.values() and s(base) == v:
            pick.update(base); pick.update(VALID=v, REJ=r, TOTAL=tt); ok = "exact"; break
        found = None
        for k in (1, 2):
            for ps in itertools.combinations(PARTIES, k):
                for alts in itertools.product(*[C[p][1:4] + ([None] if C[p][0] is None else []) for p in ps]):
                    d = dict(base)
                    for p, a in zip(ps, alts):
                        d[p] = a
                    if None in d.values():
                        # single missing cell -> solve by difference
                        miss = [p for p in PARTIES if d[p] is None]
                        if len(miss) == 1:
                            d[miss[0]] = v - sum(d[p] for p in PARTIES if p != miss[0])
                            if d[miss[0]] < 0:
                                continue
                            found = (d, "derived:" + miss[0])
                            break
                        continue
                    if s(d) == v:
                        found = (d, "alt:" + ",".join(ps)); break
                if found: break
            if found: break
        if found:
            pick.update(found[0]); pick.update(VALID=v, REJ=r, TOTAL=tt); ok = found[1]; break
    res[st] = pick
    notes[st] = ok or "UNRESOLVED"
    print(st.ljust(12), notes[st], {c: C[c][:3] for c in COLS if len(C[c]) > 1} if not ok or ok != "exact" else "")

# column checks
states = [s for s in res if s != "TOTAL"]
for c in COLS:
    tot = sum((res[s][c] or 0) for s in states)
    print(c.ljust(6), tot, res["TOTAL"][c], "OK" if tot == res["TOTAL"][c] else f"DIFF {tot - (res['TOTAL'][c] or 0)}")
json.dump({"values": res, "notes": notes}, open(sys.argv[2], "w"), indent=1)
