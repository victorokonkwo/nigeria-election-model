"""Group Vision OCR tokens (tsv from ocr.swift) into rows by y-centre.
usage: python3 -I rows.py <tsv> [ytol]
prints one line per row: y | x:text  x:text ...
"""
import sys

path = sys.argv[1]
ytol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.006
toks = []
for line in open(path, encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) < 6:
        continue
    x, y, w, h, conf = map(float, p[:5])
    toks.append((y + h / 2, x, x + w, p[5], conf))
toks.sort()
rows = []
for t in toks:
    if rows and abs(t[0] - rows[-1][0]) < ytol:
        rows[-1][1].append(t)
        rows[-1][0] = sum(a[0] for a in rows[-1][1]) / len(rows[-1][1])
    else:
        rows.append([t[0], [t]])
for yc, ts in rows:
    ts.sort(key=lambda a: a[1])
    print(f"{yc:.4f} | " + "  ".join(f"{a[1]:.3f}-{a[2]:.3f}:{a[3]}" for a in ts))
