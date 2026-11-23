"""Assemble 2015 INEC summary table from tiled OCR tokens.
usage: python3 -I grid2015.py <tiles.tsv> <W> <H> <out.json>
"""
import sys, re, json, collections

path, W, H, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
COLS = ["REG", "ACC", "AA", "ACPN", "AD", "ADC", "APA", "APC", "CPP", "HOPE", "KOWA",
        "NCP", "PDP", "PPN", "UDP", "UPP", "VALID", "REJ", "TOTAL"]
RIGHT = [0.163, 0.220, 0.266, 0.307, 0.348, 0.386, 0.427, 0.479, 0.520, 0.560, 0.601,
         0.641, 0.696, 0.735, 0.776, 0.816, 0.864, 0.921, 0.969]
STATES = ["ABIA", "ADAMAWA", "AKWA IBOM", "ANAMBRA", "BAUCHI", "BAYELSA", "BENUE", "BORNO",
          "CROSS RIVER", "DELTA", "EBONYI", "EDO", "EKITI", "ENUGU", "GOMBE", "IMO", "JIGAWA",
          "KADUNA", "KANO", "KATSINA", "KEBBI", "KOGI", "KWARA", "LAGOS", "NASARAWA", "NIGER",
          "OGUN", "ONDO", "OSUN", "OYO", "PLATEAU", "RIVERS", "SOKOTO", "TARABA", "YOBE",
          "ZAMFARA", "FCT", "TOTAL"]
Y0, Y36 = 0.1750, 0.6570
step = (Y36 - Y0) / 36
FIX = str.maketrans({"฿": "8", "B": "8", "S": "5", "O": "0", "o": "0", "l": "1", "I": "1", "|": "", "!": "1", "€": "6", "$": "5", "‚": ","})

cells = collections.defaultdict(list)
for line in open(path, encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) < 6:
        continue
    x0, y0, x1, y1, conf = map(float, p[:5])
    txt = p[5].strip()
    # split tokens that contain several numbers separated by spaces
    parts = txt.split()
    if len(parts) > 1 and all(re.fullmatch(r"[\d,.|]+", q) for q in parts):
        continue  # merged multi-cell token: ambiguous, skip
    t = txt.translate(FIX).replace(" ", "")
    if not re.fullmatch(r"[\d,.]+", t):
        continue
    xr = x1 / W
    yc = (y0 + y1) / 2 / H
    # column by nearest right edge
    ci = min(range(len(RIGHT)), key=lambda i: abs(RIGHT[i] - xr))
    if abs(RIGHT[ci] - xr) > 0.012:
        continue
    ri = round((yc - Y0) / step)
    if ri < 0 or ri > 37 or abs((yc - Y0) / step - ri) > 0.35:
        continue
    val = t.replace(",", "").replace(".", "")
    cells[(ri, ci)].append((val, conf))

table = {}
for ri, st in enumerate(STATES):
    row = {}
    for ci, c in enumerate(COLS):
        cands = cells.get((ri, ci), [])
        cnt = collections.Counter(v for v, _ in cands)
        row[c] = [[v, n] for v, n in cnt.most_common()]
    table[st] = row
json.dump(table, open(out, "w"), indent=0)
