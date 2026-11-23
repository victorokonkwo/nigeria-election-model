#!/bin/bash
# usage: wb.sh <url> <outfile>
u="$1"; out="$2"
ts=$(curl -s "https://archive.org/wayback/available?url=$u" | python3 -I -c "import sys,json; d=json.load(sys.stdin); c=d.get('archived_snapshots',{}).get('closest'); print(c['timestamp'] if c else '')")
if [ -z "$ts" ]; then echo "NO SNAPSHOT $u"; exit 1; fi
curl -sL -o "$out" -w "%{http_code} %{size_download} " "https://web.archive.org/web/${ts}id_/$u"
echo "ts=$ts"; file "$out"
