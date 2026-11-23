#!/bin/bash
# usage: tile_ocr.sh <image> <outtsv> <tileW> <tileH> <step_x> <step_y> <scale> [y0 y1]
# OCRs overlapping tiles and writes tokens in full-image pixel coords: x0 y0 x1 y1 conf text
img="$1"; out="$2"; TW=$3; TH=$4; SX=$5; SY=$6; SC=$7
S=$(cd "$(dirname "$0")" && pwd)
W=$(sips -g pixelWidth "$img" | awk '/pixelWidth/{print $2}')
H=$(sips -g pixelHeight "$img" | awk '/pixelHeight/{print $2}')
Y0=${8:-0}; Y1=${9:-$H}
tmp=$(mktemp -d)
: > "$out"
y=$Y0
while [ $y -lt $Y1 ]; do
  x=0
  while [ $x -lt $W ]; do
    tw=$TW; th=$TH
    [ $((x+tw)) -gt $W ] && tw=$((W-x))
    [ $((y+th)) -gt $H ] && th=$((H-y))
    sips -c $th $tw --cropOffset $y $x "$img" --out "$tmp/t.png" -s format png >/dev/null 2>&1
    sips -z $((th*SC)) $((tw*SC)) "$tmp/t.png" --out "$tmp/t2.png" >/dev/null 2>&1
    "$S/ocr" "$tmp/t2.png" | awk -F'\t' -v ox=$x -v oy=$y -v tw=$tw -v th=$th 'BEGIN{OFS="\t"}{print ox+$1*tw, oy+$2*th, ox+($1+$3)*tw, oy+($2+$4)*th, $5, $6}' >> "$out"
    x=$((x+SX))
  done
  y=$((y+SY))
done
rm -rf "$tmp"
