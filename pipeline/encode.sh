#!/bin/bash
# usage: encode.sh v name out.mp4
W=/tmp/claude-0/-home-user-calude-/c996a4b3-e028-50ae-a262-aa818c3654dc/scratchpad/work
v=$1; name=$2; out=$3; crf=${4:-14}
ffmpeg -v error -y -f rawvideo -pix_fmt yuv420p10le -s 1080x1920 -r 30000/1001 -i $W/${v}_${name}.yuv \
  -i $W/in/$v.mov -map 0:v -map 1:a:0 -c:a copy \
  -c:v libx265 -preset slow -crf $crf -pix_fmt yuv420p10le -tag:v hvc1 \
  -x265-params "log-level=error:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited:repeat-headers=1" \
  -color_primaries bt2020 -color_trc arib-std-b67 -colorspace bt2020nc -color_range tv \
  -movflags +faststart -shortest $out
