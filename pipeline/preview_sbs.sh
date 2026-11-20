#!/bin/bash
# usage: preview_sbs.sh original.mov result.mp4 out.mp4   -> SDR side-by-side (original | people removed)
TM="zscale=tin=arib-std-b67:pin=bt2020:min=bt2020nc:rin=tv:t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
ffmpeg -v error -y -i "$1" -i "$2" -filter_complex \
 "[0:v]scale=540:960,setsar=1,$TM[a];[1:v]scale=540:960,setsar=1,$TM[b];[a][b]hstack=inputs=2[v]" \
 -map "[v]" -map 1:a? -c:v libx264 -preset medium -crf 20 -c:a aac -b:a 128k -movflags +faststart "$3"
