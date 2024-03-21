#!/bin/bash
#
# Create flags after a CSV description file (flags.csv)
# 4 fields are expected in the file: flag ID and the 3 colors
# After generating the flags, create a montage in a single file with their names
# You need to have imagemagick installed:
# 	sudo apt install imagemagick
#
# V. Santos, 22-Apr-2023
###################################################################


cat flags.csv | awk -F, '{system("convert -size 3x2 xc:white \
    -fill "$2 " -draw \"line 0,0 0,1\" \
    -fill "$3 " -draw \"line 1,0 1,1\" \
    -fill "$4 " -draw \"line 2,0 2,1\" \
    -scale 200x200  "$1".png")}'

montage -label '%t' *.png -font Helvetica -pointsize 18 -border 1 -bordercolor black -geometry +10+10 allFlags.jpg
