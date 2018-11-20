#!/bin/bash
cd ~/Documents/ucbinternship/pywrentests
source activate py2_env
val=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)
for i in "${val[@]}"; do
	echo "${i}";
	python trajoptimize.py -s "${i}";
done