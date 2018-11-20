#!/bin/bash
cd ~/Documents/ucbinternship/pref_irl_feedback
source activate py2_env
val=(0 1 2 3 4 5 6 7 8 9)
for i in "${val[@]}"; do
	echo "${i}";
	python testDrive.py -m 13 -r "${i}";
done
