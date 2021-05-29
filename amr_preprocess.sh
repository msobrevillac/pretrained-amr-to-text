

for mode in "penman" "penman-frames" "dfs" "dfs-relations" "dfs-frames" "dfs-relations-frames"
do

python amr_preprocessing.py -input /home/msobrevillac/Projects/phd/Datasets/data/2 -output data/amr/2/$mode -mode $mode -no-variable

done
