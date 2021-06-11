

for mode in "penman" "penman-frames" "dfs" "dfs-relations" "dfs-frames" "dfs-relations-frames"
do

python amr_preprocessing.py -input /home/msobrevillac/Projects/phd/Datasets/data/$1 -output data/amr/$1/$mode -mode $mode -no-variable

done


for mode in "penman" "penman-frames"
do

python amr_preprocessing.py -input /home/msobrevillac/Projects/phd/Datasets/data/$1 -output data/amr/$1/$mode-var -mode $mode

done
