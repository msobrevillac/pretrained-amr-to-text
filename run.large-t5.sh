
for mode in "penman" "penman-frames" "dfs" "dfs-relations" "dfs-frames" "dfs-relations-frames"
do

IFS=","

for seed in "run-1",13 "run-2",14 "run-3",15 "run-4",16
do

read -a run <<< "$seed"

python Train.py -train-src data/amr/2/$mode/train/amr.txt \
    -train-tgt data/amr/2/$mode/train/sentence.txt \
    -dev-src data/amr/2/$mode/dev/amr.txt -dev-tgt data/amr/2/$mode/dev/sentence.txt \
    -test-src data/amr/2/$mode/test/amr.txt \
    -batch-size 1 -src-max-length 120 -tgt-max-length 60 -lr 5e-4 \
    -epochs 12 -gpu -print-every 10 -model unicamp-dl/ptt5-large-portuguese-vocab \
    -save-dir output/t5-large/2/$mode/fixed/${run[0]} -seed ${run[1]} -beam-size 15 \
    -representation amr -fixed-embed -pretrained-model t5 -accum-steps 32 \
    -early-stopping-patience 4 -early-stopping-criteria perplexity



python Train.py -train-src data/amr/2/$mode/train/amr.txt \
    -train-tgt data/amr/2/$mode/train/sentence.txt \
    -dev-src data/amr/2/$mode/dev/amr.txt -dev-tgt data/amr/2/$mode/dev/sentence.txt \
    -test-src data/amr/2/$mode/test/amr.txt \
    -batch-size 1 -src-max-length 120 -tgt-max-length 60 -lr 5e-4 \
    -epochs 12 -gpu -print-every 10 -model unicamp-dl/ptt5-large-portuguese-vocab \
    -save-dir output/t5-large/2/$mode/nofixed/${run[0]} -seed ${run[1]} -beam-size 15 \
    -representation amr -pretrained-model t5 -accum-steps 32 \
    -early-stopping-patience 4 -early-stopping-criteria perplexity

done
done
