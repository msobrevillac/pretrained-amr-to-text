#!/bin/bash

for mode in "dfs-relations" "dfs-frames" "dfs-relations-frames"
do

IFS=","

for seed in "run-1",13
do

read -a run <<< "$seed"

python Train.py -train-src data/amr/3/$mode/train/amr.txt \
    -train-tgt data/amr/3/$mode/train/sentence.txt \
    -dev-src data/amr/3/$mode/dev/amr.txt -dev-tgt data/amr/3/$mode/dev/sentence.txt \
    -test-src data/amr/3/$mode/test/amr.txt \
    -batch-size 16 -src-max-length 180 -tgt-max-length 80 -lr 5e-4 \
    -epochs 12 -gpu -print-every 10 -model unicamp-dl/ptt5-small-portuguese-vocab \
    -save-dir output/t5-small/3/$mode/fixed/${run[0]} -seed ${run[1]} -beam-size 15 \
    -representation amr -fixed-embed -pretrained-model t5 -accum-steps 2 \
    -early-stopping-patience 4 -early-stopping-criteria perplexity



python Train.py -train-src data/amr/3/$mode/train/amr.txt \
    -train-tgt data/amr/3/$mode/train/sentence.txt \
    -dev-src data/amr/3/$mode/dev/amr.txt -dev-tgt data/amr/3/$mode/dev/sentence.txt \
    -test-src data/amr/3/$mode/test/amr.txt \
    -batch-size 16 -src-max-length 180 -tgt-max-length 80 -lr 5e-4 \
    -epochs 12 -gpu -print-every 10 -model unicamp-dl/ptt5-small-portuguese-vocab \
    -save-dir output/t5-small/3/$mode/nofixed/${run[0]} -seed ${run[1]} -beam-size 15 \
    -representation amr -pretrained-model t5 -accum-steps 2 \
    -early-stopping-patience 4 -early-stopping-criteria perplexity

done
done
