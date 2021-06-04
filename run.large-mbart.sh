
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
    -batch-size 4 -src-max-length 180 -tgt-max-length 80 -lr 5e-4 \
    -epochs 12 -gpu -print-every 10 -model facebook/mbart-large-cc25 \
    -save-dir output/t5-small/2/$mode/fixed/${run[0]} -seed ${run[1]} -beam-size 15 \
    -representation amr -fixed-embed -pretrained-model mbart -accum-steps 8 \
    -early-stopping-patience 4 -early-stopping-criteria perplexity



python Train.py -train-src data/amr/2/$mode/train/amr.txt \
    -train-tgt data/amr/2/$mode/train/sentence.txt \
    -dev-src data/amr/2/$mode/dev/amr.txt -dev-tgt data/amr/2/$mode/dev/sentence.txt \
    -test-src data/amr/2/$mode/test/amr.txt \
    -batch-size 4 -src-max-length 180 -tgt-max-length 80 -lr 5e-4 \
    -epochs 12 -gpu -print-every 10 -model facebook/mbart-large-cc25 \
    -save-dir output/t5-small/2/$mode/nofixed/${run[0]} -seed ${run[1]} -beam-size 15 \
    -representation amr -pretrained-model mbart -accum-steps 8 \
    -early-stopping-patience 4 -early-stopping-criteria perplexity

done
done
