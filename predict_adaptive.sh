#!/bin/bash


data=DATA_DIR
modelfile=MODEL_DIR
ref_dir=REFERENCE_PATH

kmin=1
kmax=5
first_ps=( 0.2 0.4 0.6 0.8 1. 1. 1. 1. 1. )
last_ps=( 0. 0. 0. 0. 0. 0.2 0.4 0.6 0.8 )


python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt 


for i in "${!first_ps[@]}" ; do
    # generate translation
    python generate_simultaneous.py ${data} --path $modelfile/average-model.pt --kmin ${kmin} --kmax ${kmax} --first-p "${first_ps[i]}" --last-p "${last_ps[i]}" --strategy probs --batch-size 250 --beam 1 --left-pad-source False --fp16 --remove-bpe --results-file ${modelfile}/results.csv > pred.out

    grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
    grep ^T pred.out | cut -f1,2- | cut -c3- | sort -k1n | cut -f2- > target.translation

    ./multi-bleu.perl -lc ${ref_dir} < pred.translation
    BLEU=$(./multi-bleu.perl -lc ${ref_dir} < pred.translation | grep -oP 'BLEU = \K\d+\.\d+')
    echo ",$BLEU" >> ${modelfile}/results.csv

    echo "first_p: ${first_ps[i]}, last_p: ${last_ps[i]}"
    tail -n 5 pred.out
            

done