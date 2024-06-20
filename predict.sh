#!/bin/bash


data=DATA_DIR
modelfile=MODEL_DIR
ref_path=REFERENCE_PATH


# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt 


for testk in 1 3 5 7 9 11 13 15; do
    # generate translation
    python generate.py ${data} --path $modelfile/average-model.pt --batch-size 250 --beam 1 --left-pad-source False --fp16  --remove-bpe --test-wait-k ${testk} --results-file ${modelfile}/results.csv > pred.out

    grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
    grep ^T pred.out | cut -f1,2- | cut -c3- | sort -k1n | cut -f2- > target.translation
    
    ./multi-bleu.perl -lc ${ref_path} < pred.translation
    BLEU=$(./multi-bleu.perl -lc ${ref_path} < pred.translation | grep -oP 'BLEU = \K\d+\.\d+')
    echo ",$BLEU" >> ${modelfile}/results.csv

    tail -n 5 pred.out
    
done