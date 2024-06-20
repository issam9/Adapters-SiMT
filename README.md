# Fixed and Adaptive Simultaneous Machine Translation Strategies Using Adapters


Our method is implemented based on the open-source toolkit [Fairseq](https://github.com/pytorch/fairseq).

## Requirements and Installation

- Python version = 3.8

- [PyTorch](http://pytorch.org/) version = 2.0

- Install fairseq:

  ```bash
  git clone https://github.com/issam9/Adapter-SiMT.git
  cd Adapter-SiMT
  pip install --editable ./
  ```


## Quick Start

### Data Pre-processing

We use the data of IWSLT15 English-Vietnamese (download [here](https://nlp.stanford.edu/projects/nmt/)) and WMT15 German-English (download [here](https://www.statmt.org/wmt15/)).

For WMT15 German-English, we tokenize the corpus via [mosesdecoder/scripts/tokenizer/normalize-punctuation.perl](https://github.com/moses-smt/mosesdecoder) and apply BPE with 32K merge operations via [subword_nmt/apply_bpe.py](https://github.com/rsennrich/subword-nmt).

Then, we process the data into the fairseq format, adding `--joined-dictionary` for WMT15 German-English:

```bash
src=SOURCE_LANGUAGE
tgt=TARGET_LANGUAGE
train_data=TRAIN_DATA_DIR
vaild_data=VALID_DATA_DIR
test_data=TEST_DATA_DIR
data=DATA_DIR

# add --joined-dictionary for WMT15 German-English
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${train_data} --validpref ${vaild_data} \
    --testpref ${test_data}\
    --destdir ${data} \
    --workers 20
```

### Training
```bash
data=DATA_DIR
modelfile=MODEL_DIR
adapter_lagging=1,3,5,7,9,11,13,15

python train.py  --ddp-backend=no_c10d ${data} --arch transformer --share-all-embeddings \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --criterion label_smoothed_cross_entropy \
 --reset-dataloader --reset-lr-scheduler --reset-optimizer \
 --label-smoothing 0.1 \
 --encoder-attention-heads 8 \
 --decoder-attention-heads 8 \
 --left-pad-source False \
 --fp16 \
 --adapter-lagging ${adapter_lagging} \
 --save-dir ${modelfile} \
 --bottleneck-dim 64 \
 --add-adapters \
 --max-tokens 8192 \
 --update-freq 4 

```
### Inference

#### Adapters-Wait-k
```bash
data=DATA_DIR
modelfile=MODEL_DIR
ref_dir=REFERENCE_PATH


# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt 


for testk in 1 3 5 7 9 11 13 15; do
    # generate translation
    python generate.py ${data} --path $modelfile/average-model.pt --batch-size 250 --beam 1 --left-pad-source False --fp16  --remove-bpe --test-wait-k ${testk} --results-file ${modelfile}/results.csv > pred.out

    grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
    grep ^T pred.out | cut -f1,2- | cut -c3- | sort -k1n | cut -f2- > target.translation
    
    ./multi-bleu.perl -lc ${ref_dir} < pred.translation
    BLEU=$(./multi-bleu.perl -lc ${ref_dir} < pred.translation | grep -oP 'BLEU = \K\d+\.\d+')
    echo ",$BLEU" >> ${modelfile}/${prefix}results.csv

    tail -n 5 pred.out
    
done
```

#### Adaptive-Adapters
```bash
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
    python generate_simultaneous.py ${data} --path $modelfile/average-model.pt --kmin ${kmin} --kmax ${kmax} --first-p "${first_ps[i]}" --last-p "${last_ps[i]}" --batch-size 250 --beam 1 --left-pad-source False --fp16 --remove-bpe --results-file ${modelfile}/results.csv > pred.out

    grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
    grep ^T pred.out | cut -f1,2- | cut -c3- | sort -k1n | cut -f2- > target.translation

    ./multi-bleu.perl -lc ${ref_dir} < pred.translation
    BLEU=$(./multi-bleu.perl -lc ${ref_dir} < pred.translation | grep -oP 'BLEU = \K\d+\.\d+')
    echo ",$BLEU" >> ${modelfile}/results.csv

    echo "first_p: ${first_ps[i]}, last_p: ${last_ps[i]}"
    tail -n 5 pred.out
            

done

```