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
 --save-dir ${modelfile} \
 --adapter-lagging ${adapter_lagging} \
 --bottleneck-dim 64 \
 --add-adapters \
 --max-tokens 8192 \
 --update-freq 4 