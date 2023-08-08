#! /bin/bash

#SBATCH -t 12:00:00

#SBATCH -n 1

#SBATCH -c 8

SEQ=127
LR=0.001
OPT=adam
SCH=cosine
DECAY=1
E=250
WLR=5e-05
WE=50
MWLR=5e-05
MWE=50
DATA=gen_apigraph_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2018-12
RESULT_DIR=cade_results
MODEL_DATE=20230501

CNT=$1

S='triplet'
LOSS='triplet-mse'
TS=$(date "+%m.%d-%H.%M.%S")

python -u relabel.py	                                \
        --data ${DATA}                                  \
        --benign_zero                                   \
        --mdate ${MODEL_DATE}                           \
        --train_start ${TRAIN_START}                    \
        --train_end ${TRAIN_END}                        \
        --test_start ${TEST_START}                      \
        --test_end ${TEST_END}                          \
        --encoder cae                                   \
        --enc-hidden "512-384-256-128"                  \
        --loss_func ${LOSS}                             \
        --sampler ${S}                                  \
        --bsize 1536                                    \
        --optimizer ${OPT}                              \
        --scheduler ${SCH}                              \
        --learning_rate ${LR}                           \
        --lr_decay_rate ${DECAY}                        \
        --lr_decay_epochs "10,500,10"                   \
        --epochs ${E}                                   \
        --cae-lambda 0.1                                \
        --display-interval 100                          \
        --classifier "mlp"                              \
        --cls-feat "encoded"                            \
        --mlp-hidden 100-100                            \
        --mlp-dropout 0.2                               \
        --mlp-batch-size 32                             \
        --mlp-lr 0.001                                  \
        --mlp-epochs 50                                 \
        --al                                            \
        --ood                                           \
        --count ${CNT}                                  \
        --encoder-retrain                               \
        --warm_learning_rate ${WLR}                     \
        --al_epochs ${WE}                               \
        --mlp-warm-lr ${MWLR}                           \
        --mlp-warm-epochs ${MWE}                        \
        --result experiments/020_revision/${RESULT_DIR}/cade_apigraph_${SEQ}_warm_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_wlr${WLR}_we${WE}_mwlr${MWLR}_mwe${MWE}_test_${TEST_START}_${TEST_END}_cnt${CNT}.csv \
        --log_path experiments/020_revision/${RESULT_DIR}/cade_apigraph_${SEQ}_warm_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_wlr${WLR}_we${WE}_mwlr${MWLR}_mwe${MWE}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log \
        >> experiments/020_revision/${RESULT_DIR}/cade_apigraph_${SEQ}_warm_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_wlr${WLR}_we${WE}_mwlr${MWLR}_mwe${MWE}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log 2>&1 &

wait