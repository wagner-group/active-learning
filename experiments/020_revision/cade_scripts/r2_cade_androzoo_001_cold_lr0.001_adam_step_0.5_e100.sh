#! /bin/bash

#SBATCH -t 12:00:00

#SBATCH -n 1

#SBATCH -c 8

SEQ=001
LR=0.001
OPT=adam
SCH=step
DECAY=0.5
E=100
DATA=gen_androzoo_drebin
TRAIN_START=2019-01
TRAIN_END=2019-12
TEST_START=2020-01
TEST_END=2021-12
RESULT_DIR=cade_results
MODEL_DATE=20230505

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
        --cls-retrain 1                                 \
        --mlp-hidden 100-100                            \
        --mlp-dropout 0.2                               \
        --mlp-batch-size 32                             \
        --mlp-lr 0.001                                  \
        --mlp-epochs 50                                 \
        --al                                            \
        --ood                                           \
        --count ${CNT}                                  \
        --cold-start                                    \
        --encoder-retrain                               \
        --result experiments/020_revision/${RESULT_DIR}/r2_cade_androzoo_${SEQ}_cold_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}.csv \
        --log_path experiments/020_revision/${RESULT_DIR}/r2_cade_androzoo_${SEQ}_cold_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log \
        >> experiments/020_revision/${RESULT_DIR}/r2_cade_androzoo_${SEQ}_cold_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log 2>&1 &

wait