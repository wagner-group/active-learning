#! /bin/bash

#SBATCH -t 48:00:00

#SBATCH -n 1

#SBATCH -c 8

SEQ=071
LR=0.005
OPT=sgd
SCH=cosine
DECAY=1
E=200
DATA=gen_apigraph_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2018-12
RESULT_DIR=transcend_results
MODEL_DATE=20230501
SVM_C=0.1
CRITERIA=cred

CNT=$1

modeldim="512-384-256-128"
S='half'
B=1024
LOSS='hi-dist'
TS=$(date "+%m.%d-%H.%M.%S")

python -u relabel.py	                                \
        --data ${DATA}                                  \
        --benign_zero                                   \
        --mdate ${MODEL_DATE}                           \
        --train_start ${TRAIN_START}                    \
        --train_end ${TRAIN_END}                        \
        --test_start ${TEST_START}                      \
        --test_end ${TEST_END}                          \
        --encoder enc                                   \
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
        --classifier svm                                \
        --cls-feat encoded                              \
        --cls-retrain 1                                 \
        --cold-start                                    \
        --encoder-retrain                               \
        --al                                            \
        --count ${CNT}                                  \
        --transcend                                     \
        --criteria ${CRITERIA}                          \
        --svm-c ${SVM_C}                                \
        --result experiments/020_revision/${RESULT_DIR}/${SEQ}_${DATA}_${CRITERIA}_svmc${SVM_C}_cold_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}.csv \
        --log_path experiments/020_revision/${RESULT_DIR}/${SEQ}_${DATA}_${CRITERIA}_svmc${SVM_C}_cold_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log \
        >> experiments/020_revision/${RESULT_DIR}/${SEQ}_${DATA}_${CRITERIA}_svmc${SVM_C}_cold_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log 2>&1 &

wait