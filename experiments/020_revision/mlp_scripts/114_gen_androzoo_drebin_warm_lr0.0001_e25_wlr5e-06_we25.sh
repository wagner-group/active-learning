#! /bin/bash

#SBATCH -t 03:00:00

#SBATCH -n 1

#SBATCH -c 8

SEQ=114
LR=0.0001
E=25
WLR=5e-06
WE=25
DATA=gen_androzoo_drebin
TRAIN_START=2019-01
TRAIN_END=2019-12
TEST_START=2020-01
TEST_END=2021-12
RESULT_DIR=mlp_results
MODEL_DATE=20230505


CNT=$1
TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u relabel.py	                                \
            --data ${DATA}                                  \
            --mdate ${MODEL_DATE}                           \
            --train_start ${TRAIN_START}                    \
            --train_end ${TRAIN_END}                        \
            --test_start ${TEST_START}                      \
            --test_end ${TEST_END}                          \
            --classifier mlp                                \
            --mlp-hidden 100-100                            \
            --mlp-dropout 0.2                               \
            --mlp-batch-size 32                             \
            --mlp-lr ${LR}                                  \
            --mlp-epochs ${E}                               \
            --mlp-warm-lr ${WLR}                            \
            --mlp-warm-epochs ${WE}                         \
            --al                                            \
            --unc                                           \
            --count ${CNT}                                  \
            --result experiments/020_revision/${RESULT_DIR}/${SEQ}_${DATA}_warm_lr${LR}_e${E}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}.csv \
            --log_path experiments/020_revision/${RESULT_DIR}/${SEQ}_${DATA}_warm_lr${LR}_e${E}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log \
            >> experiments/020_revision/${RESULT_DIR}/${SEQ}_${DATA}_warm_lr${LR}_e${E}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log 2>&1 &

wait