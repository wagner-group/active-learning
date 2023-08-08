#! /bin/bash

#SBATCH -t 03:00:00

#SBATCH -n 1

#SBATCH -c 8

SEQ=302
LR=0.001
E=50
DATA=gen_androzoo_drebin
TRAIN_START=2019-01
TRAIN_END=2019-12
TEST_START=2020-01
TEST_END=2021-12
RESULT_DIR=mlp_multi_results
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
            --cls-retrain 1                                 \
            --mlp-hidden 100-100                            \
            --mlp-dropout 0.2                               \
            --mlp-batch-size 32                             \
            --sampler mperclass                             \
            --sample-per-class 10                           \
            --mlp-lr ${LR}                                  \
            --mlp-epochs ${E}                               \
            --multi_class                                   \
            --al                                            \
            --unc                                           \
            --count ${CNT}                                  \
            --cold-start                                    \
            --result experiments/020_revision/${RESULT_DIR}/${DATA}_${SEQ}_cold_lr${LR}_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}.csv \
            --log_path experiments/020_revision/${RESULT_DIR}/${DATA}_${SEQ}_cold_lr${LR}_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log \
            >> experiments/020_revision/${RESULT_DIR}/${DATA}_${SEQ}_cold_lr${LR}_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log 2>&1 &

wait
