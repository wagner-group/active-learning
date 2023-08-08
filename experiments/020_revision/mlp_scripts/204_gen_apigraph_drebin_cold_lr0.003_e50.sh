#! /bin/bash

#SBATCH -t 00:30:00

#SBATCH -n 1

#SBATCH -c 8

SEQ=204
LR=0.003
E=50
DATA=gen_apigraph_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2013-06
RESULT_DIR=mlp_al_results_multi
MODEL_DATE=20230501


CNT=50
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
            --result experiments/012_base_models/${RESULT_DIR}/${DATA}_${SEQ}_cold_lr${LR}_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}.csv \
            --log_path experiments/012_base_models/${RESULT_DIR}/${DATA}_${SEQ}_cold_lr${LR}_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log \
            >> experiments/012_base_models/${RESULT_DIR}/${DATA}_${SEQ}_cold_lr${LR}_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log 2>&1 &

wait
