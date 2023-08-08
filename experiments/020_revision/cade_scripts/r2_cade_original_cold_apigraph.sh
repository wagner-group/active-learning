#! /bin/bash

#SBATCH -t 48:00:00

#SBATCH -n 1

#SBATCH -c 8

CNT=$1

modeldim="512-128-32-7"
START="2013-01"
END="2018-12"
S='triplet'
B=96
E=250
LOSS='triplet-mse'
CLS='svm'
FEAT='input'
TS=$(date "+%m.%d-%H.%M.%S")

python -u relabel.py	                                \
        --data gen_apigraph_drebin                      \
        --mdate 20230501                                \
        --train_start 2012-01                           \
        --train_end 2012-12                             \
        --test_start ${START}                           \
        --test_end ${END}                               \
        --encoder cae                                   \
        --loss_func ${LOSS}                             \
        --enc-hidden ${modeldim}                        \
        --sampler ${S}                                  \
        --bsize ${B}                                    \
        --optimizer "adam"                              \
        --learning_rate 0.0001                          \
        --epoch ${E}                                    \
        --cae-lambda 0.1                                \
        --display-interval 300                          \
        --classifier ${CLS}                             \
        --cls-feat ${FEAT}                              \
        --al                                            \
        --count ${CNT}                                  \
        --ood                                           \
        --cold-start                                    \
        --encoder-retrain                               \
        --result experiments/020_revision/cade_results/r2_cade_apigraph_original_cold_${LOSS}_${S}_bsize${B}_e${E}_${CLS}_feat_${FEAT}_test_${START}_${END}_cnt${CNT}.csv \
        --log_path experiments/020_revision/cade_results/r2_cade_apigraph_original_cold_${LOSS}_${S}_bsize${B}_e${E}_${CLS}_feat_${FEAT}_test_${START}_${END}_cnt${CNT}_${TS}.log \
        >> experiments/020_revision/cade_results/r2_cade_apigraph_original_cold_${LOSS}_${S}_bsize${B}_e${E}_${CLS}_feat_${FEAT}_test_${START}_${END}_cnt${CNT}_${TS}.log 2>&1 &

wait