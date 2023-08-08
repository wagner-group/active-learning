# Continuous Learning for Android Malware Detection (USENIX Security 2023)

Yizheng Chen, Zhoujie Ding, and David Wagner

Paper: https://arxiv.org/abs/2302.04332

## Datasets

Download [this](https://drive.google.com/file/d/1O0upEcTolGyyvasCPkZFY86FNclk29XO/view?usp=drive_link) from Google Drive. The zipped file contains DREBIN features of the APIGraph dataset and AndroZoo dataset we used in the paper.

Extract the downloaded file to `data/`, such that the datasets are under `data/gen_apigraph_drebin` and `data/gen_androzoo_drebin`.

* We collected `data/gen_apigraph_drebin` by downloading the sample hashes released by the APIGraph paper. The samples are from 2012 to 2018.
* We collected `data/gen_androzoo_drebin` by downloading apps from AndroZoo. The samples are from 2019 to 2021.

## Example Active Learning Run

The following example trains an `enc-mlp` model using `hi-dist-xent` loss, i.e., our Hierarchical Contrastive Classifier, and it runs active learning with 200 samples / month budget using our Psuedo Loss Sample Selector.

```
#! /bin/bash

SEQ=088
LR=0.003
OPT=sgd
SCH=step
DECAY=0.95
E=250
WLR=0.00015
WE=100
DATA=gen_apigraph_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2018-12
RESULT_DIR=results_ours
AL_OPT=adam

CNT=200

modeldim="512-384-256-128"
S='half'
B=1024
LOSS='hi-dist-xent'
TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u relabel.py	                                \
            --data ${DATA}                                  \
            --benign_zero                                   \
            --mdate 20230501                                \
            --train_start ${TRAIN_START}                    \
            --train_end ${TRAIN_END}                        \
            --test_start ${TEST_START}                      \
            --test_end ${TEST_END}                          \
            --encoder simple-enc-mlp                        \
            --classifier simple-enc-mlp                     \
            --loss_func ${LOSS}                             \
            --enc-hidden ${modeldim}                        \
            --mlp-hidden 100-100                            \
            --mlp-dropout 0.2                               \
            --sampler ${S}                                  \
            --bsize ${B}                                    \
            --optimizer ${OPT}                              \
            --scheduler ${SCH}                              \
            --learning_rate ${LR}                           \
            --lr_decay_rate ${DECAY}                        \
            --lr_decay_epochs "10,500,10"                   \
            --epochs ${E}                                   \
            --encoder-retrain                               \
            --al_optimizer ${AL_OPT}                        \
            --warm_learning_rate ${WLR}                     \
            --al_epochs ${WE}                               \
            --xent-lambda 100                               \
            --display-interval 180                          \
            --al                                            \
            --count ${CNT}                                  \
            --local_pseudo_loss                             \
            --reduce "none"                                 \
            --sample_reduce 'mean'                          \
            --result experiments/020_revision/${RESULT_DIR}/gen_apigraph_cnt${CNT}_${SEQ}_warm_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_${AL_OPT}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}.csv \
            --log_path experiments/020_revision/${RESULT_DIR}/gen_apigraph_cnt${CNT}_${SEQ}_warm_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_${AL_OPT}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log \
            >> experiments/020_revision/${RESULT_DIR}/gen_apigraph_cnt${CNT}_${SEQ}_warm_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_${AL_OPT}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log 2>&1 &
```

### Example Scripts

We used the scripts under `experiments/020_revision` to run experiments in the paper. We ran these jobs on a Slurm GPU cluster (thanks to Center for AI Safety). If you would like to run the same script on a GPU server, not managed by Slurm, you would need to remove the lines starting with `#SBATCH` and also the last line (i.e. `wait`).