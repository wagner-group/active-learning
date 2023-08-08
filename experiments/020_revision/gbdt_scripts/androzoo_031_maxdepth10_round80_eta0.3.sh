SEQ=031
DEPTH=10
ROUND=80
ETA=0.3

CNT=$1
START="2020-01"
END="2021-12"
TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u relabel.py	                                \
            --data gen_androzoo_drebin                      \
            --mdate 20230501                                \
            --benign_zero                                   \
            --train_start 2019-01                           \
            --train_end 2019-12                             \
            --test_start ${START}                           \
            --test_end ${END}                               \
            --classifier gbdt                               \
            --max_depth ${DEPTH}                            \
            --num_round ${ROUND}                            \
            --eta ${ETA}                                    \
            --cls-retrain 1                                 \
            --al                                            \
            --unc                                           \
            --count ${CNT}                                  \
            --result experiments/020_revision/gbdt_results/androzoo_${SEQ}_maxdepth${DEPTH}_round${ROUND}_eta${ETA}_cnt${CNT}.csv \
            --log_path experiments/020_revision/gbdt_results/androzoo_${SEQ}_maxdepth${DEPTH}_round${ROUND}_eta${ETA}_cnt${CNT}_${TS}.log \
            >> experiments/020_revision/gbdt_results/androzoo_${SEQ}_maxdepth${DEPTH}_round${ROUND}_eta${ETA}_cnt${CNT}_${TS}.log 2>&1 &