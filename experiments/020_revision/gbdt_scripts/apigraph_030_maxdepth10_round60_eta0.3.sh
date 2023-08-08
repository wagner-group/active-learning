SEQ=030
DEPTH=10
ROUND=60
ETA=0.3

CNT=$1
START="2013-01"
END="2018-12"
TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u relabel.py	                                \
            --data gen_apigraph_drebin                      \
            --mdate 20230501                                \
            --benign_zero                                   \
            --train_start 2012-01                           \
            --train_end 2012-12                             \
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
            --result experiments/020_revision/gbdt_results/apigraph_${SEQ}_maxdepth${DEPTH}_round${ROUND}_eta${ETA}_cnt${CNT}.csv \
            --log_path experiments/020_revision/gbdt_results/apigraph_${SEQ}_maxdepth${DEPTH}_round${ROUND}_eta${ETA}_cnt${CNT}_${TS}.log \
            >> experiments/020_revision/gbdt_results/apigraph_${SEQ}_maxdepth${DEPTH}_round${ROUND}_eta${ETA}_cnt${CNT}_${TS}.log &
        