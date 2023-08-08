C=0.1

CNT=$1
START="2020-01"
END="2021-12"
TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u relabel.py	                                \
            --data gen_androzoo_drebin                      \
            --mdate 20230517                                \
            --benign_zero                                   \
            --train_start 2019-01                           \
            --train_end 2019-12                             \
            --test_start ${START}                           \
            --test_end ${END}                               \
            --classifier svm                                \
            --multi_class                                   \
            --svm-c ${C}                                    \
            --al                                            \
            --unc                                           \
            --count ${CNT}                                  \
            --result experiments/020_revision/svm_results/multi_androzoo_svmc${C}_test_${START}_${END}_cnt${CNT}.csv \
            --log_path experiments/020_revision/svm_results/multi_androzoo_svmc${C}_test_${START}_${END}_cnt${CNT}_${TS}.log \
            >> experiments/020_revision/svm_results/multi_androzoo_svmc${C}_test_${START}_${END}_cnt${CNT}_${TS}.log &
        