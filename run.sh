./train.sh 600 600 2 0 0.01 adagrad 0.5 0.5 -dropout -gpu_mode | tee -a "log_emblayer.txt"
# ./train.sh 256 256 1 0 0.01 rmsprop 0.2 0.5 6 -dropout -gpu_mode | tee -a "log_emblayer.txt"
# ./train.sh 256 256 1 0 0.01 rmsprop 0.4 0.5 6 -dropout -gpu_mode | tee -a "log_emblayer.txt"
