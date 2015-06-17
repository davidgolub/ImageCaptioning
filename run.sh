./train.sh 600 600 2 0 0.01 rmsprop 0.5 0.5 -dropout -gpu_mode | tee -a "log_emblayer.txt"

# ./train.sh 256 256 3 0 0.005 rmsprop -gpu_mode -dropout | tee -a "log_emblayer.txt"
# ./train.sh 256 256 1 0 0.05 -gpu_mode | tee -a "log_emblayer.txt"
# ./train.sh 600 600 2 0 0.05 rmsprop -gpu_mode -dropout | tee -a "log_emblayer.txt"
# ./train.sh 256 256 2 0 0.05 -gpu_mode -dropout | tee -a "log_emblayer.txt"
# ./train.sh 256 256 1 0 0.05 -gpu_mode -dropout | tee -a "log_emblayer.txt"
# ./train.sh 256 256 2 0 0.05 -gpu_mode -dropout | tee -a "log_emblayer.txt"
# ./train.sh 100 150 1 0 0.05 -dropout
# ./train.sh 600 600 1 0 0.01 rmsprop -gpu_mode | tee -a "log_emblayer.txt"
# ./train.sh 1000 1000 1 0 0.01 rmsprop -gpu_mode -dropout| tee -a "log_emblayer.txt"