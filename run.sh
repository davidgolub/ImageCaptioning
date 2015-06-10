./train.sh 600 600 1 0 0.05 -gpu_mode -dropout | tee -a "log_emblayer.txt"
./train.sh 600 600 1 0 0.05 -gpu_mode
./test.sh


# ./train.sh 100 150 1 0 0.05 -dropout