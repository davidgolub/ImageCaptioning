./train.sh 2 600 600 0 0.001 rmsprop 0.5 0.5 5 concatprojlayer -gpu_mode -dropout | tee -a "log_emblayer.txt"
