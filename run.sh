./train.sh 1 256 256 0 0.005 rmsprop 0.5 0.5 5 embedlayer flickr8k -dropout -gpu_mode| tee -a "log_emblayer.txt"
./train.sh 1 256 256 0 0.01 rmsprop 0.5 0.5 5 concatprojlayer flickr8k -dropout -gpu_mode| tee -a "log_concatprojlayer.txt"

