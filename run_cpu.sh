
./train.sh 1 512 512 0 0.0005 rmsprop 0.5 0.5 1 embedlayer projlayer coco -dropout -gpu_mode > "log_emblayer_goog.txt" &
./train.sh 1 256 256 0 0.0005 rmsprop 0.5 0.5 1 embedlayer projlayer coco -gpu_mode | tee -a "log_projlayer.txt"

./train.sh 1 100 100 0 0.0005 rmsprop 0.5 0.5 1 embedlayer projlayer flickr8k | tee -a "log_emblayer.txt"

62.3/42.5/27.6/17.3

 61.7/42.2/27.5/17.5 epoch 22 beam size 10 no dropout
 63.1/43.5/28.2/17.7 epoch 22 beam size 25 no dropout

with dropout beam size 5
 BLEU = 61.3/42.2/26.9/16.8 (BP=1.000, ratio=1.004, hyp_len=57012, ref_len=56783)
bleu score on val set is 
BLEU = 58.2/38.7/24.0/14.9 (BP=1.000, ratio=1.011, hyp_len=9587, ref_len=9484)
bleu score on test set is 
BLEU = 58.6/38.8/23.5/14.3 (BP=1.000, ratio=1.006, hyp_len=9612, ref_len=9557)

with beam size 10
59.3/38.9/23.7/13.9

./train.sh 1 256 256 0 0.0005 rmsprop 0.1 0.1 1 embedlayer projlayer flickr8k | tee -a "log_emblayer.txt"

Epoch 25
Beam size 10 
62.6/43.0/28.4/18.5

Beam size 25
63.8/44.0/29.0/18.9

./train.sh 1 256 256 0 0.0005 rmsprop 0.5 0.5 1 embedlayer projlayer coco -gpu_mode | tee -a "log_projlayer.txt"



