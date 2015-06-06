
# Convert coco dataset from cPickle to .txt
cd predictions/bleu 
./multi-bleu.pl gold_standard < output.pred
./multi-bleu-karpathy.pl gold_standard < output.pred
cd ../../