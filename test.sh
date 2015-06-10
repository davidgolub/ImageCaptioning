
# Convert coco dataset from cPickle to .txt
cd predictions/bleu 
echo 'bleu score on train set is '
./multi-bleu.pl gold_standard_train < output_train.pred

echo 'bleu score on val set is '
./multi-bleu.pl gold_standard_val < output_val.pred

echo 'bleu score on test set is '
./multi-bleu.pl gold_standard_test < output_test.pred
cd ../../