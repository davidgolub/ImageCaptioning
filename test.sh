train_file=${1-gold_standard_train}
val_file=${2-gold_standard_val}
test_file=${3-gold_standard_test}

# Do flickr8k predictions
cd predictions/bleu/flickr8k
echo 'FLICKR8k ===============   ' 
echo 'bleu score on train set is '
./multi-bleu.pl $train_file < output_train.pred

echo 'bleu score on val set is '
./multi-bleu.pl $val_file < output_val.pred

echo 'bleu score on test set is '
./multi-bleu.pl $test_file < output_test.pred
cd ../../../

# Do coco predictions
cd predictions/bleu/coco
echo 'COCO ===============   ' 
echo 'bleu score on train set is '
./multi-bleu.pl $train_file < output_train.pred

echo 'bleu score on val set is '
./multi-bleu.pl $val_file < output_val.pred

echo 'bleu score on test set is '
./multi-bleu.pl $test_file < output_test.pred
cd ../../../

