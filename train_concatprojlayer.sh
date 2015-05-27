export PATH=/Users/david/torch/install/bin:$PATH

th image_captioning/main.lua \
-batch_size 33 \
-mem_dim 150 \
-optim rmsprop \
-epochs 300 \
-combine_module concatprojlayer \
-learning_rate 0.1 \
-emb_learning_rate 0.01 \
| tee -a "log_concatprojlayer_gpu.txt"