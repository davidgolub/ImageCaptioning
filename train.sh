export PATH=/Users/david/torch/install/bin:$PATH

th image_captioning/main.lua \
-batch_size 33 \
-mem_dim 150 \
-emb_dim 50 \
-epochs 100 \
-num_layers 1 \
-combine_module singleaddlayer \
-learning_rate 0.1 \
-optim rmsprop \
| tee -a "log_singeaddlayer.txt"