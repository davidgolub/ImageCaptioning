export PATH=/Users/david/torch/install/bin:$PATH

th image_captioning/main.lua \
-batch_size 33 \
-mem_dim 250 \
-emb_dim 150 \
-epochs 100 \
-num_layers 1 \
-combine_module singleaddlayer \
-learning_rate 0.1 \
-optim rmsprop \
| tee -a "log_singeaddlayer.txt"
# -dropout \
# -gpu_mode \
