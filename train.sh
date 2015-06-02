export PATH=/Users/david/torch/install/bin:$PATH

th image_captioning/main.lua \
-load_model \
-epochs 289 \
-batch_size 33 \
-mem_dim 150 \
-emb_dim 50 \
-dropout \
-combine_module singleaddlayer \
-learning_rate 0.1 \
-num_layers 1 \
-optim rmsprop \
| tee -a "log_singeaddlayer.txt"