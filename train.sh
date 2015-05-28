export PATH=/Users/david/torch/install/bin:$PATH

th image_captioning/main.lua \
-batch_size 50 \
-mem_dim 50 \
-emb_dim 50 \
-epochs 300 \
-combine_module addlayer \
-learning_rate 0.5 \
-optim rmsprop \
-gpu_mode
| tee -a "log_concatlayer.txt"