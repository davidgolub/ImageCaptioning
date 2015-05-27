export PATH=/Users/david/torch/install/bin:$PATH

th image_captioning/main.lua \
-batch_size 33 \
-optim rmsprop \
-mem_dim 150 \
-epochs 300 \
-combine_module concatlayer \
-learning_rate 0.1 \
-emb_learning_rate 0.1 \
-gpu_mode
| tee -a "log_concatlayer_gpu.txt"