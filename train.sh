export PATH=/Users/david/torch/install/bin:$PATH

th image_captioning/main.lua \
-batch_size 1 \
-mem_dim 1 \
-epochs 300 \
-combine_module concatprojlayer \
-learning_rate 1.5 \
-emb_learning_rate 1.5 \
-gpu_mode
| tee -a "log_concatlayer.txt"