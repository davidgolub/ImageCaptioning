export PATH=/Users/david/torch/install/bin:$PATH

th image_captioning/main.lua \
-load_model \
-model_epoch 20 \
-optim rmsprop \
-batch_size 33 \
-mem_dim 150 \
-epochs 300 \
-combine_module addlayer \
-learning_rate 0.1 \
-emb_learning_rate 0.001 
# -gpu_mode
| tee -a "log_addlayer_gpu.txt"