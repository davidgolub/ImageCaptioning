export PATH=/Users/david/torch/install/bin:$PATH

th image_captioning/main.lua \
-model_epoch 1 \
-load_model \
-batch_size 33 \
-mem_dim 150 \
-epochs 100 \
-combine_module addlayer \
-learning_rate 0.5 \
-emb_learning_rate 0.05 \
| tee -a "log_3.txt"