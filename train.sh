export PATH=/Users/david/torch/install/bin:$PATH

th image_captioning/main.lua \
-model_epoch 1 \
-load_model \
-batch_size 33 \
-mem_dim 150 \
-epochs 300 \
-combine_module concatprojlayer \
-learning_rate 1.5 \
-emb_learning_rate 1.5 \
| tee -a "log_concatlayer.txt"