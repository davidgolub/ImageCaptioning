th image_captioning/main.lua \
-load_model \
-batch_size 33  \
-mem_dim 150 \
-epochs 100 \
-combine_module addlayer \
-learning_rate 0.5 \
-emb_learning_rate 0.1 \
| tee -a "log_3.txt"