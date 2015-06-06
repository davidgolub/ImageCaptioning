export PATH=/Users/david/torch/install/bin:$PATH

th image_captioning/main.lua \
-batch_size 33 \
-mem_dim $1 \
-emb_dim $2 \
-epochs 100 \
-num_layers 1 \
-combine_module singleaddlayer \
-learning_rate $3 \
$4 $5\
-optim rmsprop \
| tee -a "log_singeaddlayer.txt"

# First argument is memory dimensions
# Second argument is embed dimensions
# Third argument is learning rate
# Fourth argument is dropout
# Fifth argument is gpu_mode
# -dropout \
# -gpu_mode \
