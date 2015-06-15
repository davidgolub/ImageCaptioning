export PATH=/Users/david/torch/install/bin:$PATH

emb_dim=${2-50}
mem_dim=${1-100}
num_layers=${3-1}

learning_rate=${5-0.1}
optim_method=${6-adagrad}
regularization=${4-1e-5}
dropout=${7-}
gpu_mode=${8-}

th image_captioning/main.lua \
-batch_size 33 \
-mem_dim $mem_dim \
-emb_dim $emb_dim \
-epochs 100 \
-num_layers $num_layers \
-combine_module embedlayer \
-hidden_module projlayer \
-learning_rate $learning_rate \
$dropout $gpu_mode \
-optim $optim_method

# First argument is memory dimensions
# Second argument is embed dimensions
# Third argument is number of layers
# Fourth argument is learning rate
# Fifth argument is dropout
# Sixth argument is gpu_mode
# -dropout \
#-load_model \
#-model_epoch 98 \
# -gpu_mode \
