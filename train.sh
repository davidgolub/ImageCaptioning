export PATH=/Users/david/torch/install/bin:$PATH

emb_dim=${2-50}
mem_dim=${1-100}
num_layers=${3-1}

learning_rate=${5-0.1}
optim_method=${6-adagrad}
regularization=${4-1e-5}
in_dropout_prob=${7-0.2}
hidden_dropout_prob=${8-0.5}
beam_size=${9-6}
combine_module=${10-embedlayer}
dropout=${11-}
gpu_mode=${12-}

th image_captioning/main.lua \
-batch_size 100 \
-dataset coco \
-mem_dim $mem_dim \
-emb_dim $emb_dim \
-epochs 100 \
-num_layers $num_layers \
-combine_module $combine_module \
-hidden_module projlayer \
-in_dropout_prob $in_dropout_prob \
-hidden_dropout_prob $hidden_dropout_prob \
-learning_rate $learning_rate \
-beam_size $beam_size \
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
