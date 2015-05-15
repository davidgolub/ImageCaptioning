# Download data
wget -P data/flickr8k/ https://bitbucket.org/softmaxinc/image-captioner/downloads/dataset.json
wget -P data/flickr8k/ https://bitbucket.org/softmaxinc/image-captioner/downloads/googlenet_feats.json

# Preprocess captioning data
python scripts/download.py
python scripts/create_vocab.py

# Install dependencies
luarocks install torch7
luarocks install nn
luarocks install nngraph
luarocks install optim
luarocks install cutorch
luarocks install cunn
luarocks install json

# Preprocess glove vectors
glove_dir="data/glove"
glove_pre="glove.840B"
glove_dim="300d"
if [ ! -f $glove_dir/$glove_pre.$glove_dim.th ]; then
    th scripts/convert_wordvecs.lua $glove_dir/$glove_pre.$glove_dim.txt \
        $glove_dir/$glove_pre.vocab $glove_dir/$glove_pre.$glove_dim.th
fi

# Convert features to torch format
image_dir="data/flickr8k"
image_pre="googlenet_feats"

if [ ! -f $image_dir/$image_pre.th ]; then
    th scripts/convert_imagefeats.lua $image_dir/$image_pre.json \
        $image_dir/$image_pre.th
fi