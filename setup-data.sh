# Download flickr8k data
wget -P data/flickr8k/ https://bitbucket.org/softmaxinc/image-captioner/downloads/dataset.json
wget -P data/flickr8k/ https://bitbucket.org/softmaxinc/image-captioner/downloads/googlenet_feats.json

# Download coco data train
wget -P data/coco/train -O dataset.json https://softmaxstorage.blob.core.windows.net/coco_feats/coco_dataset_train.json
wget -P data/coco/train -O googlenet_feats.cPickle ttps://softmaxstorage.blob.core.windows.net/coco_feats/googlenet_feats_train.cPickle

# Download coco data test
wget -P data/coco/test -O dataset.json https://softmaxstorage.blob.core.windows.net/coco_feats/coco_dataset_test.json
wget -P data/coco/test -O googlenet_feats.cPickle ttps://softmaxstorage.blob.core.windows.net/coco_feats/googlenet_feats_test.cPickle

# Preprocess captioning data
python scripts/download.py
python scripts/create_vocab.py

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

# Convert features to torch format
image_dir="data/coco"
image_pre="train.txt"

if [ ! -f $image_dir/$image_pre.th ]; then
    th scripts/convert_imagefeats.lua $image_dir/$image_pre.txt \
        $image_dir/$image_pre.th
fi

# Install dependencies
luarocks install torch7
luarocks install nn
luarocks install nngraph
luarocks install optim
luarocks install cutorch
luarocks install cunn
luarocks install json