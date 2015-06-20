export PATH=/Users/david/torch/install/bin:$PATH

# Download coco data train
# wget -O data/coco/train/dataset.json https://softmaxstorage.blob.core.windows.net/coco-feats/coco_dataset_train.json
# wget -O data/coco/train/googlenet_feats.txt https://softmaxstorage.blob.core.windows.net/coco-feats/googlenet_feats_train.txt

# Download coco data test
# wget -O data/coco/test/dataset.json https://softmaxstorage.blob.core.windows.net/coco-feats/coco_dataset_val.json
# wget -O data/coco/test/googlenet_feats.txt https://softmaxstorage.blob.core.windows.net/coco-feats/googlenet_feats_val.txt

# python scripts/create_vocab.py
# Convert coco train dataset from .txt to .th
image_dir="data/coco/train"
image_pre="googlenet_feats"
dataset_pre="dataset"

if [ ! -f $image_dir/$image_pre.th ]; then
    th scripts/convert_imagefeats_txt.lua $image_dir/$image_pre.txt \
        $image_dir/$image_pre.th
fi

if [ ! -f $image_dir/$dataset_pre.th ]; then
    th scripts/convert_dataset.lua $image_dir/$dataset_pre.json \
        $image_dir/$dataset_pre.th
fi

# Convert coco test dataset from .txt to .th
image_dir="data/coco/test"
image_pre="googlenet_feats"
dataset_pre="dataset"

if [ ! -f $image_dir/$image_pre.th ]; then
    th scripts/convert_imagefeats_txt.lua $image_dir/$image_pre.txt \
        $image_dir/$image_pre.th
fi

if [ ! -f $image_dir/$dataset_pre.th ]; then
    th scripts/convert_dataset.lua $image_dir/$dataset_pre.json \
        $image_dir/$dataset_pre.th
fi

