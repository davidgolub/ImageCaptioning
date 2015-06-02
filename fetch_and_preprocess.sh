# Convert features to torch format
image_dir="data/flickr8k"
image_pre="googlenet_feats"

if [ ! -f $image_dir/$image_pre.th ]; then
    th scripts/convert_imagefeats.lua $image_dir/$image_pre.json \
        $image_dir/$image_pre.th
fi