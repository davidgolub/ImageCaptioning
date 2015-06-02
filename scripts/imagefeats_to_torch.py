"""
Converts cocodataset image features into 
torch-compatible format
"""

from __future__ import print_function
import urllib2
import sys
import os
import shutil
import zipfile
import gzip

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    u = urllib2.urlopen(url)
    f = open(filepath, 'wb')
    filesize = int(u.info().getheaders("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
            ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath

def convert_imagefeats(dirpath, feats_path, save_path):
    files = pickle.load(open(dirpath + feats_path, "rb" ) ) 
    with open(save_path,"w") as f:
        f.write("\n".join(" ".join(map(str, x)) for x in (a,b)))


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # data
    train_dir = os.path.join(base_dir, 'data/coco/train')
    test_dir = os.path.join(base_dir, 'data/coco/test')

    convert_imagefeats(train_dir, "googlenet_feats.cPickle", "googlenet_feats.txt")
    convert_imagefeats(test_dir, "googlenet_feats.cPickle", "googlenet_feats.txt")

