"""
Preprocessing script for SICK data.

"""

import json
import os
import glob
import time

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def build_vocab(dataset_path, dst_path, word_count_threshold = 5):
  dataset = json.load(open(dataset_path, 'r'))
  sentence_iterator = dataset['images']

  # count up all word counts so that we can threshold
  # this shouldnt be too expensive of an operation
  print ('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, ))
  t0 = time.time()
  word_counts = {}
  nsents = 0
  for sentences in sentence_iterator:
    for sent in sentences['sentences']:
      nsents += 1
      for w in sent['tokens']:
        word_counts[w] = word_counts.get(w, 0) + 1

  vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
  print ('filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0))

  with open(dst_path, 'w') as f:
    for w in sorted(vocab):
      f.write(w + '\n')
      
  print ('saved vocabulary to %s' % dst_path)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing Caption dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    flickr8k_dir = os.path.join(data_dir, 'flickr8k')

    # get vocabulary
    build_vocab(
        os.path.join(flickr8k_dir, 'dataset.json'),
        os.path.join(flickr8k_dir, 'vocab.txt'))

