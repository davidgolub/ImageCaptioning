--[[

  Functions for loading data from disk.

--]]
  
function imagelstm.read_embedding(vocab_path, emb_path)
  -- Reads vocabulary from vocab_path, 
  -- Reads word embeddings from embd_path
  local vocab = imagelstm.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function imagelstm.read_sentences(path, vocab)
  -- Reads sentences from specified path
  -- Reads vocab from vocab path
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    sentences[#sentences + 1] = sent
  end

  file:close()
  return sentences
end


--[[

  Functions for loading caption data from disk.

--]]
function imagelstm.read_dataset(dataset_path)
  -- Reads image features from disk
  -- Returns vocab, embeddings

  print('Reading caption dataset from ' .. dataset_path)
  local dataset = json.load(dataset_path)
  local images = dataset['images']

  -- TODO: preprocess images
  print('Done reading caption dataset from ' .. dataset_path)
  return images
end

function imagelstm.read_image_features(feature_path)
  -- Reads all image features from file
  -- Returns Torch tensor of size (image_feature_size)

  print('Reading image features from ' .. feature_path)
  vecs = torch.load(feature_path)
  print('Done reading image features from ' .. feature_path)

  return vecs
end

--[[

 Read captions dataset

--]]


--[[

 Read captions dataset

--]]
function imagelstm.read_caption_sentences(dir, vocab)
  local caption_dataset = {}

  local annotation_dataset = imagelstm.read_dataset(dir .. 'dataset.json')
  local num_images = #annotation_dataset

  -- get input and output sentences
  local sentences = {}
  local out_sentences = {}
  local image_ids = {}
  for i = 1, num_images do
    local curr_image = annotation_dataset[i]

    -- dataset is zero indexed, torch is 1 indexed
    local curr_imgid = curr_image['imgid'] + 1
    local curr_sentences = curr_image['sentences']

    local tokens = curr_sentences[1]['tokens']
    table.insert(tokens, "</s>")
    table.insert(sentences, tokens)
  end
  
  caption_dataset.sentences = sentences
  caption_dataset.size = #sentences
  return caption_dataset
end

function imagelstm.read_caption_dataset(dir, vocab, gpu_mode)
  local caption_dataset = {}

  local annotation_dataset = imagelstm.read_dataset(dir .. 'dataset.json')
  local num_images = #annotation_dataset

  -- get input and output sentences
  local sentences = {}
  local out_sentences = {}
  local image_ids = {}
  for i = 1, num_images do
    local curr_image = annotation_dataset[i]

    -- dataset is zero indexed, torch is 1 indexed
    local curr_imgid = curr_image['imgid'] + 1
    local curr_sentences = curr_image['sentences']

    for j = 1, #curr_sentences do
      local tokens = curr_sentences[j]['tokens']
      table.insert(tokens, "</s>")

      -- first get labels for sentence
      local out_ids = vocab:map(tokens)

      -- then get input sentence stuff: off by one language model
      table.insert(tokens, 1, "<s>")
      table.remove(tokens)
      local in_ids = vocab:map(tokens)
      
      -- then make a new one with special start symbol
      table.insert(image_ids, curr_imgid)
      table.insert(out_sentences, out_ids)
      table.insert(sentences, in_ids)
    end
  end

  image_feats = imagelstm.read_image_features(dir .. 'googlenet_feats.th')
  if gpu_mode then
    image_feats:cuda()
  end

  caption_dataset.vocab = vocab
  caption_dataset.image_feats = image_feats
  caption_dataset.image_ids = image_ids
  caption_dataset.pred_sentences = out_sentences
  caption_dataset.sentences = sentences
  caption_dataset.size = #sentences
  return caption_dataset
end
