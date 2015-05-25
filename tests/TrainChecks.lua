--[[

  Image Captioning using an LSTM and a softmax for the language model

--]]

local TrainChecks = torch.class('imagelstm.TrainChecks')

function TrainChecks:__init(config)
 -- directory containing dataset files
  local data_dir = 'data/flickr8k/'

  -- load vocab
  local vocab = imagelstm.Vocab(data_dir .. 'vocab.txt')

  -- load embeddings
  print('Loading word embeddings')
  local emb_dir = 'data/glove/'
  local emb_prefix = emb_dir .. 'glove.840B'
  local emb_vocab, emb_vecs = imagelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
  local emb_dim = emb_vecs:size(2)

  -- use only vectors in vocabulary (not necessary, but gives faster training)
  local num_unk = 0
  local vecs = torch.Tensor(vocab.size, emb_dim)
  for i = 1, vocab.size do
    local w = string.gsub(vocab:token(i), '\\', '') -- remove escape characters
    if emb_vocab:contains(w) then
      vecs[i] = emb_vecs[emb_vocab:index(w)]
    else
      num_unk = num_unk + 1
      vecs[i]:uniform(-0.05, 0.05)
    end
  end
  print('unk count = ' .. num_unk)
  emb_vocab = nil
  emb_vecs = nil
  collectgarbage()

  -- load datasets
  print('Loading datasets')
  local train_dir = data_dir
  self.train_dataset = imagelstm.read_caption_dataset(train_dir, vocab, params.gpu_mode)

  collectgarbage()
  printf('Num train = %d\n', self.train_dataset.size)

  -- initialize model
  self.model = imagelstm.ImageCaptioner{
    batch_size = 33,
    emb_vecs = vecs,
    num_classes = vocab.size + 3, --For start, end and unk tokens
    gpu_mode = config.gpu_mode or false
  }

end

function TrainChecks:check_train()
  self.model:train(self.train_dataset)
end




