--[[

  ImageCaption-LSTM training script for image caption generation

--]]

require('.')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Image Captioning Parameters')
cmd:text('Options')
cmd:option('-gpu_mode', false, 'gpu mode')
cmd:option('-epochs', 10,'number of epochs')
cmd:option('-load_model', false, 'load model')
cmd:option('-batch_size', 33, 'batch_size')
cmd:option('-image_dim', 1024, 'input image size into captioner')
cmd:option('-mem_dim', 150,'memory dimension of captioner')
cmd:option('-learning_rate', 0.01, 'learning rate')
cmd:option('-emb_learning_rate', 0.005, 'embedding learning rate')
cmd:option('-data_dir', 'data/flickr8k/', 'directory of caption dataset')
cmd:option('-emb_dir', 'data/glove/', 'director of word embeddings')
cmd:option('-combine_module', 'addlayer', 'type of input layer')
cmd:text()

-- parse input params
params = cmd:parse(arg)

local use_gpu_mode = params.gpu_mode or false
local num_epochs = params.epochs

if use_gpu_mode then 
  print("Loading gpu modules")
  require('cutorch') -- uncomment for GPU mode
  require('cunn') -- uncomment for GPU mode
end

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

header('Image-Captioning with LSTMs')

-- directory containing dataset files
local data_dir = params.data_dir

-- load vocab
local vocab = imagelstm.Vocab(data_dir .. 'vocab.txt')

-- load embeddings
print('loading word embeddings')
local emb_dir = params.emb_dir
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
local train_dir = data_dir
train_dataset = imagelstm.read_caption_dataset(train_dir, vocab, params.gpu_mode)

-- initialize model
model = imagelstm.ImageCaptioner{
  batch_size = params.batch_size,
  emb_vecs = vecs,
  combine_module = params.combine_module,
  learning_rate = params.learning_rate,
  emb_learning_rate = params.emb_learning_rate,
  image_dim = params.image_dim,
  mem_dim = params.mem_dim,
  num_classes = vocab.size + 3, --For start, end and unk tokens
  gpu_mode = use_gpu_mode -- Set to true for GPU mode
}

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

local model_save_path = string.format(
  imagelstm.models_dir .. '/image_captioning_lstm.%d.%d.th', model.mem_dim, 1)

model = imagelstm.ImageCaptioner.load(model_save_path) -- uncomment to load model

print("Predicting on dataset")
prediction = model:predict(train_dataset.image_feats[1], 1)