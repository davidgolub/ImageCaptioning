--[[

  ImageCaption-LSTM training script for image caption generation

--]]

require('.')

-- parse input params
params = {}
params.batch_size = 33
params.image_dim = 1024
params.emb_dim = 600
params.mem_dim = 600
params.learning_rate = 0.01
params.data_dir = 'data/flickr8k/'
params.emb_dir = 'data/glove/'
params.combine_module = 'embedlayer'
params.hidden_module = 'hiddenlayer'
params.num_layers = 1
params.load_model = true
params.dropout = true
params.num_epochs = 100
params.epochs = 98
params.gpu_mode = true

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

local opt_method
if params.optim == 'adagrad' then
  opt_method = optim.adagrad
else
  opt_method = optim.rmsprop
end

header('Image-Captioning with LSTMs')

-- directory containing dataset files
local data_dir = params.data_dir

-- load vocab
vocab = imagelstm.Vocab(data_dir .. 'vocab.txt')

-- load datasets

-- load train dataset
local train_dir = data_dir
train_dataset = imagelstm.read_caption_dataset(train_dir, vocab, params.gpu_mode,
  'train')

-- load val dataset
local val_dir = data_dir
val_dataset = imagelstm.read_caption_dataset(val_dir, vocab, params.gpu_mode, 'val')

-- load test dataset
local test_dir = data_dir
test_dataset = imagelstm.read_caption_dataset(test_dir, vocab, params.gpu_mode, 
  'test')

-- load embeddings
print('loading word embeddings')
local emb_dir = params.emb_dir
local emb_prefix = emb_dir .. 'glove.840B'
--local emb_vocab, emb_vecs = imagelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
--local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, params.emb_dim)
for i = 1, vocab.size do
  local w = string.gsub(vocab:token(i), '\\', '') -- remove escape characters
  -- if emb_vocab:contains(w) then
     -- vecs[i] = emb_vecs[emb_vocab:index(w)]
  -- else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  --end
end

print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil

collectgarbage()
printf('num train = %d\n', train_dataset.size)

-- initialize model
model = imagelstm.ImageCaptioner{
  batch_size = params.batch_size,
  optim_method = opt_method,
  --emb_vecs = vecs,
  dropout = params.dropout,
  num_layers = params.num_layers,
  num_classes = vocab.size,
  emb_dim = params.emb_dim,
  combine_module = params.combine_module,
  learning_rate = params.learning_rate,
  reg = params.reg,
  image_dim = params.image_dim,
  mem_dim = params.mem_dim,
  num_classes = vocab.size, --For start, end and unk tokens
  gpu_mode = use_gpu_mode -- Set to true for GPU mode
}

-- print information
header('model configuration')
printf('max epochs = %d\n', params.num_epochs)
model:print_config()

local model_save_path = string.format(
  imagelstm.models_dir .. model:getPath(num_epochs))

if params.load_model then
--if true then
  print("Loading model from file " .. model_save_path)
  model = imagelstm.ImageCaptioner.load(model_save_path) -- uncomment to load model
end

--train_predictions = model:get_sentences(model:predict_dataset(train_dataset, 5, 30))
--test_predictions = model:get_sentences(model:predict_ataset(test_dataset, 5, 30))
--val_predictions = model:get_sentences(model:predict_dataset(test_dataset, 5, 30))