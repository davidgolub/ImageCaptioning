--[[

  ImageCaption-LSTM training script for image caption generation

--]]

require('..')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Image Captioning Parameters')
cmd:text('Options')
cmd:option('-gpu_mode', false, 'gpu mode')
cmd:option('-optim', 'rmsprop', 'optimization type')
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
cmd:option('-model_epoch', 98, 'epoch to load model from')
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
print('loading datasets')
local train_dir = data_dir
local train_dataset = imagelstm.read_caption_dataset(train_dir, vocab, params.gpu_mode)

collectgarbage()
printf('num train = %d\n', train_dataset.size)

-- initialize model
local model = imagelstm.ImageCaptioner{
  batch_size = params.batch_size,
  optim_method = optim_method,
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
  imagelstm.models_dir .. '/image_captioning_lstm.%s.%d.%d.th', 
  model.combine_module_type,
  model.mem_dim, params.model_epoch)

if params.load_model then
--if true then
  model = imagelstm.ImageCaptioner.load(model_save_path) -- uncomment to load model
end

if lfs.attributes(imagelstm.models_dir) == nil then
  lfs.mkdir(imagelstm.models_dir)
end

if lfs.attributes(imagelstm.predictions_dir) == nil then
  lfs.mkdir(imagelstm.predictions_dir)
end

-- train
local train_start = sys.clock()
local best_train_score = -1.0
local best_train_model = model

local loss = 0.0
header('Training Image Captioning LSTM')
for i = 1, num_epochs do
  curr_epoch = i
  -- get training predictions
  local train_predictions = model:predict_dataset(train_dataset)
  printf('-- predicting sentences on a sample set of 100\n')

  -- save them to disk for later use
  local predictions_save_path = string.format(
  imagelstm.predictions_dir .. '/image_captioning_lstm.%s.%d.%d.pred', 
  model.combine_module_type, model.mem_dim, i)

  model:save_predictions(predictions_save_path, train_predictions)

  local start = sys.clock()
  printf('-- epoch %d\n', i)
  loss = model:train(train_dataset)
  printf("Average loss %.4f \n", loss)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)
  
  model_save_path = string.format(
  imagelstm.models_dir .. '/image_captioning_lstm.%s.%d.%d.th', 
  model.combine_module_type,
  model.mem_dim, params.model_epoch)

  model:save(model_save_path)
end

local gold_save_path = string.format(
  imagelstm.predictions_dir .. '/gold_standard.txt')


-- evaluate
header('Evaluating on test set')
printf('-- using model with train score = %.4f\n', loss)
local test_predictions = model:predict_dataset(train_dataset)

-- write model to disk

  
local model_save_path = string.format(
  imagelstm.models_dir .. '/image_captioning_lstm.%s.%d.%d.th', 
  model.combine_module_type,
  model.mem_dim, params.model_epoch)

print('writing model to ' .. model_save_path)
model:save(model_save_path)

-- to load a saved model
-- local loaded = imagelstm.ImageCaptioner.load(model_save_path)
