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
cmd:option('-epochs', 10,'number of epochs')
cmd:option('-load_model', false, 'load model')
cmd:option('-batch_size', 33, 'batch_size')
cmd:text()

-- parse input params
params = cmd:parse(arg)

print (params.gpu)
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
local data_dir = 'data/flickr8k/'

-- load vocab
local vocab = imagelstm.Vocab(data_dir .. 'vocab.txt')

-- load embeddings
print('loading word embeddings')
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
print('loading datasets')
local train_dir = data_dir
local train_dataset = imagelstm.read_caption_dataset(train_dir, vocab, config.gpu_mode)

collectgarbage()
printf('num train = %d\n', train_dataset.size)

-- initialize model
local model = imagelstm.ImageCaptioner{
  batch_size = config.batch_size,
  emb_vecs = vecs,
  num_classes = vocab.size + 3, --For start, end and unk tokens
  gpu_mode = use_gpu_mode -- Set to true for GPU mode
}

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

local model_save_path = string.format(
  imagelstm.models_dir .. '/image_captioning_lstm.%d.%d.th', model.mem_dim, 10)

if params.load_model then
  model = imagelstm.ImageCaptioner.load(model_save_path) -- uncomment to load model
end

-- train
local train_start = sys.clock()
local best_train_score = -1.0
local best_train_model = model

local loss = 0.0
header('Training Image Captioning LSTM')
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  loss = model:train(train_dataset)
  printf("Average loss %.4f \n", loss)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)
  
  model_save_path = string.format(
  imagelstm.models_dir .. '/image_captioning_lstm.%d.%d.th', model.mem_dim, i)

  model:save(model_save_path)

  local train_predictions = model:predict_dataset(train_dataset)
  printf('-- predicting sentences on a sample set of 100')

  for j = 1, #train_predictions do
    prediction = train_predictions[j]
    sentence = vocab:tokens(prediction)
    print(sentence)
  end

end

local gold_save_path = string.format(
  imagelstm.predictions_dir .. '/gold_standard.txt')


-- evaluate
header('Evaluating on test set')
printf('-- using model with train score = %.4f\n', loss)
local test_predictions = model:predict_dataset(train_dataset)

if lfs.attributes(imagelstm.predictions_dir) == nil then
  lfs.mkdir(imagelstm.predictions_dir)
end
local predictions_save_path = string.format(
  imagelstm.predictions_dir .. '/imagecaptioning-lstm.%d.pred', model.mem_dim)

local predictions_file, err = io.open(predictions_save_path,"w")

print('writing predictions to ' .. predictions_save_path)
for i = 1, #test_predictions do
  local test_prediction = test_predictions[i][1]
  local likelihood = test_prediction[1]
  local tokens = test_prediction[2]
  local sentence = table.concat(vocab:tokens(tokens), ' ')
  predictions_file:write(sentence .. '\n')
end
predictions_file:close()

-- write model to disk
if lfs.attributes(imagelstm.models_dir) == nil then
  lfs.mkdir(imagelstm.models_dir)
end
local model_save_path = string.format(
  imagelstm.models_dir .. '/image_captioning_lstm.%d.%d.th', model.mem_dim, num_epochs + 1)
print('writing model to ' .. model_save_path)
model:save(model_save_path)

-- to load a saved model
-- local loaded = imagelstm.ImageCaptioner.load(model_save_path)
