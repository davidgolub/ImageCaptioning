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
cmd:option('-dropout', false, 'use dropout')
cmd:option('-load_model', false, 'load model')
cmd:option('-batch_size', 33, 'batch_size')
cmd:option('-image_dim', 1024, 'input image size into captioner')
cmd:option('-emb_dim', 100, 'embedding size')
cmd:option('-mem_dim', 150,'memory dimension of captioner')
cmd:option('-learning_rate', 0.01, 'learning rate')
cmd:option('-data_dir', 'data/flickr8k/', 'directory of caption dataset')
cmd:option('-emb_dir', 'data/glove/', 'director of word embeddings')
cmd:option('-combine_module', 'addlayer', '[embedlayer] [addlayer] [singleaddlayer] [concatlayer] [concatprojlayer]')
cmd:option('-hidden_module', 'hiddenlayer', '[hiddenlayer]')
cmd:option('-model_epoch', 98, 'epoch to load model from')
cmd:option('-num_layers', 4, 'number of layers in lstm network')
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

-- load datasets

-- load train dataset
local train_dir = data_dir
local train_dataset = imagelstm.read_caption_dataset(train_dir, vocab, params.gpu_mode,
  'train')

-- load val dataset
local val_dir = data_dir
local val_dataset = imagelstm.read_caption_dataset(val_dir, vocab, params.gpu_mode, 'val')

-- load test dataset
local test_dir = data_dir
local test_dataset = imagelstm.read_caption_dataset(test_dir, vocab, params.gpu_mode, 
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
local model = imagelstm.ImageCaptioner{
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

local loss = 0.0

-- evaluates the model on the test set
function evaluate(model, beam_size, dataset, save_path)
  printf('-- using model with train score = %.4f\n', loss)
  --if model.gpu_mode then
  --   model:set_cpu_mode()
  --end

  local test_predictions = model:predict_dataset(dataset, beam_size, dataset.size)

  if model.gpu_mode then
    model:set_gpu_mode()
  end
  print("Saving predictions to ", save_path)
  model:save_predictions(save_path, loss, test_predictions)
end

-- Calculates BLEU scores on train, test and val sets
function evaluate_results()
    -- evaluate
  header('Evaluating on test set')
  evaluate(model, 5, test_dataset, 'predictions/bleu/output_test.pred')

  header('Evaluating on train set')
  evaluate(model, 5, train_dataset, 'predictions/bleu/output_train.pred')

  header('Evaluating on val set')
  evaluate(model, 5, val_dataset, 'predictions/bleu/output_val.pred')

  os.execute("./test.sh")
end

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

local model_save_path = string.format(
  imagelstm.models_dir .. model:getPath(num_epochs))

if params.load_model then
--if true then
  print("Loading model from file " .. model_save_path)
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

    -- save them to disk for later use
local predictions_save_path = string.format(
imagelstm.predictions_dir .. model:getPath(2))

header('Training Image Captioning LSTM')
for i = 1, params.epochs do
  curr_epoch = i

  if curr_epoch % 20 == 1 then
    evaluate_results()
  end

  local start = sys.clock()
  printf('-- epoch %d\n', i)
  loss = model:train(train_dataset)
  printf("Average loss %.4f \n", loss)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)

  local train_loss = model:eval(train_dataset)
  printf("Train loss is %.4f \n", train_loss)

  local test_loss = model:eval(test_dataset)
  printf("Test loss is %.4f \n", test_loss)

  local val_loss = model:eval(val_dataset)
  printf("Val loss is %.4f \n", val_loss)

    -- save them to disk for later use
  local predictions_save_path = string.format(
  imagelstm.predictions_dir .. model:getPath(i))

  local test_predictions = model:predict_dataset(test_dataset, 5, 30)

  print("Saving predictions to ", predictions_save_path)
  model:save_predictions(predictions_save_path, loss, test_predictions)


  model_save_path = string.format(
  imagelstm.models_dir .. model:getPath(i))


  print("Model save path is", model_save_path)
  model:save(model_save_path)
  --model = imagelstm.ImageCaptioner.load(model_save_path)

end

-- write model to disk
  
local model_save_path = string.format(
  imagelstm.models_dir .. model:getPath(params.model_epoch))
print('writing model to ' .. model_save_path)
model:save(model_save_path)

-- to load a saved model
-- local loaded = imagelstm.ImageCaptioner.load(model_save_path)
