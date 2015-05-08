require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
require('io')
require('json')

imagelstm = {}

print("Starting to load requirements")

include('util/Vocab.lua')
include('util/read_data.lua') -- For reading word embeddings, image features, and captions
include('layers/CRowAddTable.lua')

-- models
include('models/ImageCaptioner.lua')
include('models/ImageCaptionerLSTM.lua')
include('models/LSTM.lua')
include('models/LSTM_Full.lua')

print("Loaded requirements")

printf = utils.printf

-- global paths (modify if desired)
imagelstm.data_dir        = 'data'
imagelstm.models_dir      = 'trained_models'
imagelstm.predictions_dir = 'predictions'

-- share parameters of nngraph gModule instances
function share_params(cell, src, ...)
  for i = 1, #cell.forwardnodes do
    local node = cell.forwardnodes[i]
    if node.data.module then
      node.data.module:share(src.forwardnodes[i].data.module, ...)
    end
  end
end

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end
