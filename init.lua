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

-- Math helper functions
include('util/math.lua')

-- For reading word embeddings, image features, and captions
include('util/Vocab.lua')
include('util/read_data.lua') 

-- New forward/backward layers
include('layers/CRowAddTable.lua')
include('layers/CRowSingleTable.lua')
include('layers/CRowJoinTable.lua')

-- New image/word embedding layers
include('layers/AddLayer.lua')
include('layers/SingleAddLayer.lua')
include('layers/ConcatProjLayer.lua')
include('layers/ConcatLayer.lua')

-- models
include('models/ImageCaptioner.lua')
include('models/ImageCaptionerLSTM.lua')
include('models/LSTM.lua')
include('models/LSTM_Full.lua')
include('models/LSTM_Multilayered.lua')

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
