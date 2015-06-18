--[[

  Hidden Layer base class

--]]

local InputLayer = torch.class('imagelstm.InputLayer')

function InputLayer:__init(config)
  assert(config.emb_dim ~= nil)
  assert(config.num_classes ~= nil)
  assert(config.dropout_prob ~= nil)
  assert(config.image_dim ~= nil)

  self.gpu_mode = config.gpu_mode or false
  self.emb_dim = config.emb_dim or 300
  self.vocab_size = config.num_classes or 300
  self.dropout_prob = config.dropout_prob or 0.5
  if config.emb_vecs ~= nil then
    self.vocab_size = config.emb_vecs:size(1)
  end
  self.image_dim = config.image_dim or 1024
  self.dropout = config.dropout and false or config.dropout
end

-- Returns all of the weights of this module
function InputLayer:getWeights()
	error("Get weights not implemented!")
end

-- Sets gpu mode
function InputLayer:set_gpu_mode()
	error("Set gpu mode not implemented!")
end

function InputLayer:set_cpu_mode()
	error("Set cpu mode not implemented!")
end

-- Enable Dropouts
function InputLayer:enable_dropouts()
	error("Enable dropouts not implemented!")
end

-- Disable Dropouts
function InputLayer:disable_dropouts()
	error("Disable dropouts not implemented!")
end


-- Does a single forward step of concat layer, concatenating
-- Input 
function InputLayer:forward(word_indeces, image_feats)
	error("Forward not implemented!")
end

function InputLayer:backward(word_indices, image_feats, err)
	error("Backward not implemented!")
end

-- Returns size of outputs of this combine module
function InputLayer:getOutputSize()
	error("Get output size not implemented!")
end

function InputLayer:getParameters()
	error("Get parameters not implemented!")
end

-- zeros out the gradients
function InputLayer:zeroGradParameters() 
	error("Zero grad parameters not implemented!")
end

function InputLayer:getModules() 
  error("Get modules not implemented!")
end

function InputLayer:normalizeGrads(batch_size)
	error("Normalize grads not implemented!")
end


