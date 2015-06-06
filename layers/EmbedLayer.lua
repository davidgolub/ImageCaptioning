--[[

  Embed layer: Simple word embedding layer for input into lstm

--]]

local EmbedLayer = torch.class('imagelstm.EmbedLayer')

function EmbedLayer:__init(config)
   self.emb_learning_rate       = config.emb_learning_rate or 0.01
   self.gpu_mode = config.gpu_mode or false
   self.emb_dim = config.emb_dim or 300
   self.vocab_size = config.num_classes or 300
   if config.emb_vecs ~= nil then
    self.vocab_size = config.emb_vecs:size(1)
   end

   self.emb = nn.LookupTable(self.vocab_size, self.emb_dim)
   self.params, self.grad_params = self.emb:getParameters()

  if gpu_mode then
    self:set_gpu_mode()
  end

  -- Copy embedding weights
  if config.emb_vecs ~= nil then
    self.emb.weight:copy(config.emb_vecs)
  end
  -- Copy the image embedding vectors
  if config.combine_weights ~= nil then
    self.params:copy(config.combine_weights)
  end
end

-- Returns all of the weights of this module
function EmbedLayer:getWeights()
  return self.params
end

-- Sets gpu mode
function EmbedLayer:set_gpu_mode()
  self.emb:cuda()
end


-- Does a single forward step of concat layer, concatenating
-- Input 
function EmbedLayer:forward(word_indeces)
   self.word_proj = self.emb:forward(word_indeces)
   return self.word_proj
end

function EmbedLayer:backward(word_indices, err)
   self.emb:backward(word_indices, err)
end

-- Returns size of outputs of this combine module
function EmbedLayer:getOutputSize()
  return self.emb_dim
end

function EmbedLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function EmbedLayer:zeroGradParameters() 
  self.emb:zeroGradParameters()
end

function EmbedLayer:getModules() 
  return {self.emb}
end

function EmbedLayer:normalizeGrads(batch_size)
  self.emb.gradWeight:div(batch_size)
end

function EmbedLayer:updateParameters()
  -- normalize gradients by batch size
  self.emb:updateParameters(self.emb_learning_rate)
end