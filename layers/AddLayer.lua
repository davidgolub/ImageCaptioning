--[[

  Add layer: adds image features and word embeddings together
  at each step
--]]

local AddLayer = torch.class('imagelstm.AddLayer')

function AddLayer:__init(config)
   self.gpu_mode = config.gpu_mode or false
   self.emb_learning_rate  = config.emb_learning_rate or 0.01
   self.emb_dim = config.emb_vecs:size(2)
   self.image_dim = config.image_dim or 1024
   self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)

   -- image feature embedding
   self.image_emb = nn.Linear(self.image_dim, self.emb_dim)

   -- Do a linear combination of image and word features
   local x1 = nn.Identity(self.emb_dim)()
   local x2 = nn.Identity(self.emb_dim)()
   local a = imagelstm.CRowAddTable()({x1, x2})
  
   self.lstm_emb = nn.gModule({x1, x2}, {a})

   self.emb.weight:copy(config.emb_vecs)
  
   local modules = nn.Parallel()
    :add(self.image_emb)
    :add(self.emb)

   self.params, self.grad_params = modules:getParameters()

   if self.gpu_mode then 
    self:set_gpu_mode()
   end
   -- Copy the image embedding vectors
   if config.combine_weights ~= nil then
     self.params.weight:copy(config.image_weights)
   end
end

-- Sets gpu mode
function AddLayer:set_gpu_mode()
  self.image_emb:cuda()
  self.emb:cuda()
  self.lstm_emb:cuda()
  self.params:cuda()
end

-- Returns all of the weights of this module
function AddLayer:getWeights()
  return self.params
end

-- Does a single forward step of add layer
function AddLayer:forward(word_indeces, image_feats)
    self.text_inputs = self.emb:forward(word_indeces)
    self.image_inputs = self.image_emb:forward(image_feats)
    self.inputs = self.lstm_emb:forward({self.text_inputs, self.image_inputs})

    return self.inputs
end

function AddLayer:backward(word_indices, image_feats, grads)
  -- backprop the gradients through the linear combination step
  local input_emb_grads = self.lstm_emb:backward({self.text_inputs, self.image_inputs}, grads)
  local emb_grads = input_emb_grads[1]
  local image_grads = input_emb_grads[2]

  self.emb:backward(word_indices, emb_grads)
  self.image_emb:backward(image_feats, image_grads)
end

function AddLayer:getOutputSize()
  return self.emb_dim
end

function AddLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function AddLayer:zeroGradParameters() 
  self.grad_params:zero()
  self.image_emb:zeroGradParameters()
  self.emb:zero()
  self.lstm_emb:zeroGradParameters()
end

function AddLayer:normalizeGrads(batch_size)
  self.image_emb.gradWeight:div(batch_size)
  self.emb.gradWeight:div(batch_size)
  self.lstm_emb.gradWeight:div()
end

function AddLayer:updateParameters()
    self.emb:updateParameters(self.emb_learning_rate)
    self.image_emb:updateParameters(self.emb_learning_rate)
end
